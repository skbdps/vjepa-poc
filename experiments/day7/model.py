"""Learned persistent part slots over frozen, projected V-JEPA patch tokens.

The model receives current/past tokens and frame-zero soft part masks only. Each
slot keeps an immutable appearance anchor, a recurrent state, and same-owner
sibling context. A forecast made eight feature steps earlier is explicitly used
in the current retrieval query. Thus the predictor affects inference even when
the auxiliary future-target loss has weight zero.

There are no learned positions, slot IDs, or owner-ID embeddings. Patch order
and slot order are equivariant when inputs and owner assignments are permuted
consistently. Causality is in feature steps: the upstream V-JEPA encoder itself
may attend to future frames inside its independently encoded video window.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)


class PersistentPartJEPA(nn.Module):
    """Recurrent retrieval with an immutable anchor and a queued latent forecast.

    Args:
        input_dim: Dimension of the runner's fixed projected V-JEPA tokens.
        hidden_dim: Learned persistent slot-state dimension.
        horizon: Forecast delay in feature steps (8 for one encoder window).

    Forward inputs:
        features: [B,T,N,D], frozen projected patch features.
        initial_weights: [B,K,N], nonnegative frame-zero part weights.
        owner_ids: [K] or [B,K]. IDs express grouping only, not semantic values.

    Returns a dictionary with location+null logits [B,T,K,N+1], normalized future
    features [B,T,K,D], states [B,T,K,H], visibility probabilities [B,T,K],
    conditional spatial attention [B,T,K,N], and the prior used by retrieval.
    The final logit is absence. The model never receives training target masks,
    future feature targets, or evaluation annotations.
    """

    def __init__(self, input_dim=256, hidden_dim=128, horizon=8):
        super().__init__()
        if min(input_dim, hidden_dim, horizon) <= 0:
            raise ValueError("Dimensions and forecast horizon must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        d, h = self.input_dim, self.hidden_dim
        self.anchor_state = nn.Linear(d, h)
        self.key_residual = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.query_residual = nn.Sequential(nn.Linear(2*d+2*h, d), nn.GELU(), nn.Linear(d, d))
        self.absence = nn.Sequential(nn.Linear(3*d+2*h+3, h), nn.GELU(), nn.Linear(h, 1))
        self.update_input = nn.Sequential(nn.Linear(2*d+h, h), nn.GELU())
        self.memory = nn.GRUCell(h, h)
        self.future_predictor = nn.Sequential(nn.Linear(d+2*h, d), nn.GELU(), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        # A fixed nonzero path makes forecasts part of retrieval in both arms.
        self.forecast_weight = 0.25
        # Begin close to a normalized initial-prototype retrieval baseline.
        nn.init.normal_(self.key_residual[-1].weight, std=0.002)
        nn.init.zeros_(self.key_residual[-1].bias)
        nn.init.normal_(self.query_residual[-1].weight, std=0.002)
        nn.init.zeros_(self.query_residual[-1].bias)
        nn.init.normal_(self.absence[-1].weight, std=0.002)
        nn.init.constant_(self.absence[-1].bias, -2.0)

    @staticmethod
    def _sibling_context(states, owner_ids):
        k = states.shape[1]
        same_owner = owner_ids[:, :, None] == owner_ids[:, None, :]
        sibling = same_owner & ~torch.eye(k, device=states.device, dtype=torch.bool)[None]
        weights = sibling.to(states.dtype)
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1.)
        return torch.einsum("bkj,bjh->bkh", weights, states)

    def _prepare(self, features, initial_weights, owner_ids):
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected features [B,T,N,{self.input_dim}]")
        b, t, n, _ = features.shape
        if min(b, t, n) < 1 or initial_weights.ndim != 3:
            raise ValueError("Empty features or incorrectly shaped initial weights")
        if initial_weights.shape[0] != b or initial_weights.shape[2] != n:
            raise ValueError("Initial weights must share feature batch and patch dimensions")
        k = initial_weights.shape[1]
        if k < 1:
            raise ValueError("At least one part slot is required")
        features = features.to(dtype=self.anchor_state.weight.dtype)
        weights = initial_weights.to(device=features.device, dtype=features.dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(-1) <= 0).any():
            raise ValueError("Every part requires finite nonnegative frame-zero weights with positive mass")
        if owner_ids is None:
            if k != 4:
                raise ValueError("Default owner grouping is defined for four parts only")
            owner_ids = torch.tensor([0, 0, 1, 1], device=features.device)
        else:
            owner_ids = torch.as_tensor(owner_ids, device=features.device)
        if owner_ids.ndim == 1 and owner_ids.shape[0] == k:
            owner_ids = owner_ids[None].expand(b, -1)
        if owner_ids.shape != (b, k):
            raise ValueError("Owner IDs must have shape [K] or [B,K]")
        # Pool the fixed projected features, then normalize once. This is also
        # the representation whose detached future counterpart can be targeted.
        weights = weights/weights.sum(-1, keepdim=True)
        anchors = _unit(torch.einsum("bkn,bnd->bkd", weights, features[:, 0]))
        return features, anchors, owner_ids

    def forward(self, features, initial_weights, owner_ids=None):
        features, anchors, owner_ids = self._prepare(features, initial_weights, owner_ids)
        b, steps, patches, _ = features.shape
        parts = anchors.shape[1]
        states = torch.tanh(self.anchor_state(anchors))
        futures, state_history, logits_history = [], [], []
        visibility_history, attention_history, prior_history = [], [], []
        scale = self.logit_scale.exp().clamp(1., 50.)
        for t in range(steps):
            tokens = features[:, t]
            unit_tokens = _unit(tokens)
            keys = _unit(unit_tokens+self.key_residual(unit_tokens))
            owner = self._sibling_context(states, owner_ids)
            prior = futures[t-self.horizon] if t >= self.horizon else anchors
            context = torch.cat((anchors, states, owner, prior), dim=-1)
            query = _unit(anchors+self.forecast_weight*prior+self.query_residual(context))
            spatial_logits = scale*torch.einsum("bkd,bnd->bkn", query, keys)
            attention = spatial_logits.softmax(-1)
            retrieved = torch.einsum("bkn,bnd->bkd", attention, unit_tokens)
            similarities = spatial_logits/scale
            statistics = torch.stack((similarities.max(-1).values,
                                      similarities.mean(-1),
                                      similarities.std(-1, unbiased=False)), dim=-1)
            absence_log_odds = self.absence(torch.cat((context, retrieved, statistics), -1)).squeeze(-1)
            # Factor spatial localization from presence. The null class gets
            # exactly sigmoid(absence_log_odds) probability, independent of N.
            null_logit = torch.logsumexp(spatial_logits, dim=-1)+absence_log_odds
            logits = torch.cat((spatial_logits, null_logit[..., None]), dim=-1)
            visibility = torch.sigmoid(-absence_log_odds)
            update = self.update_input(torch.cat((retrieved, anchors, owner), dim=-1))
            proposed = self.memory(update.reshape(b*parts, -1), states.reshape(b*parts, -1))
            proposed = proposed.reshape(b, parts, self.hidden_dim)
            states = states+visibility[..., None]*(proposed-states)
            updated_owner = self._sibling_context(states, owner_ids)
            future = _unit(self.future_predictor(torch.cat((anchors, states, updated_owner), dim=-1)))
            futures.append(future)
            state_history.append(states)
            logits_history.append(logits)
            visibility_history.append(visibility)
            attention_history.append(attention)
            prior_history.append(prior)
        return {"logits": torch.stack(logits_history, dim=1),
                "future": torch.stack(futures, dim=1),
                "states": torch.stack(state_history, dim=1),
                "visibility": torch.stack(visibility_history, dim=1),
                "attention": torch.stack(attention_history, dim=1),
                "forecast_used": torch.stack(prior_history, dim=1),
                "anchors": anchors}


def self_test(device="cpu"):
    """Architectural invariants on fabricated inputs, not research evidence."""
    torch.manual_seed(712)
    device = torch.device(device)
    model = PersistentPartJEPA(input_dim=16, hidden_dim=8, horizon=3).to(device)
    model.eval()
    features = torch.randn(2, 7, 11, 16, device=device)
    weights = torch.rand(2, 4, 11, device=device)
    owners = torch.tensor([[0, 0, 1, 1], [4, 4, 9, 9]], device=device)
    output = model(features, weights, owners)
    assert output["logits"].shape == (2, 7, 4, 12)
    assert output["states"].shape == (2, 7, 4, 8)
    assert output["future"].shape == (2, 7, 4, 16)
    assert torch.allclose(output["future"].norm(dim=-1), torch.ones(2, 7, 4, device=device), atol=1e-5)
    assert all(torch.isfinite(value).all() for value in output.values())
    probability = output["logits"].softmax(-1)[..., :-1].sum(-1)
    assert torch.allclose(probability, output["visibility"], atol=1e-6)
    assert torch.allclose(output["forecast_used"][:, 3:], output["future"][:, :-3], atol=0.)
    assert torch.allclose(output["forecast_used"][:, :3], output["anchors"][:, None].expand(-1, 3, -1, -1))

    changed = features.clone()
    changed[:, 4:] = 20*torch.randn_like(changed[:, 4:])
    changed_output = model(changed, weights, owners)
    for name in ("logits", "future", "states", "visibility"):
        assert torch.allclose(output[name][:, :4], changed_output[name][:, :4], atol=0., rtol=0.)

    slot_order = torch.tensor([2, 0, 3, 1], device=device)
    permuted = model(features, weights[:, slot_order], owners[:, slot_order])
    for name in ("logits", "future", "states", "visibility"):
        assert torch.allclose(permuted[name], output[name][:, :, slot_order], atol=2e-5, rtol=1e-5)
    patch_order = torch.randperm(11, device=device)
    reordered = model(features[:, :, patch_order], weights[:, :, patch_order], owners)
    assert torch.allclose(reordered["logits"][..., :-1], output["logits"][..., :-1][..., patch_order], atol=2e-5, rtol=1e-5)
    assert torch.allclose(reordered["logits"][..., -1], output["logits"][..., -1], atol=2e-5, rtol=1e-5)

    # A localization-only loss after the forecast delay must train the predictor;
    # before the delay the predictor has no path into retrieval yet.
    future_parameters = tuple(model.future_predictor.parameters())
    early_loss = output["logits"][:, :3, :, 0].sum()
    early_grad = torch.autograd.grad(early_loss, future_parameters, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.abs().any() for g in early_grad)
    late_loss = -output["logits"][:, 3:, :, :11].log_softmax(-1)[..., 0].mean()
    late_grad = torch.autograd.grad(late_loss, future_parameters, allow_unused=True)
    gradient_norm = sum(float(g.abs().sum().detach().cpu()) for g in late_grad if g is not None)
    assert gradient_norm > 1e-8
    return {"shapes_and_finite": True, "unit_future_predictions": True,
            "null_probability_matches_visibility": True, "exact_forecast_queue": True,
            "future_prefix_isolation": True, "slot_permutation_equivariance": True,
            "patch_permutation_equivariance": True,
            "localization_loss_reaches_future_predictor": True,
            "future_predictor_localization_gradient_l1": gradient_norm,
            "real_model_results": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
