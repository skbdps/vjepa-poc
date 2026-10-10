# What the translator does

This is a native-image pilot using the conditioning branch of a video generator.
Its purpose is to test whether an explicit JEPA edit can reach a real generator
while retaining source appearance. It does not yet model a video trajectory.

## Why ControlNet is useful, and what it does not supply

The original [ControlNet implementation](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py)
adds learned spatial control to a pretrained diffusion backbone. That is an
architectural precedent, not a decoder for arbitrary JEPA vectors. Our actual
generator is Wan2.1-VACE-1.3B, which already has a pretrained spatial condition
branch. We reuse that branch and train a smaller translator to its input space.

The audited [Diffusers 0.35.1 VACE pipeline](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/wan/pipeline_wan_vace.py)
constructs a 96-channel condition: 16 inactive VAE channels, 16 reactive VAE
channels, and a 64-channel packed mask. In the all-known image case the reactive
input is the VAE encoding of zero-valued normalized RGB, not literal latent
zeros. The bridge reproduces this interface exactly. Ordinary RGB conditioning
and direct injection produced bitwise-identical denoised latents in all four
initial oracle checks with the same prompt and noise.

## The learned map

Let J(x) be frozen dense V-JEPA2.1 features and A(x) the normalized posterior
mode of the frozen Wan VAE. They have different training purposes, coordinates,
and spatial resolutions. There is no assumption that one is an invertible
change of basis of the other.

We train F with genuine image pairs using

    minimize_F mean_x ||F(J(x)) - A(x)||².

F maps 1024 channels on a 24×24 grid to 16 channels on a 48×48 grid. It receives
train-normalized features and fixed grid coordinates. It does not receive the
requested displacement, target image, or target mask. Both a spatial CNN and
a shared linear projection are trained under the same fixed budget.

For an edited grid j_e, the absolute route supplies F(j_e). The original
source-preserving route supplies

    A(s) + F(j_e) - F(J(s)).

The second route gives exact conditioning identity for a zero edit. It does
not give exact output-pixel identity after the generator processes a nonzero
edit. Its observed failure was that residual texture stayed at the old object
position, producing ghosts even when genuine target JEPA features were used.

## Transporting source appearance with the edit

The explicit copy/repair editor retains whole JEPA vectors at copied cells.
For each destination cell q, the follow-up infers a unique source cell p by
exact equality of all 1024 components. Ambiguous matches are rejected. No
target pixels, target masks, or requested displacement enter this matching.
One JEPA cell corresponds to a 2×2 tile of the Wan conditioning grid.

The primary corrected route is:

    unchanged q:  A(s)[q]
    copied p→q:   F(j_e)[q] + A(s)[p] - F(J(s))[p]
    unmatched q:  F(j_e)[q]

All reads come from immutable source arrays, including overlapping moves.
Source-only local fill and direct source-tile copy are separate comparators.
The local-fill version uses no values from the learned translator, which tests
whether the learned part adds anything beyond transporting known appearance.

This correspondence is editor-specific provenance. Arbitrary unique token
identifiers could carry the same copy operation. Therefore a transport gain
alone is not evidence that JEPA semantics are uniquely responsible. Genuine
target encodings have no exact matches here and fall back to the learned map;
their results are a different diagnostic, not an upper bound on transport.

## What a successful pilot would establish

The useful result is conditional: a supplied source selection and explicit
JEPA movement can be translated to Wan conditioning, and the rendered result
follows that movement while retaining useful source detail. It would not prove
automatic part discovery, face identity, learned physics, or temporal stability.

The squared-error map can average detail it cannot predict reliably from JEPA.
Our experiments do not distinguish irreversible information loss from limited
data, capacity, and optimization. Source appearance is consequently retained
explicitly. The generator remains probabilistic, and even a deterministic
sampler with fixed noise does not enforce geometric or identity constraints.
Single-frame success must be followed by a separate temporal experiment with
native video features and the video VAE's temporal compression accounted for.

## Next experiment that would distinguish the mechanisms

Before increasing video length, test a fractional-patch movement with a shared,
explicit transport map P. For example, an 8-pixel movement cannot be achieved
by unchanged copies of 16-pixel JEPA cells. Supply the same resampling map to
the learned and no-F arms rather than recovering matches by exact equality.
This is a proposed follow-up, not part of the current frozen evaluation.

Compare edited JEPA and independently encoded genuine-target JEPA using the
same appearance term:

    condition(z) = F(z) + P[A(source) - F(J(source))].

Their difference then isolates F(edited JEPA) - F(genuine target JEPA). Retain
the no-F appearance-transport/local-fill comparator with the same geometry.
If the oracle succeeds and the edit fails, investigate the edited representation.
If both fail, investigate the bridge or information retained in its inputs.
If no-F remains comparable, attribute the useful engineering result primarily
to source-appearance transport. The current exact-copy test alone cannot resolve
these questions, and it does not use the JEPA predictor to constrain dynamics.
