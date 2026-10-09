# JEPA editing as a control interface for video generation

Research design note, 2026-10-09 UTC. No new experiment, implementation, or novelty claim accompanies this note. Proposed equations describe hypotheses and design requirements, not established capabilities.

## Recommendation and evidence

Investigate **source-aware JEPA edit control, original appearance information, and explicit preservation constraints**. The generator should render a specified change while retaining the source details that the control representation does not protect. A universal JEPA-to-video-latent translator is a less defensible starting point.

Our existing results concern supplied masks, commanded translations, simple synthetic scenes, and coarse readouts. They provide evidence of useful representation interventions, not realistic video generation:

| Experiment | Verified finding | Consequence for this proposal |
| --- | --- | --- |
| [Day8](../experiments/day8/RESULTS.md) | Direct copying achieved regional latent-error ratio 0.1187 and 1.33-pixel centroid error across 16 held-out scenes. Learned correction regressed to 0.1499. | Preserve the successful intervention; do not assume another learned component improves it. |
| [Day9](../experiments/day9/RESULTS.md) | Temporal repair reduced source-hole ratio from 0.224936 to 0.039466 across 16 fresh scenes. The variant winning latent distance did not win RGB error. | Observed background memory helps; representation agreement and visual quality are different objectives. |
| [Day10](../experiments/day10/RESULTS.md) | Native-image follow-up achieved ratio 0.1306, 1.495-pixel centroid error and 24/24 coarse-color agreement on four fresh images. Five of twenty moves worsened hole RGB error. | Controlled single-image edits are promising, but JEPA improvement does not guarantee better pixels. |

The [Day8 evidence](../experiments/day8/RESULTS.md), [Day9 audit](../experiments/day9/run_2026-10-09/analysis/independent_audit.json), and [fresh Day10 audit](../experiments/day10/native_readout_run_2026-10-10/fresh_test/analysis/independent_audit.json) support the saved arithmetic within their documented scopes.

Day10's [initial attempt](../experiments/day10/INITIAL_RESULTS.md) failed: its transferred video readout misclassified every genuine target's coarse color. A separately trained native-image readout passed fresh validation without changing the editor. **Native image features cannot simply be assumed interchangeable with native video features.** A bridge must validate its input branch, temporal sampling, feature layers, and resolution.

Exact unchanged tokens and exact returns are enforced by the editor. They do not establish learned identity, fine texture preservation, automatic tracking, or preservation after a generative decoder.

## Different mathematical jobs

Use \(x\) for source pixels, \(j=E(x)\) for JEPA features, and \(z=A(x)\) for a reconstructive generator latent.

An LLM typically learns conditional token probabilities, schematically
\[
\mathcal L_{\rm LM}=-\sum_i\log p(w_i\mid w_{<i},c).
\]
It can translate an instruction into object IDs, constraints and edit parameters. That does not make its token space a geometrically editable scene.

JEPA predicts representations of withheld observations:
\[
\mathcal L_{\rm JEPA}\sim
\|P(E(x_{\rm context}),m)-\operatorname{sg}(E_{\rm target}(x)_m)\|.
\]
This is schematic, omitting architecture-specific supervision and collapse prevention. [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) adds dense predictive supervision, intermediate-layer supervision and image/video tokenizers. Its objective supports useful features; it does not impose a pixel reconstruction identity.

A diffusion/flow generator learns a conditional distribution through denoising or velocity prediction. Its autoencoder is trained for visual reconstruction, approximately \(D(A(x))\approx x\). [Latent diffusion](https://arxiv.org/abs/2112.10752) demonstrates generation in pretrained autoencoder space with conditioning interfaces. Thus “latent” does not mean the same coordinates, information, or operation across models. A video generator is also not necessarily an LLM.

Our problem is constrained counterfactual generation: satisfy the requested change, preserve specified content, and retain plausible interactions. Text-conditioned plausibility alone does not impose those constraints. This is a design diagnosis, not a claim that existing generators cannot support additional controls.

## What a translation layer must satisfy

For a deterministic bridge \(B\) to satisfy \(A=B\circ E\) exactly on a domain, a necessary and sufficient set-theoretic condition is
\[
E(x_1)=E(x_2)\ \Longrightarrow\ A(x_1)=A(x_2).
\]
In words, the reconstructive target must be constant on every set of inputs with the same JEPA representation. Otherwise one input to \(B\) requires two different outputs. Approximate collisions can also make inversion unstable.

This is a mathematical condition, not proof of collisions in our checkpoint. JEPA training does not guarantee the condition or a stable inverse. In particular, dimension counting alone does not prove non-invertibility of our native-image output. Keeping source pixels or appearance features supplies information a JEPA-only bridge may lack.

More importantly, reconstruction alignment is weaker than **intervention alignment**. Let \(T_u\) be our token edit and \(U_u(x)\) the genuinely edited scene. Learning
\[
B(E(x))\approx A(x)
\]
on natural examples does not establish
\[
B(T_u(E(x)))\approx A(U_u(x)).
\]
Copied contextual tokens may not form the encoding of any plausible scene. A decoder might repair that mismatch by moving the object, changing its identity, or disturbing the background. Successful reconstruction of untouched inputs therefore cannot certify edit fidelity.

Use source-conditioned differences instead:
\[
\hat x=G(x,\ R(x),\ j,\ T_u(j)-j,\ u,\ M,\ \epsilon),
\]
where \(R(x)\) retains appearance, \(M\) defines permissible changes, and \(\epsilon\) fixes sampling randomness. This is a proposed interface; whether the generator follows it remains untested.

## Architectures worth distinguishing

**1. Source-aware edit adapter.** Train a small adapter to inject spatially and temporally indexed JEPA edit information into generator features or denoising updates. Train on paired interventions, including zero edits and held-out objects, rather than only matching representations of unchanged scenes. Keep the source appearance path independent. A zero adapter residual does not itself guarantee zero visual change; exact no-op behavior needs a source bypass or composition rule.

[JEPA Guided Diffusion](https://arxiv.org/html/2609.21379v1) already maps V-JEPA 2.1 features plus captions to frozen Cosmos cross-attention context while retaining source frames. It supports interoperability, not invertible video-latent translation or local edit fidelity. Its teacher targets derive from refined training captions. [VACE](https://arxiv.org/html/2503.07598v1) provides another relevant precedent for source, mask and reference conditioning through adapters. Neither establishes our proposed intervention requirement.

**2. Constrained inference or a critic.** Before training an adapter, test whether edited JEPA targets can guide an accessible generator:
\[
\mathcal L(y)=\lambda_e\|W(E(y)-j^\star)\|^2+
\lambda_p L_{\rm preserve}(y,x)+
\lambda_a L_{\rm appearance}(y,x)+
\lambda_t L_{\rm temporal}(y).
\]
This proposed positive mismatch energy separates edit correctness, preservation and temporal behavior. With gradients, optimize intermediate generation controls; without gradients, rank a fixed candidate set. Ranking cannot rescue an edit the generator never produces. Optimizing JEPA distance can exploit its blind spots, so independent visual measurements remain necessary.

[WMReward](https://arxiv.org/html/2601.10553v1) uses latent-world-model surprise for candidate selection and sampling guidance. It supports world-model-based assessment of generated video; physical predictability is not equivalent to the artist's intent. We do not import its printed reward sign convention into this proposal.

**3. Persistent object and part state.** Maintain shared asset IDs, part hierarchies, appearance references, poses, cameras and lighting across shots. A door-angle revision changes this common state, then each shot receives corresponding controls. JEPA can assist association, state estimation and validation, but token coordinates are not persistent object IDs. Cross-view geometry, occlusion and part correspondence require additional representations and evidence.

**4. Layers with generative residuals.** Preserve finished content through
\[
y=(1-M)\odot x+M\odot \tilde y.
\]
Where \(M=0\), pixels are unchanged by construction. Deterministic object warping can retain observed texture; generation fills genuinely missing content. [Vera](https://arxiv.org/html/2606.23610v1) researches edit layers and alpha compositing for preservation. Our distinct experimental question would be whether JEPA improves the controlled edit or fill beyond the identical layered pipeline without JEPA.

**5. Action-conditioned planning followed by rendering.** A model \(j_{t+1}=P(j_{\le t},a_{\le t})\) could optimize trajectories before visual synthesis. Creative parameters require an appropriate action space and training; robot actions do not automatically encode animation instructions. [RoboJEPA](https://arxiv.org/html/2610.10515v1) supplies a concrete robotic planning precedent and a separate diffusion decoder conditioned on V-JEPA 2.1 features. That decoder visualizes rollouts; it is not part of the planner and does not establish decoding fidelity for our token edits.

**6. Longer-term representation autoencoders.** Train a projector and reconstructive decoder around pretrained features, possibly retaining multiple layers or an appearance residual. This could reduce interface mismatch, but requires reconstruction and generation training. The [VideoRAE author page](https://zhxie0117.github.io/VideoRAE/) describes compact reconstructive latents derived from frozen multiscale video-foundation features. Generation/reconstruction results do not establish that our edit survives projection and decoding. This is a larger alternative, not the cheapest next test.

## Why some controls conflict

A moved car changes shadows, reflections, occlusion and exposed background. An allowed edit region must include the intended consequences. Requiring all other pixels to remain fixed can contradict physical coherence. Across camera changes, preserving identity also cannot mean preserving pixel values.

JEPA tokens carry context: unchanged pixels can acquire different features when another object moves. Consequently, copying some tokens while freezing every other token can conflict with the representation of a valid edited scene. Lower feature error remains informative but is not a proof of physical realizability.

A local controllability diagnostic makes the issue explicit. Let \(a\) denote adjustable conditioning, noise, or adapter controls, and \(R(a)\) the complete generated and decoded video. Define \(F(a)=E(R(a))\), and let \(H(a)=\operatorname{protected}(R(a))\) measure protected properties. At one operating point, write \(J=\partial F/\partial a\) and \(K=\partial H/\partial a\). A desired small change requires
\[
J\,\delta a=\delta j,\qquad K\,\delta a=0.
\]
If columns of \(N\) span \(\ker K\), feasibility requires \(\delta j\in\operatorname{Range}(JN)\). A small feasible subspace explains why increasing guidance may damage preserved content instead of improving control. This is a first-order diagnostic, not a global guarantee; computing full Jacobians is optional.

## A falsifiable next experiment

Start with controlled image edits or short static-camera clips. Use fresh textured objects, supplied masks, known target renders, fixed generator settings and paired random seeds. The research hypothesis is that JEPA edits add useful control beyond source appearance and explicit geometry.

Compare:

1. Source, command and geometry without JEPA.
2. The same system with our edited JEPA target.
3. Unedited JEPA features, testing whether gains merely come from extra source conditioning.
4. Wrong-direction and spatially permuted, magnitude-matched edits.
5. Genuine target-encoded JEPA features as an evaluation-only diagnostic, never a deployable input.
6. For learned adapters, equal-capacity geometry-only and alternative-feature controls.

Measure **joint success per scene**: requested displacement/part state, protected-region error, appearance under the intended transform, ghosting, temporal stability, and return/revision fidelity must all satisfy thresholds frozen on separate development data. Test return-to-source for reversible edits, order consistency for genuinely independent edits, and preservation of approved earlier constraints during later revisions. Interacting edits need not commute. Report each component and their conjunction; an average can hide failed preservation behind better motion. Evaluate output independently of the JEPA loss being optimized. Exact token preservation is not an output metric.

Interpret outcomes before escalating:

- Genuine target conditioning works, copied targets fail: investigate intervention validity or the edited-feature distribution.
- Neither works: investigate bridge controllability, feature information, or generator support.
- Both work, geometry-only matches them: JEPA has not demonstrated added value.
- JEPA error falls while visual identity degrades: the control metric is insufficient.
- Actual edit accuracy improves with preservation intact: evidence supports the combined interface within the tested scope.

Genuine target conditioning is a diagnostic reference, not a guaranteed upper bound for every learned interface. Cross-shot tests should follow with shared assets, held-out viewpoints and repeated revisions. The central unresolved question is whether controlled JEPA interventions can remain controlled after realistic rendering; existing results justify testing that question, not claiming it solved.
