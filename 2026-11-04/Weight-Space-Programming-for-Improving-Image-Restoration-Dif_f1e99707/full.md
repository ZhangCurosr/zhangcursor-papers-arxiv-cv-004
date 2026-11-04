# Weight-Space Programming for Improving Image-Restoration Diffusion Models: Technical Conjectures and a Recursive Improvement Loop

Zhuoran Zheng and Doubao

zhengzr@njust.edu.cn

## Abstract

Can a large language model (LLM) directly edit the weights of a diffusion-based image-restoration model to improve its restoration quality—without gradients, without paired training data, and without fine-tuning? Recent experiments on language models suggest frontier LLMs can locate and fix structural bugs by modifying as little as 0.016% of parameters, but cannot instill genuinely new capabilities. We ask whether this weightspace programming (WSP) paradigm can be pushed from debugging to improvement in the precise-regression setting of image restoration (deblurring, super-resolution, denoising, inpainting). We argue that restoration diffusion UNets present a unique middle ground: low-level signal-processing layers are concentrated enough for targeted weight-direction edits, while high-level semantic layers remain distributed and resistant to sparse modification. We propose three concrete technical conjectures: (C1) an LLM can diagnose which layer causes a specific quality gap (e.g., over-smoothing, color shift, hallucinated texture) within a 200 forward-pass budget; (C2) editing a small set of identified weight directions (e.g., scaling up high-frequency filter norms, correcting output-bias channels) can measurably improve restoration PSNR and perceptual quality; (C3) a recursive loop-LLM diagnoses → weight edit → evaluate → re-diagnose-can compound improvements across multiple rounds, constituting a minimal recursive self-improvement (RSI) loop for vision models. We analyze what AI capability level is required (arguing that targeted layer-level edits need not require superintelligence, but general capability generation does), and we detail the unique challenges of precise pixel-level regression. We provide a deployment pipeline in PyTorch-style pseudocode and outline a benchmark for empirical validation. The numerical conjectures in this manuscript are merely hypotheses.

## 1 Introduction

Image restoration-recovering clean images from blurry, noisy, low-resolution, or corrupted inputs-is one of the most practical computer-vision problems. Modern diffusion-based restoration models achieve state-of-the-art quality, yet they over-smooth fine textures, hallucinate plausible-but-fake details, shift colors under specific lighting, and fail on degradation types absent from training data. Fixing these failures today requires collecting paired data and fine-tuning: expensive, slow, and risky for the rest of the distribution.

What if an LLM could read the restoration UNet’s weights, understand what each layer computes, and directly edit the weight tensors to make the model sharper, more color-accurate, and less hallucinatory—no gradients, no training data?

This is the promise of Weight-Space Programming (WSP): treating neural network parameters as a text-like object an LLM can inspect, reason about, and patch [Takehi, 2026]. The first evidence: a frontier LLM inspected Qwen3.5-0.8B weights and eliminated a reasoning-loop bug by modifying only 0.016% of parameters [Takehi, 2026, RikiyaT, 2026b,a]. The same experiments produced a crucial negative result: endowing a 270M Gemma3 with new math ability via weight editing failed. WSP can unlock latent competence already present in weights, but cannot create competence from nothing.

This paper asks: can WSP go beyond bug-fixing and actually improve restoration quality? And does it require superintelligence?

We focus on restoration diffusion models because: (1) restoration is a precise regression task with continuous pixel outputs and noisy perceptual metrics-harder than language modeling but more informative about WSP’s limits; (2) restoration UNets have clear layered specialization (early = low-level, bottleneck = semantic, late = output), making some layers concentrated (editable) and others distributed (not); (3) artifacts are visually verifiable.

We make three concrete technical conjectures (Section 4), analyze what AI level is required (Section 5), describe a deployment pipeline (Section 6), and discuss RSI implications (Section 9).

## 2 Background: What WSP Has Demonstrated

Positive result. A frontier LLM inspected Qwen3.5-0.8B weights and eliminated a reasoning-loop bug by modifying only 0.016% of parameters [Takehi, 2026]. This worked because: (1) the bug was a concentrated direction in weight space, and (2) the underlying reasoning competence was already present—the bug was interference, not absence of ability. Patched weights: [RikiyaT, 2026a]; full logs: [RikiyaT, 2026b].

Negative result. The same experiments tried to give 270M Gemma3 new mathematical ability via direct weight editing and failed. The LLM cannot write distributed representational structure by inspecting and modifying weight entries.

This establishes WSP’s boundary: it unlocks latent competence suppressed by concentrated interference, but cannot create new distributed competence. The question for restoration is: which parts of a diffusion UNet are concentrated (editable) and which are distributed (not)?

## 3 Why Restoration Diffusion Models Are Different

Layered specialization. A restoration UNet has three functional regions: early encoder layers do low-level signal processing (edges, colors, textures)—relatively interpretable; the bottleneck does high-level semantic reasoning—most distributed; late decoder layers do output reconstruction—the final 1 × 1 conv bias directly controls color balance. Our hypothesis: early and late layers are WSP-friendly; the bottleneck is WSP-resistant.

Precise regression. Restoration outputs continuous pixel tensors, not discrete tokens. Quality is measured by noisy continuous metrics (PSNR, SSIM, LPIPS), not binary correct/incorrect. A weight edit has a continuous, graded effect. This makes WSP evaluation harder than in language models.

Diffusion chain. The UNet runs T = 200 times along a denoising Markov chain. An early-layer edit affects coarse structure; a late-layer edit affects fine detail. This temporal dimension is a bonus: if a hallucinated texture appears only in the last steps, the LLM knows to look at late decoder layers.

Permutation symmetry. Swapping conv channels leaves the network functionally identical. The LLM may edit a channel that looks wrong but is actually a permuted counterpart of the true causal channel-wasting budget.

## 4 Three Technical Conjectures

We state three concrete, testable conjectures. Each specifies the LLM’s input, the weight it edits, the expected measurable improvement, and the failure boundary.

## 4.1 Conjecture 1: Layer-level quality-gap diagnosis

Goal. Given a trained restoration diffusion model and a description of a quality gap (e.g., “outputs are too smooth,” “colors are slightly washed out,” “fine textures are hallucinated”), the LLM identifies which layer(s) are responsible, within a 200 forward-pass budget.

What the LLM sees. For each layer chunk (5–10 layers at a time): layer name, type (conv/bn/attention), channel counts, kernel size, weight histogram statistics (mean, std, sparsity), top-k largest-magnitude entries, and mean activation magnitude on a small feedback set.

What the LLM does. It forms hypotheses based on architectural priors:

• “Too smooth” → look at late decoder layers (reconstruction) and bottleneck (semantic level of detail).

• “Color washed out” → look at final output conv bias and early encoder color-filtering layers.

• “Hallucinated texture” → look at mid-level decoder upsampling blocks.

It then requests targeted ablations (e.g., “scale up all filter norms in decoder block 3 by 1.2×”) and observes whether the feedback-set PSNR/LPIPS improves.

Expected result. The LLM narrows down the responsible layer(s) to a short list (≤ 5 layers) within 30 forward passes. This is not yet an edit—it is a diagnosis. The edit comes in C2.

Boundary. Diagnosis works for systematic quality gaps (the model is consistently too smooth on all images). It fails for image-specific failures (the model hallucinates on only one category of images), because the feedback set is too small to capture those distribution-specific effects.

Falsification. If, on a benchmark of 20 quality-gap restoration models, the LLM fails to identify the correct layer within 50 passes in more than half of cases, C1 is false.

## 4.2 Conjecture 2: Targeted weight-direction editing improves restoration quality

Goal. Given a correctly diagnosed layer (from C1), the LLM proposes a specific weight delta that measurably improves restoration quality (PSNR ≥ +0.2 dB, or perceptually sharper with no new artifacts), modifying < 0.1% of total parameters.

## Three concrete edit types we expect to work:

(a) Output-bias correction (color balance). If the model produces a consistent color cast (e.g., slightly greenish shadows), the LLM identifies the final 1 × 1 output convolution’s bias vector and proposes shifting the offending channel. This is the most concentrated edit possible: a 3-element bias vector for RGB output. Expected effect: color cast corrected, PSNR slightly improved, no side effects.

(b) High-frequency filter scaling (sharpness). If the model is over-smooth, the LLM identifies the decoder layers responsible for detail reconstruction and proposes scaling up the norms of high-frequency filters (filters with large high-pass spectral energy). This is a semi-concentrated edit: it affects a set of filters within one or two decoder blocks, not individual entries. Expected effect: sharper output, possibly slightly more noise, PSNR improvement depends on the specific degradation.

(c) Skip-connection gating (artifact suppression). If the model produces halos around high-contrast edges, the LLM identifies the skip connection that carries edge information into the decoder and proposes scaling down the skip-connection weights at the specific resolution where halos appear. This is a semi-concentrated edit targeting a specific resolution block.

What does NOT work (distributed defects). General over-smoothing caused by the bottleneck being too compressed, or hallucinated detail caused by the model’s semantic prior being too strong, cannot be fixed by sparse edits. These are distributed properties of the entire network’s representation, not localized weight directions.

Boundary. Sparse weight editing works when the quality gap corresponds to a concentrated weight direction. It fails when the gap is a distributed property of the representation. The taxonomy in Table 1 classifies common restoration quality gaps.

Table 1: Expected editability of common restoration quality gaps.
<table><tr><td>Quality gap</td><td>Concentration</td><td>Sparse edit works?</td></tr><tr><td>Color cast / white balance</td><td>Concentrated</td><td>Yes (output bias)</td></tr><tr><td>Checkerboard / upconv artifact</td><td>Concentrated</td><td>Yes (transpose conv weights)</td></tr><tr><td>Halos on edges</td><td>Semi-concentrated</td><td>Maybe (skip connection scale)</td></tr><tr><td>Over-sharpening / ringing</td><td>Semi-concentrated</td><td>Maybe (high-freq filter norm)</td></tr><tr><td>Global over-smoothing</td><td>Distributed</td><td>No (needs retraining)</td></tr><tr><td>Hallucinated texture</td><td>Distributed</td><td>No (semantic prior issue)</td></tr><tr><td>Domain gap (RAW vs sRGB)</td><td>Distributed</td><td>No (representation mismatch)</td></tr></table>

Falsification. If the LLM cannot improve PSNR by $\ge 0 . 2$ dB on at least half of the concentrated-defect benchmark cases, with < 0.1% parameters edited, C2 is false.

## 4.3 Conjecture 3: Recursive improvement loop

Goal. After one successful edit, the LLM re-diagnoses the improved model and proposes a second edit, compounding improvements across multiple rounds. This recursive loop—diagnose → edit → evaluate → re-diagnose—is a minimal form of recursive self-improvement (RSI) for vision models.

How it works. Round 1: LLM diagnoses color cast, fixes output bias. Round 2: with color cast gone, the LLM re-evaluates and now notices the model is slightly over-smooth on texture. It scales up high-frequency filters in decoder block 3. Round 3: now sharper, the LLM notices a mild halo around edges and adjusts the skip-connection scale. Each round starts from the improved model and targets the next most prominent quality gap.

Why this is plausible. Each edit is small (< 0.1% cumulative), and each round’s diagnosis is on a model that is slightly better than the last. The LLM never has to solve the whole problem at once; it solves one sub-problem per round. This is analogous to how a human engineer iteratively improves a model: fix the worst artifact first, then the next, then the next.

Boundary and diminishing returns. Improvements should be monotonically positive but diminishing. After 5–10 rounds, the remaining quality gaps should be distributed (not sparsely editable), and the loop should plateau. The total improvement over a full loop might be +0.5–1.5 dB PSNR—meaningful, but not a leap from bad to state-of-the-art.

Why this matters for RSI. This loop is a concrete, small-scale RSI mechanism: the LLM improves a model, then uses the improved model as the starting point for further improvement. It is not AGI-level recursive self-improvement; it is a constrained, budget-limited debugging loop. But it demonstrates the core principle: a model-improving agent (the LLM) can iteratively modify its target (the restoration UNet) in the weight space.

Falsification. If improvements do not compound across rounds (i.e., round 2 edit makes things worse or no better than round 1 baseline), or if total improvement over 10 rounds $\mathrm { i s < 0 . 2 d B }$ , C3 is false.

Table 2: Summary of the three conjectures.
<table><tr><td>Conjecture</td><td>What it does</td><td>Expected gain</td></tr><tr><td>C1: Diagnosis</td><td>Locate responsible layer(s)</td><td>20–50 fwd passes</td></tr><tr><td>C2: Sparse edit</td><td>Improve quality via weight delta</td><td>+0.2–0.5 dB PSNR</td></tr><tr><td>C3: Recursive loop</td><td>Compound improvements</td><td>+0.5-1.5 dB total</td></tr></table>

## 5 What Level of AI Is Needed?

What current frontier LLMs can do. Based on language-model evidence [Takehi, 2026], a GPT-6-class LLM can: read weight statistics and form layer-function hypotheses, propose simple structured deltas (shift a bias channel, scale a filter norm), and use architectural priors (early layers = low-level, late = output). These capabilities suffice for C1 (diagnosis) and C2 (concentrated-defect editing). This is pattern-matching + reasoning, not superintelligence.

What current LLMs cannot do. Write distributed representations from scratch (the Gemma3-270M math failure), reason about cross-layer interactions requiring two distant layers in context simultaneously, or predict edit effects across the full T-step denoising chain without running it.

Do we need AGI? For targeted editing (fixing known quality gaps in an already-trained model), probably not. Current frontier LLMs have enough reasoning and numeric output capability. For from-scratch weight generation, yes—writing a functional 10M-parameter conv net from scratch requires understanding the optimization landscape that produced those weights.

Hybrid middle ground. The most practical path: LLM diagnoses and proposes a weight direction, then a small gradient fine-tune polishes it. The LLM provides high-level direction; the optimizer provides pixel-leve precision.

## 6 Deployment Pipeline

The sandbox sits between the LLM and the frozen restoration UNet. It enforces budgets, isolates data, and provides structured feedback.

## 6.1 Sandbox architecture

1. Chunked weight access: LLM receives 5–10 layers at a time, with compressed statistics (histogram, top-k entries, activation norms).

2. Forward-pass budget: 30–50 total evaluations per WSP session.

3. Edit-ratio cap: < 0.1% of total parameters modified cumulatively.

4. Feedback/test isolation: 32–64 image pairs for feedback; held-out test set for final evaluation.

5. Automatic acceptance criterion: accept an edit only if PSNR improves by ≥ 0.1 dB on feedback set and no new artifacts appear.

```python
class WSPSandbox:
def __init__(self, unet, feedback, budget=40,
max_ratio=1e-3, psnr_min=0.1):
self.unet, self.feedback = unet, feedback
self.budget, self.used = budget, 0
self.max_ratio, self.psnr_min = max_ratio, psnr_min
self.base_psnr = self._eval_psnr()
def propose_edit(self, layer, slc, delta):
if self.used >= self.budget:
raise BudgetExceeded()
edited = deepcopy(self.unet)
set_weight(edited, layer, slc, delta)
self.used += 1
new_psnr = self._eval_psnr(edited)
if new_psnr - self.base_psnr >= self.psnr_min:
```

```python
self.unet = edited
self.base_psnr = new_psnr
return f"accepted: {new_psnr:.2f} dB"
return f"rejected: {new_psnr:.2f} dB"
def _eval_psnr(self, model=None):
m = model or self.unet
with torch.no_grad():
return mean(psnr(run_chain(m, d), c)
for d, c in self.feedback)
```  
Listing 1: WSP sandbox: budget-controlled edit evaluation.

## 6.2 Recursive improvement loop

The outer loop repeatedly calls the LLM with the current model state and quality-gap description, accepts improving edits, and stops when the budget is exhausted or no further improvement is found.

```python
class RecursiveWSP:
def __init__(self, sandbox, llm, chunks):
self.sandbox, self.llm = sandbox, llm
self.chunks, self.log = chunks, []
def improve(self, gap_desc):
rnd = 0
while self.sandbox.used < self.sandbox.budget:
rnd += 1
found = False
for chunk in self.chunks:
stats = self.sandbox.layer_stats(chunk)
prompt = build_prompt(gap_desc, stats, rnd)
prop = self.llm.complete(prompt) # JSON
if prop["type"] == "edit":
res = self.sandbox.propose_edit(
prop["layer"], prop["slice"], prop["delta"])
self.log.append({"rnd": rnd, "prop": prop, "res": res})
if res.startswith("accepted"):
found = True
break
if not found:
break
return self.sandbox.unet, self.log
```  
Listing 2: Recursive WSP improvement loop.

## 7 Challenges of Precise Regression

Continuous vs. discrete feedback. In language models, feedback is binary (correct/incorrect token). In restoration, feedback is a noisy scalar (∆PSNR). The LLM must correlate weight changes with tiny PSNR deltas—harder than binary signals.

Sampling stochasticity. Diffusion sampling is seed-dependent. A 0.1 dB difference may be within noise. The sandbox must average over multiple seeds per evaluation, but this consumes budget: with 40 passes, only 2–3 seeds per evaluation.

Pixel metrics vs. perception. PSNR/SSIM correlate poorly with human perception. An edit that raises PSNR may look worse (over-smoothing). The LLM needs both scalar metrics and visual feedback (before/after image grids), but visual feedback costs more tokens.

Denoiser chain dynamics. An edit may help at T = 50 steps but hurt at T = 100. Single-number PSNR

## 8 Limitations

Scale. A 100M-parameter UNet cannot fit in any LLM context. The LLM works on compressed summaries, not raw weights.

Distributed defects. General over-smoothing or hallucinated texture are distributed properties—no sparse edit fixes them. WSP is a precision instrument, not a general training method.

Permutation symmetry. Channel permutation makes weight interpretation ambiguous; the LLM may waste budget on the wrong channel.

OOD side effects. An edit that helps the feedback set may hurt out-of-distribution images. Full regression testing is mandatory.

LLM search bias. The LLM may prefer editing late layers (easier to reason about). Randomizing chunk order mitigates this.

## 9 RSI Implications and Discussion

## 9.1 Is this a path to recursive self-improvement?

The recursive loop in C3 is a minimal RSI mechanism: an LLM improves a model, then uses the improved model as the starting point for further improvement. But it is bounded: budget-limited, human-supervised, incremental (< 0.1% parameters, < 1.5 dB total). It is not an autonomous self-improving system—the human defines what to improve, the sandbox evaluates, and the loop plateaus at distributed-defect boundaries.

That said, it is a proof of concept for the RSI principle: a model-improving agent operating in weight space. As LLMs improve at weight reasoning and sandbox budgets grow, the loop will compound more. The next threshold is whether the LLM can start proposing what to improve (not just how to edit).

## 9.2 Comparison to LoRA and fine-tuning

WSP is complementary, not competitive. LoRA/FT adjust 1–100% of parameters with gradients over hours/days; WSP adjusts < 0.1% with LLM reasoning over minutes. Practical workflow: WSP for quick targeted fixes, LoRA/FT for broad quality improvements.

Table 3: WSP vs. LoRA vs. full fine-tuning for restoration.
<table><tr><td>Dimension</td><td>WSP</td><td>LoRA</td><td>Full FT</td></tr><tr><td>Gradient needed?</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Params edited</td><td>&lt;0.1%</td><td>1-10%</td><td>100%</td></tr><tr><td>Data needed</td><td>Feedback set</td><td>Paired dataset</td><td>Paired dataset</td></tr><tr><td>Speed</td><td>Minutes</td><td>Hours</td><td>Days</td></tr><tr><td>Best for</td><td>Targeted fixes</td><td>Domain adaptation</td><td>Full retrain</td></tr></table>

## 9.3 Benchmark design

Test set: concentrated defects (injected: output-bias color cast, transposed-conv checkerboard, skip-connection halo) as positive cases; distributed defects (natural: over-smoothing, hallucinated texture, domain gap) as negative

## 10 Conclusion

We have proposed three concrete technical conjectures for applying Weight-Space Programming to improve image-restoration diffusion models: (C1) layer-level quality-gap diagnosis within a 20–50 forward-pass budget; (C2) targeted weight-direction edits improving PSNR by 0.2–0.5 dB on concentrated defects; (C3) a recursive improvement loop compounding gains to 0.5–1.5 dB total.

We argue that targeted, layer-level weight editing does not require superintelligence: current frontier LLMs have enough architectural reasoning and numeric output capability to propose simple structured edits (shift a bias, scale a filter norm, adjust a skip connection). What current LLMs cannot do is generate new distributed representations from scratch—that still requires gradient-based training.

Image restoration is a hard testbed for WSP because it is a precise regression task with continuous outputs, stochastic sampling, and noisy perceptual metrics. But it is also a valuable testbed: artifacts are visually verifiable, models are small enough to sandbox, and failure modes are well-documented.

The practical decision rule: if your restoration model has a specific, identifiable quality gap (wrong color balance, checkerboard artifacts, edge halos), try WSP as a fast, cheap first step. If your model has a general quality gap (too blurry, hallucinates detail, fails on new degradation types), WSP will not help—collect data and fine-tune.

The recursive loop in C3 is a small but concrete step toward weight-space self-improvement. Whether it scales from targeted debugging to general capability improvement depends on whether LLMs can eventually reason about distributed representations—the same boundary that the Gemma3-270M math experiment [Takehi, 2026] first revealed. The next step is to run the experiments.

## References

RikiyaT. Qwen3.5-0.8B-weight-space-programming. Hugging Face model, 2026a. URL https:// huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming. Accessed: 2026- 11-01.

RikiyaT. lm\_understand-logs. GitHub repository, 2026b. URL https://github.com/RikiyaT/lm\_ understand-logs. Accessed: 2026-11-01.

T. Takehi. Sparks of weight-space programming: LLMs can natively read & write neural network weights. Blog post, 2026. URL https://rikiyat.github.io/blog/weight-space-programming/. Accessed: 2026-11-01.