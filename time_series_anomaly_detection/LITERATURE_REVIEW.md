# Time Series Anomaly Detection & Explanation — Research Landscape (2026)

A working literature map for planning a paper. Organized as: method eras →
explainability → foundation models/LLMs → evaluation crisis → open gaps you
could target as a novel contribution.

---

## 1. Taxonomy of detection approaches

Following the "Decade Review" taxonomy (Si et al., arXiv:2412.20512), methods
fall into three families:

| Family | Idea | Landmark methods |
|---|---|---|
| **Distance-based** | Flag points/subsequences far from their neighbours | KNN, Local Outlier Factor (2000), Matrix Profile / discords (2016) |
| **Density-based** | Flag points in low-density regions of a learned representation | One-Class SVM (2003), PCA-residual (2003), Isolation Forest (2008) |
| **Prediction-based** | Model "normal" dynamics, flag large forecast/reconstruction error | ARIMA, LSTM/GRU forecasters (2015+), Autoencoders & GANs (2016+) |

The repo's `anomaly_detection.py` (rolling z-score / IQR / trend-residual) is
a hand-built instance of the prediction-based family — reconstruction against
a rolling-mean "model" of normal behavior.

### Deep learning era (2018–2023)
- **Reconstruction-based**: LSTM-VAE, OmniAnomaly, USAD, TranAD (deep
  transformer encoder-decoder for multivariate TSAD).
- **Attention/Transformer-based**: Anomaly Transformer (association
  discrepancy between prior- and series-attention), TFAD (time-frequency
  decomposition + transformer).
- Known weaknesses flagged in the 2025 surveys: expensive training,
  instability under adversarial/rare perturbations, poor generalization
  across domains without retraining.

### Foundation models / zero-shot (2024–2026)
The newest wave reframes TSAD as a downstream task of a pretrained
time-series foundation model rather than a bespoke model per dataset:
- **General TS foundation models**: TimesFM, Chronos, MOMENT, TimeGPT,
  TEMPO, Time-MoE — mainly built/evaluated for forecasting, repurposed for
  anomaly scoring via forecast residuals.
- **TSAD-specific foundation approaches**: TimeRCD (arXiv:2509.21190) —
  pretrains on a large synthetic corpus with token-level anomaly labels and
  directly supervises *relative context discrepancy* rather than
  reconstruction error, beating prior zero-shot detectors on 14 datasets.
  STAR (arXiv:2510.16014) adapts general TS foundation models to TSAD via a
  lightweight "state-aware adapter" instead of full fine-tuning.
- **KAN-AD** (arXiv:2411.00278, ICML 2025) — not a foundation model, but
  relevant here as a *parameter-efficient* alternative: models normal
  behavior as a smooth function using Kolmogorov-Arnold Networks with
  truncated Fourier basis (swapped in for B-splines because B-splines were
  too locally sensitive → false positives). <1,000 parameters, ~15%
  accuracy gain over baselines, no explanation output.

**Framing for a paper**: the field is visibly mid-transition from
"one model per dataset" (2018-2023 deep learning era) to "one pretrained
model, zero/few-shot everywhere" (2024-2026). Positioning a paper relative
to this shift (e.g., "does the foundation-model paradigm hold up when we
also demand explanations, not just scores?") is a live, citable framing.

---

## 2. Explainability for TSAD

This is the newest and least mature sub-area — most surveys explicitly call
it a gap (see §4).

**Two distinct explanation goals**, per the industry-perspective paper
(arXiv:2502.05392):
1. **Why did the alert fire?** — justify the anomaly score itself
   (feature/timestep attribution, pattern description).
2. **What is the root cause?** — diagnose the underlying process change,
   which requires context beyond the flagged signal (related metrics,
   external events) — largely unaddressed in current academic work.

**Approaches so far**:
- **Post-hoc model-agnostic XAI** (LIME, SHAP) applied to anomaly scores —
  general surveys note these "fall short of specificity" and produce
  low-quality attributions on deep/temporal models, since they weren't
  designed for sequential structure.
- **LLM-as-explainer**:
  - **AXIS** (arXiv:2509.24378, 2025) — the most complete system found: a
    frozen LLM conditioned on three fused hints (raw numeric window,
    step-aligned embeddings from a pretrained TS encoder projected into
    LLM embedding space via learned cross-attention, and global "task
    prior" tokens), trained with a two-phase pipeline (encoder pretraining,
    then a lightweight "hint tuner"). Produces pattern-level, contextualized
    natural-language explanations, validated by both LLM-judge and human
    evaluation against several baselines (image/vision-LLM, AnomLLM
    variants). Ships its own benchmark built via a "procedural anomaly
    generation + multi-agent LLM labeling" pipeline (paired normal/abnormal
    series → Question Agent → Answer Agent → quality filtering).
  - **Can LLMs Understand Time Series Anomalies?** (ICLR 2025) and
    **"Delving into LLMs for Effective TSAD"** (OpenReview 2025) — probe
    *why* raw LLMs are weak at this: lossy time-to-text serialization,
    context-length limits, chunked inference causing "memory decay" and
    boundary artifacts at chunk edges.
  - **ChatAD** (2601.13546) — reasoning-enhanced TSAD via multi-turn
    instruction evolution (iteratively refining the LLM's diagnostic
    questions).
  - **Argos** (arXiv:2501.14170) — agentic approach: LLM autonomously
    generates and refines *rules* for anomaly detection rather than scoring
    directly, which is inherently more explainable (the rule *is* the
    explanation) but trades off flexibility.
  - **Vision-LLM approaches** (arXiv:2506.06836, "ViTs" arXiv:2510.04710) —
    render the series as a plot/image and let a vision-language model
    reason over it "like a human expert," sidestepping numeric
    serialization entirely.
- **Human-in-the-loop** (arXiv:2405.03234) — treats explanation/trust as an
  interactive loop rather than a one-shot generation problem.

**Gap called out repeatedly**: no standard benchmark or metric for
*explanation quality* in TSAD (accuracy metrics are mature; explanation
evaluation is ad hoc, usually LLM-judge + small human study, as in AXIS).
This is a strong opening for methodological contribution.

---

## 3. Evaluation is arguably the field's biggest current controversy

Worth a dedicated section in a paper's related work, because it affects how
you should evaluate anything you build:

- **PA-F1 (point-adjusted F1)**, the dominant TSAD metric for years, is
  now widely discredited: papers have shown **random anomaly scores beat
  state-of-the-art detectors** under PA-F1, because adjusting an entire
  ground-truth anomaly range to "correct" once any one point in it is
  flagged systematically inflates scores and rewards low-precision/
  high-recall guessing.
- Analysis of 37 metrics in the literature found **none satisfy all
  desirable properties**, explaining why published rankings of methods are
  often inconsistent across papers (arXiv:2510.17562).
- Proposed alternatives: **PATE** (proximity-aware evaluation with buffer
  zones, arXiv:2405.12096), **DQE** (semantic-aware metric, 2603.06131),
  "balanced point adjustment," decay-function-based scoring
  (arXiv:2305.09691).
- **Streaming vs. batch mismatch**: academic benchmarks give the model the
  full series at once; production systems must score each point causally
  using only past data. The industry-perspective paper argues this
  invalidates a lot of comparative academic results outright.

**Practical implication for your own experiments**: if you build/evaluate a
detector for the paper, report at least one metric beyond PA-F1 (e.g. PATE,
or plain event-based F1 without point adjustment), and be explicit about
whether your evaluation is causal/streaming or batch — reviewers in this
subfield now actively check for this.

---

## 4. Explicitly named open problems (good "gap" statements to cite)

From the industry-perspective paper (arXiv:2502.05392) and cross-referenced
against the surveys:

1. **Root-cause explanation**, not just alert justification — almost
   entirely unaddressed.
2. **Conditional/contextual anomalies** — e.g. a spike that's *expected*
   given an external covariate (temperature, promotion calendar) shouldn't
   fire, but most detectors are univariate and context-blind.
3. **Streaming-valid evaluation protocols** — most benchmarks are batch;
   almost no published TSAD work evaluates under realistic causal
   constraints.
4. **Human-in-the-loop feedback** — almost no academic work models how a
   human's accept/reject of an alert should update the detector.
5. **Explanation-quality benchmarks/metrics** — detection accuracy has
   mature metrics (however flawed); explanation quality does not.
6. **Signal preprocessing** (periodicity detection, irregular-event
   resampling, threshold selection) — described as "critical but
   unstudied," oddly neglected relative to modeling novelty.
7. **Fragmentation** — different subfields (stats, ML systems, LLM/NLP)
   publish on disjoint benchmarks/baselines, making the literature hard to
   compare (Decade Review's core complaint).

Any of #1, #2, #5 pairs well with the "detect + explain" theme you're
already building in this repo, and each is explicitly called out as
under-addressed rather than solved — i.e. defensible as a paper's
contribution rather than incremental restatement of AXIS/KAN-AD.

---

## 5. Suggested angles for *your* paper

Given the direction of this project (lightweight, from-scratch detectors +
plain-language explanations of anomaly *type*), a few positioning options,
roughly cheapest → most novel:

- **A**: Empirical comparison paper — benchmark classical (z-score/IQR),
  KAN-AD-style, and LLM-based (AXIS-style) explanations on the same data
  under a *non-PA-F1* streaming-valid protocol, and score explanation
  quality with a consistent rubric. Fills gap #3 + #5 directly.
- **B**: Extend the "explanation taxonomy" (spike / level-shift /
  volatility-change, as already implemented here) into a **rule-based,
  fully interpretable explainer** that's cheap enough to run alongside any
  detector (no LLM required) — a lightweight alternative to AXIS aimed at
  latency/cost-sensitive production settings, evaluated for explanation
  *fidelity* against AXIS's LLM-generated explanations as ground truth.
- **C**: Attack the **root-cause gap (#1)** directly — extend point-level
  "why is this a spike" explanations to multivariate root-cause
  attribution (which upstream series/covariate most plausibly explains the
  deviation), which no current paper does well.

Happy to help draft a related-work section, build experiments for whichever
angle you pick, or extend `anomaly_detection.py` into a testbed for
comparison (A) or a standalone interpretable explainer (B).

---

## 6. Reference list

- Si et al., *Dive into Time-Series Anomaly Detection: A Decade Review*,
  arXiv:2412.20512 — https://arxiv.org/html/2412.20512v1
- *Open Challenges in Time Series Anomaly Detection: An Industry
  Perspective*, arXiv:2502.05392 — https://arxiv.org/html/2502.05392v1
- *A Survey of Deep Anomaly Detection in Multivariate Time Series:
  Taxonomy, Applications, and Directions*, PMC11723367 —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/ (also MDPI Sensors
  25(1):190)
- *Deep Learning for Time Series Anomaly Detection: A Survey*,
  arXiv:2211.05244 — https://arxiv.org/html/2211.05244v3
- Zhou et al., *KAN-AD: Time Series Anomaly Detection with
  Kolmogorov–Arnold Networks*, ICML 2025, arXiv:2411.00278 —
  https://arxiv.org/abs/2411.00278
- *AXIS: Explainable Time Series Anomaly Detection with Large Language
  Models*, arXiv:2509.24378 — https://arxiv.org/abs/2509.24378
- *Can LLMs Understand Time Series Anomalies?*, ICLR 2025 —
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/05774fb74e863308c4b460c9f49f6918-Abstract-Conference.html
- *Large Language Models can Deliver Accurate and Interpretable Time Series
  Anomaly Detection*, KDD 2025 —
  https://dl.acm.org/doi/10.1145/3711896.3737239
- *Delving into Large Language Models for Effective Time-Series Anomaly
  Detection*, OpenReview — https://openreview.net/forum?id=6rpy7X1Of8
- Argos: *Agentic Time-Series Anomaly Detection with Autonomous Rule
  Generation via Large Language Models*, arXiv:2501.14170 —
  https://arxiv.org/pdf/2501.14170
- *Harnessing Vision-Language Models for Time Series Anomaly Detection*,
  arXiv:2506.06836 — https://arxiv.org/pdf/2506.06836
- TimeRCD: *Towards Foundation Models for Zero-Shot Time Series Anomaly
  Detection: Leveraging Synthetic Data and Relative Context Discrepancy*,
  arXiv:2509.21190 — https://arxiv.org/html/2509.21190v1
- STAR: *Boosting Time Series Foundation Models for Anomaly Detection
  through State-aware Adapter*, arXiv:2510.16014 —
  https://arxiv.org/pdf/2510.16014
- PATE: *Proximity-Aware Time series anomaly Evaluation*, arXiv:2405.12096
  — https://arxiv.org/html/2405.12096
- *Formally Exploring Time-Series Anomaly Detection Evaluation Metrics*,
  arXiv:2510.17562 — https://arxiv.org/html/2510.17562v1
- *A Reliable Framework for Human-in-the-Loop Anomaly Detection in Time
  Series*, arXiv:2405.03234 — https://arxiv.org/pdf/2405.03234
- Curated lists: [Awesome-Anomaly-Detection-Foundation-Models](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models),
  [Awesome-Time-Series-Explainability](https://github.com/JHoelli/Awesome-Time-Series-Explainability)

*Note: several arXiv IDs above (e.g. 2601.x, 2602.x, 2603.x) are from
2026 preprints per current search results — worth double-checking
publication/venue status before citing in a formal submission, since very
recent preprints may still be under review.*
