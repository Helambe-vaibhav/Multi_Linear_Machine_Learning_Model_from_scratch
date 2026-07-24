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

## 2.5 Root-cause analysis for multivariate time series (chosen direction)

This is the closest prior art to `root_cause_analysis.py`, and it is more
crowded than gap #1 in §4 initially suggested — read this before framing a
paper's contribution as "the first to do RCA for TSAD."

- **AERCA** (ICLR 2025, OpenReview 6fde96479648d71e4fd9724374bf76eb) — the
  most direct match. Frames anomalies as **interventions on exogenous
  variables**: it jointly (a) learns Granger-causal structure among the
  series under normal conditions and (b) models the expected distribution
  of each series' exogenous/innovation term, then flags the variable whose
  exogenous term deviates most as the root cause. Conceptually this is a
  neural, distributional generalization of exactly the
  correlation + Granger-F-test + co-occurrence combination implemented
  here — worth reading in full before writing related work, since it will
  likely be a required citation and possibly a baseline to beat.
- **Neural Granger causal discovery for microservice RCA** (AAAI 2024) and
  **"Root Cause Analysis for Microservices based on Causal Inference"**
  (arXiv:2408.13729) — apply learned (neural) Granger causality to rank
  which microservice metric caused an incident, in the AIOps/microservices
  setting specifically rather than general multivariate TSAD.
- **PyRCA** (arXiv:2306.11417, Salesforce) — an open-source library
  bundling several classical metric-based RCA algorithms (correlation-based
  ranking, Bayesian-network structure learning, ε-diagnosis) behind one
  API; useful as a baseline suite and for realistic evaluation protocols,
  though the paper itself is a systems/library paper, not a new algorithm.
- **Causelens** (IEEE/ACM IWQoS 2025) — causality-based, explicitly
  *interpretable* RCA for microservices; relevant if you want prior art on
  the interpretability angle specifically, not just detection accuracy.
- **Entropy Causal Graphs for Multivariate TSAD** (arXiv:2312.09478) and
  **Root Cause Analysis with Latent Confounders using Partial Ancestral
  Graphs** (arXiv:2606.20912) — address the confounder problem explicitly
  (the same caveat in this repo's README "Limitations" section) using
  causal-graph discovery (PC-like algorithms, ancestral graphs) instead of
  pairwise Granger tests. This is the most direct answer to "what if two
  candidates share a hidden common cause?" — a question pairwise Granger
  causality (as implemented here) cannot resolve on its own.
- **Agentic/LLM RCA systems** (KRCA arXiv:2607.01788, TopoEvo
  arXiv:2605.15611, OpenRCA ICLR 2025) — newest wave, using multi-agent
  LLMs over telemetry/logs/topology rather than pure time-series statistics;
  relevant context for where the field is heading, less relevant as a
  direct baseline for a from-scratch statistical method.

**What's actually still open, given this crowd**: production-grade RCA is
dominated by (a) black-box neural causal discovery (AERCA-style) or
(b) LLM/agentic systems with heavy infrastructure (topology, logs). A
**lightweight, fully transparent, pairwise-interpretable RCA method with no
learned parameters** — every number in its output traceable to a lag, a
correlation, an F-statistic — sits in a real gap: cheap enough to explain
to a non-ML on-call engineer, and honest about being pairwise/correlational
rather than a full causal graph. That transparency-vs-power tradeoff, made
explicit and empirically measured against AERCA/PyRCA baselines, is a
defensible framing rather than a claim of beating them on raw accuracy.

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

## 5. Chosen direction: Option C — root-cause analysis

Status: **in progress.** `root_cause_analysis.py` now implements this angle:
given a target series' detected anomaly and a set of candidate
upstream/related series, it ranks candidates by combining (1) lagged
cross-correlation, (2) a Granger-causality F-test built on plain OLS (the
same linear-regression machinery as the repo's core notebook, applied to
two nested multi-linear models), and (3) whether the candidate showed its
own co-occurring anomaly. Validated on a synthetic demo with one true
lagged cause and three adversarial decoys (shared-seasonality-only,
unrelated-anomaly, pure-noise) — the true cause is correctly ranked first
by a wide score margin. See `README.md` "Root-cause analysis" section for
the full writeup and demo plot.

Per §2.5, this sits deliberately on the *lightweight/transparent* end of
the RCA spectrum vs. AERCA (neural, distributional) and agentic/LLM
systems (KRCA, TopoEvo) — every score component is directly attributable
(a lag, a correlation coefficient, an F-statistic), at the cost of being
pairwise/linear rather than a full causal graph and not handling
confounders. That tradeoff is the paper's likely contribution framing, not
"we detect root causes first."

The explainability layer (score decomposition, comparative ranking
narrative, attribution chart — see README "Explainability" section) is
also done: every candidate's score is an exact sum of three named terms,
which sidesteps the SHAP/LIME approximation problem entirely rather than
solving it post-hoc.

**Real-world detector validation done (including a shape-based detector),
RCA validation still pending.** `evaluate_ucr.py` ran four detectors
against the real UCR Anomaly Archive (250 files): three point-based
methods (10.0% z-score, 6.0% IQR, 10.8% trend-residual top-1 hit rate) plus
a new **shape discord detector** (`shape_discord_detection.py`) built via
"point extension" - z-normalized Euclidean subsequence matching (MASS/FFT),
the same family of technique as Matrix Profile discord discovery, the
actual reference method this archive was validated against. Shape discord
scored **20.4%** — nearly double the best point-based method — and any of
all four methods agreeing reached **31.2%**, vs. 16.8% for the point-based
three alone and a 0.84% random baseline.

**The key finding is complementarity, not superiority**: 36 files were
caught *only* by shape discord (mostly ECG/apnea-ECG/gait morphology
cases), but 27 were caught *only* by a point-based method (mostly CIMIS/
GP/Lab/STAFFIIIDatabase/CHARISten cases where a fixed subsequence length
`m=100` doesn't resolve the anomaly well, or where a value-level shift
really is the better description). Only 4 files had all four agree. This
is a stronger, more specific paper contribution than "shape-aware methods
beat point methods" — it's "different anomaly *types* need different
detector *families*, on the same benchmark under the same protocol,"
which motivates an ensemble/detector-selection argument rather than a
single best detector. Full writeup, category breakdown, and example plots
(including a side-by-side pattern comparison for a file every point-based
method missed) are in README.md "Real-world validation" and "Shape-based
detection".

This is a citable, honest limitation for the paper, not a result to hide:
it demonstrates precisely the gap between simple detectors and the
shape-aware/foundation-model methods discussed in §1, and it means a paper
claiming RCA or explanation quality should be explicit that those
components are detector-agnostic wrappers, not a fix for base detection
recall. **Root-cause validation on real multivariate data is still open**
— UCR has no covariates, so a real RCA test needs a different dataset
(SMD, MSAP/MSL, or PSM — all have multiple correlated sensor channels and
are used as AERCA/PyRCA baselines).

**Fifth detector added and now validated on the full 250-file archive.**
`fourier_embedding_detection.py` addresses shape discord's biggest named
weakness — a hand-picked fixed window length `m` — by setting `m` from
the period of the series' own dominant FFT frequency, then embedding every
window as its z-normalized Fourier-magnitude spectrum and visualizing
normal-vs-anomaly separation via a from-scratch PCA (plain
`numpy.linalg.svd`). The initial 3-file pilot (2/3 hits) motivated a full
run, which is now the strongest result in the project:

**34.8% (87/250) top-1 hit rate — the best of all five detectors, by a
statistically significant margin.** Paired McNemar's tests (same 250
files under each method) confirm this isn't noise: Fourier embedding vs.
trend-residual (best point-based method) gives χ²=43.5, p<0.0001; vs.
shape discord gives χ²=14.6, p=0.0001. Combining all five methods ("any
agrees") reaches 50.8% (127/250) — the first time more than half the
archive is caught by at least one method. Only 3 of 250 files had all
five agree, meaning the five detectors still disagree almost completely
about *which* points are anomalous even though each clears chance by a
wide margin individually. 49 files were caught *only* by Fourier
embedding. Category breakdown explains why: it jumps to 81.3% on ECG,
84.6% on InternalBleeding, 80% on GP, 66.7% on CIMIS — domains with strong
natural periodicity, exactly what an FFT-derived window size and spectral
embedding are built to exploit — but scores 0% on non-periodic-dominant
domains (gait, apneaecg, taichidbS, tilt).

Two engineering notes worth keeping for the paper's methods section: (1)
a naive implementation of "auto window size from FFT" crashed via OOM on
a real file (a ~300k-point series produced a spurious ~60,000-sample
"period" from trend leakage into low-frequency bins) — fixed by linearly
detrending and Hann-tapering before the period-estimation FFT, plus
bounding memory by BOTH sample count and window length, not just sample
count. This is a genuinely useful cautionary result: "auto-selected
window/scale hyperparameters need their own robustness testing" is a
citable methods point, not just a debugging footnote. (2) The correct
comparison for "is method A really better than method B" here is a
**paired** test (McNemar), not comparing confidence intervals by eye —
this is exactly the kind of evaluation rigor the field's own literature
(§3) criticizes TSAD papers for skipping.

Full writeup, statistical tests, and category breakdown are in
README.md's "Real-world validation" section.

**Concrete next steps toward a paper draft:**
1. **Real or more realistic synthetic benchmarks.** The current demo has
   one obvious cause; a paper needs harder cases — multiple simultaneous
   candidate causes, confounded pairs (shared hidden driver), mediator
   chains (A causes B causes target), and no-clean-answer cases. Consider
   adapting AERCA's or PyRCA's evaluation datasets for a head-to-head
   comparison.
2. **Confounder handling.** Right now this is pairwise Granger — add a
   partial/conditional variant (control for other candidates when testing
   each pair) as an ablation, and cite the ancestral-graph line
   (arXiv:2606.20912) as the "what a full solution would need" contrast.
3. **Evaluation metric for RCA itself**: precision@1 / precision@k of the
   ranked candidate list against known ground-truth causes (used in
   AERCA/PyRCA-style papers) — decide this before running experiments so
   results are comparable to baselines.
4. **Baselines to run against**: at minimum, plain (lag-0) correlation
   ranking and PyRCA's bundled methods, to demonstrate the lagged/Granger
   combination's marginal value over naive correlation.
5. **Ablations**: score with each of the three signals alone vs. combined,
   to justify the combination rather than asserting it.

Happy to help draft the related-work section around §2.5, design the
harder synthetic benchmarks (confounders/mediators) for step 1, or extend
`root_cause_analysis.py` with a conditional/partial-Granger variant for
step 2.

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
- **AERCA**: *Root Cause Analysis of Anomalies in Multivariate Time Series
  through Granger Causal Discovery*, ICLR 2025 —
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/6fde96479648d71e4fd9724374bf76eb-Abstract-Conference.html
- *Root Cause Analysis for Microservices based on Causal Inference*,
  arXiv:2408.13729 — https://arxiv.org/pdf/2408.13729
- PyRCA: *A Library for Metric-based Root Cause Analysis*, arXiv:2306.11417
  — https://arxiv.org/pdf/2306.11417
- *Entropy Causal Graphs for Multivariate Time Series Anomaly Detection*,
  arXiv:2312.09478 — https://arxiv.org/pdf/2312.09478
- *Root Cause Analysis with Latent Confounders using Partial Ancestral
  Graphs*, arXiv:2606.20912 — https://arxiv.org/pdf/2606.20912
- KRCA: *An Efficient Root Cause Analysis System in Hyper-Scale
  Microservice Systems via Agentic AI*, arXiv:2607.01788 —
  https://arxiv.org/pdf/2607.01788
- TopoEvo: *A Topology-Aware Self-Evolving Multi-Agent Framework for Root
  Cause Analysis in Microservices*, arXiv:2605.15611 —
  https://arxiv.org/pdf/2605.15611
- Wu, R. & Keogh, E., *Current Time Series Anomaly Detection Benchmarks are
  Flawed and What to Do About It* (introduces the UCR Anomaly Archive used
  in `evaluate_ucr.py`), 2021 — archive:
  https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
- Yeh, C-C. M. et al., *Matrix Profile I: All Pairs Similarity Joins for
  Time Series* (introduces the Matrix Profile / MASS technique that
  `shape_discord_detection.py` implements a lightweight FFT-based version
  of), ICDM 2016.
- Mueen, A. et al., *The Fastest Similarity Search Algorithm for Time
  Series Subsequences under Euclidean Distance* (MASS algorithm) —
  reference: https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
- Curated lists: [Awesome-Anomaly-Detection-Foundation-Models](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models),
  [Awesome-Time-Series-Explainability](https://github.com/JHoelli/Awesome-Time-Series-Explainability)

*Note: several arXiv IDs above (e.g. 2601.x, 2602.x, 2603.x) are from
2026 preprints per current search results — worth double-checking
publication/venue status before citing in a formal submission, since very
recent preprints may still be under review.*
