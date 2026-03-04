# Road to GFM4ET

- tried next point prediction => caught trends
- moved on to classification/regression
- emotion classification
- created class for HeteroGraph: SpatioTemporalDataset
- trying simple GCN on spatiotemporaldataset (no MLP preprocessing)

## First results for inferring emotions from eye-tracking data

- eSEEd_v2 only
- whole dataset
- output: 4 scalar emotions (regression)

### Baselines

| Model         | MSE      | MAE    | SD_Err | R²      | D²     | Pearson R |
|---------------|----------|--------|--------|---------|--------|-----------|
| MeanEstimator | 12.4251  | 3.0708 | 3.5243 | 0.0634  | 0.0634 | 0.2526    |
| SVM           | 15.2372  | 2.7305 | 3.5730 | -0.1486 | -0.1486| 0.2135    |
| GaussianNB    | 20.3462  | 3.8823 | 4.2364 | -0.5337 | -0.5337| -0.0329   |
| LightGBM      | 10.8878  | 2.7743 | 3.2984 | 0.1793  | 0.1793 | 0.4243    |

### Spatio Temporal Hetero GCN

| Model                      | MSE     | MAE    | SD_Err | R²     | D²     | Pearson R |
|----------------------------|---------|--------|--------|--------|--------|-----------|
| Spatio Temporal Hetero GCN | 11.8035 | 2.9173 | 3.4356 | 0.1128 | 0.1128 | 0.3359    |

#### Data params

- window_length=10,
- kt=2,
- ks=2,

#### Model params

- 2 GCN conv layers
- self.head = nn.Sequential(
    nn.Linear(hidden_channels, hidden_channels),
    nn.ReLU(),
    nn.Linear(hidden_channels, out_channels),
    nn.Sigmoid()  # Output in [0, 1], will scale to [0, 10]
    )

#### Added preprocessing MLP and LOO splits

28.12.2025

- preprocess MLP added
- dropout added
- sum aggr -> mean aggr
- self_loops = False
- LOO splits
    - subject
    - recording
    - combined

- training: 100 epochs, 3 LOO variants, win_len = 10, kt=ks=2
- Final Comparison Across All Strategies

| Strategy      | MSE     | MAE    | SD_ERROR | R²       | Pearson R |
|---------------|---------|--------|----------|----------|-----------|
| subject_loo   | 12.2358 | 2.9737 | 3.1599   | -0.5041  | 0.3185    |
| recording_loo | 13.5568 | 3.1660 | 3.2017   | -1.0884  | 0.0446    |
| combined_loo  | 13.5180 | 3.1524 | 2.5973   | -3.5471  | 0.0040    | 

#### Parameter search 

29.12.2025 

Doing parameter search over 50 random samples for the following params:
- window_length: [5, 10, 30, 60]  # Random choice from list
- kt: [1, 30]  # Random integer in range [1, 30]
- ks: [1, 6]  # Random integer in range [1, 6]
- hidden_channels: [32, 64, 128, 256, 512]  # Random choice from list
- use_preprocess_mlp: [true, false]  # Random choice from list
- add_self_loops: [true, false]  # Random choice from list

Expected time of search: 5-10min one config & 50 configs ==> time < 500min < 9h.

Number of training iterations:
- 10 epochs
- subset of first 10 subjects, all 10 recordings => 10x10 grid
- LOSO => 10 epochs * 10 subjects = 100 epochs
- LORO => 10 epochs * 10 recordings = 100 epochs
- combined LOO => 10 epochs * 10 rec * 10 subj = 1000 epochs
- total: 1200 epochs * (max) 5 seconds * 50 configs = (max) 84 hours
- we should probably drop the combined LOO for param search
- if we drop combined LOO => 83% less work => (max) 15 hours

**Results of param. search:**
	1.	k_t: Pearson r — večji k_t je bolje
	2.	k_s: pri RecordingLOO izgleda, da večji k_s pomaga, vendar je to lahko specifika podatkovne baze, ker so vsi subjekti gledali iste filmčke v istem vrstnem redu
	3.	velikost okna: po vseh metrikah so manjša okna boljša (5–10 s)
	4.	preprocess MLP: ne kaže očitnih znakov pomoči
	5.	self loops: ne kažejo očitnih znakov pomoči


#### Classical ML baselines to beat

| Strategy        | Baseline      | MSE     | MAE    | SD_Error | R²       | Pearson R |
|-----------------|---------------|---------|--------|----------|----------|-----------|
| subject_loo     | MeanEstimator | 12.7008 | 3.1039 | 3.2169   | -0.6298  | 0.2611    |
**| subject_loo   | SVM           | 14.6872 | 2.7996 | 3.2816   | -0.5008  | 0.2699    |**
| subject_loo     | GaussianNB    | 18.8645 | 3.7430 | 3.8806   | -2.0684  | 0.0011    |
| subject_loo     | MLP           | 13.9244 | 3.0553 | 3.3636   | -1.0275  | 0.2614    |
| subject_loo     | LightGBM      | 13.0676 | 3.0373 | 3.2700   | -0.7352  | 0.2721    |
|                 |               |         |        |          |          |           |
| recording_loo   | MeanEstimator | 13.7904 | 3.2620 | 3.2170   | -1.2703  | 0.0715    |
**| recording_loo | SVM           | 16.5039 | 3.0595 | 3.4087   | -1.2817  | 0.0521    |**
| recording_loo   | GaussianNB    | 19.5374 | 3.7891 | 3.7776   | -2.5449  | -0.0273   |
| recording_loo   | MLP           | 15.2606 | 3.2298 | 3.4546   | -1.6540  | 0.0484    |
| recording_loo   | LightGBM      | 14.8453 | 3.2624 | 3.3680   | -1.5904  | 0.0556    |
|                 |               |         |        |          |          |           |
**| combined_loo  | MeanEstimator | 13.6940 | 3.2437 | 2.5945   | -4.4296  | 0.0825    |**
| combined_loo    | SVM           | 17.0358 | 3.0881 | 2.8063   | -3.2391  | 0.0079    |
| combined_loo    | GaussianNB    | 19.7463 | 3.8350 | 3.1807   | -10.1552 | -0.0188   |
| combined_loo    | MLP           | 16.4226 | 3.3535 | 2.9072   | -5.4677  | -0.0330   |
| combined_loo    | LightGBM      | 15.4480 | 3.3348 | 2.8231   | -5.2698  | -0.0121   |

We need to beat MAE of 3 for all CVs, 0.26 pearson r for LOSO and 0.10 for LORO and combined LOO.

- should have normalized feats for SVM, forgot


#### Rezultati prečnega preverjanja

Baselines
Split	Model(s)	MAE	Pearson r
Subject LOO	Mean, SVM, LigthGBM, MLP	3.0	0.26
Recording LOO	Mean, SVM, LigthGBM, MLP	3.2	0.05
Combined LOO	Mean	3.2	0.08

GNN
Za izbrane “najboljše” parametre. Treniranje sem omejil na 10 epoh, ker sem opazil, da med 10 in 100 epoh ni velike razlike v rezultatih.
Split	MAE	Pearson r
Subject LOO	3.0	0.33
Recording LOO	3.2	-0.03
Combined LOO	3.2	0.03

**Comment from Gašper**: Pearson r may not be very suitable to observe since we don't care about the temporal order of emotions but rather about accuracy of predictions (MAE, maybe accuracy/f1/auc if we discretize).

### Analysis of model outputs (per-emotion)
Analysis done in `src/emotions/results_analysis/analyse-mode-outputs.ipynb`.
#### Baseline models
For all models except Gaussian NB, the models regress the intensity (float). Gaussian NB predicts classes.

- MeanEstimator
    - always predicts mean of train set (slightly variable across folds)
- LightGBM
    - subject LOO
        - is bounded to 0-10 range
        - predictions normally distributed around the mean of the emotion
    - recording LOO
        - similar as above
        - flatter gaussians
- GaussianNB
    - subject LOO
        - predicts integers (classes)
        - most common prediction: 4
        - bad performance
    - recording LOO
        - similar as above
- MLP
    - subject LOO
        - predictions not bounded to 0-10! (could add that to model definition); range is big (-10)-20
        - otherwise predictions similar to LightGBM's
    - recording LOO
        - similar as above
        - prediction range is bigger (-15)-30
- SVM
    - subject LOO
        - predictions not bounded to 0-10, but close; (-2)-10
        - lower intensities overall
        - skewed normal distributions
        - interesting for tenderness: almost always predicts ~0
    - recording LOO
        - similar as above
        - flatter gaussians

#### Spatio Temporal Hetero GCN
- subject LOO
    - predictions correctly bounded to 0-10 
    - not normally distributed, but quite a "smooth" distribution
    - peak aruond 3-4 for all but tenderness
- recording LOO
    - similar distribution as above but less "smooth"
- difficult subjects based on MAE
    - average MAE = 3
    - a lot of subjects (>20) have MAE > 4 which means they are like random because max sensible MAE would be 5 if we predicted 5 all the time (scale 0-10)
    - conclusion: can't expose any truly difficult subjects, except potentially subject 15 for emotion sadness only


## EDA of eSEEd_v2
- 1.32 % of rows are NaN
    - subeject 3: 19.5%
    - subject 15: 16.9%
- x, y, pupil sizes are distributed normally
    - x, y around 0.5 with std ~0.1
    - pupil size around 4 with std ~0.8
- emotions (0-10) are not normally distributed
    - 0 is very dominant
    - otherwise looks near random, while emotion-dependent
    - anger has a lot of higher values
    - tenderness has very little higher values
    - sadness and disgust seem quite random
- average emotion per recording
    - recordings 4 and 5 are completely missmatched (reported vs intended emotion)
    - recordings 8, 7, 6 are somewhat matched with intended 
- scatter plots of x vs y colored by emotion
    - don't indicate any obvious position-related bias 
- average emotion intensity by pupil size bins
    - sadness average emotion is the same for all bins
    - disgust average intensity drops with large or small pupil size
    - anger average intensity is slightly lower for average pupil size
    - tenderness has overall lower intensity than others, dropping especially with larger pupil
- average pupil size by emotion intensity
    - very random (uniformly distributed)
- unique emotion intensity looks okayish
- 48/480 (10%) subject-recording combinations report total neutrality, i.e., all emotions are 0!
- weird average pupil sizes for subject-recording pairs
    - s-r = 6-10: left = 2.7, right = 7,6
    - s-r = 6-1: left = 4.5, right = 2.5
    - s-r = 13-8: left = 5.3, right = 10.5
    - and many more...
- subject outliers
    - 41 - all emotions very high all the time except tenderness
    - 48, 39, 26, 5 - all emotions extremely low all the time
- recording outliers
    - 2, 5 - tenderness high, other low - intended for 2, not for 5

Idea: remove data with "bad" emotion reports. -- What is bad? 

## Which metrics to use and why?
- MAE for absolute errors
- correlations:
    - Spearman ro coeff. for emotion intensity ranking across recordings
    - CCC for absolute intensity agreement
    - both above are permutation invariant
    - calculate them on aggregated predictions


## What to infer?
The data is labelled with intensities (0-10) for 4 distinct emotions. We contemplate different options.

#### Option A — 4 continuous intensities (multi-target regression)
- **Label**: y ∈ R^4 (sadness, anger, tenderness, disgust), e.g., 1–10 (or 0–10)
- **Train**: one model outputs 4 numbers per recording (after temporal/graph pooling)
- **Pros**: no information loss; supports mixed emotions; aligns with dataset’s native labels
- **Cons**: subject rating-scale bias; noisier supervision; needs careful normalization/metrics
- **Good metrics**: MAE/RMSE + Spearman ρ; CCC if you care about calibration

#### Option B — Discretize each emotion into levels (4 parallel classifiers)
- **Label**: for each emotion: {neutral, low, medium, high} (or {0,1,2,3})
- **Train**: 4 heads (or one multi-head) classify per-emotion level
- **Pros**: more robust to label noise; easier interpretation than exact numbers
- **Cons**: threshold choices arbitrary; class imbalance likely; loses continuous info
- **Tip**: use **ordinal classification** loss (better than plain softmax)

#### Option C — Single dominant emotion + neutral (multi-class)
- **Label**: argmax emotion among 4, plus **neutral** if all intensities < τ
- **Train**: 5-class classifier (sad/ang/tend/disg/neutral)
- **Pros**: simple; comparable to prior work; easy reporting
- **Cons**: discards mixed states; sensitive to small top-2 differences; needs neutral threshold τ
- **Tip**: add **mixed/uncertain** class if top-2 are close

#### Option D — Two-stage: neutral detection then emotion estimation
- **Label**: Stage 1: neutral vs emotional; Stage 2: (A) regress 4 intensities or (B) classify levels
- **Train**: pipeline or multitask
- **Pros**: handles neutral dominance; reduces imbalance for stage-2; clearer decision logic
- **Cons**: error propagation; more complexity
- **Use when**: many recordings are low-intensity / near neutral

#### Option E — Multi-label presence/absence (4 binary labels)
- **Label**: for each emotion: present (1) if intensity ≥ τ_e else absent (0)
- **Train**: 4 sigmoid outputs; optionally add neutral = none present
- **Pros**: supports co-occurrence; simpler than full regression
- **Cons**: still threshold-dependent; ignores intensity magnitude
- **Upgrade**: ordinal multi-label (presence + intensity bins)

#### Option F — Relative / within-subject labels (rank or above-median)
- **Label**: per subject, transform intensities to ranks or z-scores; or binary “above subject median”
- **Train**: regress normalized targets or classify relative levels
- **Pros**: reduces subject-scale bias; often improves cross-subject generalization
- **Cons**: loses absolute meaning (“7” vs “3”); requires per-subject statistics
- **Great for**: LOSO evaluation where rating styles differ widely

#### Option G — Derived Valence/Arousal targets (auxiliary or primary)
- **Label**: map emotions to VA (binary/3-class/continuous), predict valence and arousal
- **Train**: VA-only or multitask (VA + 4D intensities)
- **Pros**: coarser labels can be easier; useful regularizer; common in literature
- **Cons**: less aligned with 4-emotion intensities; mapping can be simplistic
- **Use as**: auxiliary task to stabilize representation learning

#### Option H — Distributional / soft targets over emotions
- **Label**: convert intensity vector to soft distribution p via softmax-like transform
- **Train**: predict p with KL-div / cross-entropy (optionally also regress totals)
- **Pros**: keeps mixed-emotion structure; avoids brittle argmax
- **Cons**: p is not “true probability”; depends on temperature/scale choice
- **Good when**: you want classification-style training but retain intensity information

#### Option I — Ordinal regression (per emotion)
- **Label**: ordered levels per emotion (e.g., 0–10, or {neutral, low, medium, high})
- **Train**: for each emotion, predict K−1 threshold probabilities P(y ≥ k) with an ordinal loss (e.g., CORAL / cumulative BCE)
- **Pros**: preserves ordering without assuming equal spacing; often more robust than pure regression for subjective ratings
- **Cons**: slightly more complex; can suffer if some levels are very rare (imbalance across thresholds)
- **Good when**: labels are ordered, subjective, and constant per recording; you want “intensity-aware” learning without arbitrary binning

### Let's go step by step

Simplify the problem as much as possible. Advance to the next step only when the current step works!

0. Drop all NaNs, maybe interpolate the short sections. Combine GNN and baseline training to run both always (because baseline is fairly inexpensive and we are currently changing inputs a lot). Pick only the best few baselines: Mean, SVM, LightGBM. Check big pupil size differences!
1. Simplify to one emotion only (e.g., anger), drop other emotions. 
2. Binary classification: 0 vs. >0 (for only one emotion).
3. "something" on labels 1-10. Figure out that "something". Is it regression, ordinal regression, classification, ...?
4. Two-stage model:
    1. Binary 0 vs. >0.
    2. "something" on labels 1-10.
5. Think about presenting preliminary work on a workshop/poster.

## Binary classification on the **whole** eSEEd_v2 dataset
### Emotion: Anger
Complete results in `results/binary/RETAIN_2026-02-12_14-50-31`.
#### Recording LOO

| Model    | Accuracy | Precision | Recall | F1    | AUC   |
|----------|----------|-----------|--------|-------|-------|
| Mean     | 0.461    | 0.442     | 0.900  | 0.546 | 0.500 |
| SVM      | 0.448    | 0.535     | 0.791  | 0.550 | 0.556 |
| LightGBM | 0.469    | 0.539     | 0.735  | 0.541 | 0.564 |
| MLP      | 0.507    | 0.555     | 0.664  | 0.535 | 0.575 |
| GNN      | 0.522    | 0.522     | 1.000  | 0.636 | 0.506 |

#### Subject LOO

| Model    | Accuracy | Precision | Recall | F1    | AUC   |
|----------|----------|-----------|--------|-------|-------|
| Mean     | 0.570    | 0.570     | 1.000  | 0.704 | 0.479 |
| SVM      | 0.536    | 0.563     | 0.812  | 0.627 | 0.512 |
| LightGBM | 0.546    | 0.578     | 0.739  | 0.618 | 0.513 |
| MLP      | 0.526    | 0.586     | 0.630  | 0.597 | 0.516 |
| GNN      | 0.570    | 0.570     | 1.000  | 0.704 | 0.558 |


## Dataset cleaning (eSEEd_v2)
### Pupil size problems
We noticed major problems with inconsistent pupil sizes. Left and right pupil size sometimes differ for a couple of millimeters at the same point in time. This is worse in some subjects than others. We removed datapoints with |MAE| between left and right pupil size > 1.5. Subjects that still had |MAE| > 0.5, were removed.
### Coordinates (x, y)
There were not (yet) many issues found with the coordinates' data. For now only some missing data and some big outliers. Outlier points - those with x or y outside (0,1) range were removed, then the subjects with a lot of missing data were removed (3,15,16,31 additionally to the previously removed subjects from pupil size cleaning).
### 31 subjects remaining
Remaining subjects: 1, 2, 4, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 35, 38, 41, 42, 43, 44, 45, 46, 47


## Binary classification on the **cleaned** dataset
- Emotion: Anger
    - `results/binary/RETAIN_2026-02-12_19-05-24`.
- Emotion: Disgust
    - `results/binary/RETAIN_2026-02-12_19-19-45`
- Emotion: Sadness
    - `results/binary/RETAIN_2026-02-12_19-34-11`
- Emotion: Tenderness
    - `results/binary/RETAIN_2026-02-12_19-45-27`

The results show the problem is still difficult. Majority baseline is rarely beaten. GNN has a serious issue - it collapses to always predicting 1 in all experiments (all emotions, both LOOs).

## GNN collapse
Debugging embedding collapse. 
#### Embedding Statistics (Batch 2/10, size=176)

| Stage | var_mean | cos_mean | L2_mean | within_var | Status |
|-------|----------|----------|---------|-----------|--------|
| input_raw | 1.36e+02 | 0.9207 | 29.9191 | 1.56e+00 | ✓ |
| preprocess_none | 1.36e+02 | 0.9207 | 29.9191 | 1.56e+00 | ✓ |
| conv1 | 2.67e+00 | 0.9232 | 22.1433 | 6.69e-02 | ⚠ Variance collapse |
| conv2 | 9.37e-01 | 0.9323 | 12.5929 | 2.73e-02 | ✓ |
| pool_g2 | 9.89e-01 | 0.9504 | 12.7663 | N/A | ✓ |
| head_logits | 1.11e-02 | 1.0000 | 0.1215 | N/A | ⚠ Variance collapse |
| head_prob | 6.48e-04 | 0.0000 | 0.0000 | N/A | ⚠ Variance collapse |

**Logits details**: min=−0.5743, max=−0.1745, range=0.3998, std=0.1052, ratio(std/range)=0.2630

#### Aggregate Metrics (Mean Across Batches)

| Stage | var_mean | std | cos_mean | L2_mean |
|-------|----------|-----|----------|---------|
| input_raw | 1.37e+02 | 1.87e+01 | 0.9202 | 29.4389 |
| preprocess_none | 1.37e+02 | 1.87e+01 | 0.9202 | 29.4389 |
| conv1 | 2.69e+00 | 2.64e+00 | 0.9205 | 21.8115 |
| conv2 | 9.44e-01 | 1.52e+00 | 0.9332 | 12.2126 |
| pool_g2 | 9.98e-01 | 1.58e+00 | 0.9510 | 12.6060 |
| head_logits | 1.13e-02 | 1.06e-01 | 1.0000 | 0.1221 |
| head_prob | 6.65e-04 | 2.58e-02 | 0.0000 | 0.0000 |

### Discussion

The embedding diagnostics indicate that the model suffers from an early-stage representation collapse. With preprocessing disabled, raw node features show substantial diversity both within graphs (`within_var ≈ 1.5`) and between graphs (`var_mean ≈ 1e2`). The first major loss of information then occurs at the first message-passing step (**conv1**), where `within_var` drops by ~20–25× (to ~6e−2), consistent with strong smoothing from neighbor aggregation. Given the graph construction (temporal neighbors + spatial kNN, unweighted edges), a GCN-style update can quickly homogenize node embeddings even with self-loops, suggesting oversmoothing begins immediately.

Importantly, **enabling the preprocess MLP yields a very similar overall picture**, but shifts the first major loss of information *one step earlier*: the preprocess MLP itself produces a large reduction in within-graph variance, after which the subsequent GNN layers continue smoothing. This suggests that the preprocessing stage can already compress subject/graph-specific variability (likely due to feature scale/unit mismatch and the MLP learning a dominant “common component”), making the downstream message passing start from a less informative representation.

Downstream, pooled graph embeddings (`pool_g2`) retain some variance, but the head output becomes nearly constant: `head_logits var_mean ≈ 1e−2` and logits lie in a narrow negative range (≈ −0.57 to −0.17), mapping to probabilities mostly below 0.5. The observed `cos_mean = 1.0` for `head_logits` is expected for a 1D output when logits share the same sign and should not be over-interpreted; the key issue is the **low logit spread** and resulting near-baseline predictions.

Overall, the results point to two interacting problems: (i) **early collapse** (either in the preprocess MLP when enabled, or in conv1 when preprocessing is disabled), and (ii) a **low-variance head regime** that yields near-constant outputs. This motivates focusing mitigation on the first collapsing step (feature normalization and/or residualized/normed preprocessing; reduced mixing strength/attention/weighted edges for conv1), rather than tuning later layers.

### What we tried 
What we tried in order. Each next step includes all previous steps unless stated otherwise. All were tried with and without preprocess mlp. 
1. z-score normalization of all 5 features (time, x, y, pupil-l, pupil-r)
    - the normalization was done per subject on all data (not on train and test separately) in order to do quick testing
2. drop time from nodes => we have only 4 features in nodes (and no edge weights)
3. add dt as edge weights (both edge types)
    - w = exp(-dt / tau) 
    - tau = 0.05 (meaning w0 = 1, w1 = 0.82, w2 = 0.67, ...)
    - note: internal self loops in pyg take care of w0 = 1 (always)
4. added LayerNorm to preprocess MLP
    - now: Linear -> GELU -> LN -> Dropout -> Linear -> LN
    - no embedding collapse in the preprocess MLP anymore!
    - results for emotion-tenderness:
        - within_var: 0.535 → 0.386 → 0.163 → 0.048 (smooth, not a sudden 20–30× crash)
        - prediction quartiles for [0.05,0.25,0.5,0.75,0.95] ~ [0.200, 0.366, 0.488, 0.547, 0.587]
        - ~48% of predictions are over 0.50
    - results for emotion-sadness a bit more clustered around 0.53 with std of 0.08

We assessed embeddings with `debug_embedding_collapse.py` - we log var_mean, std, cos_mean, L2_mean at each layer of GNN (raw input, (preprocess mlp), conv1, conv2, pool, head logits, head prob)

Results (embedding behavior) didn't change much until ... (still waiting)

## Transitioning to MAHNOB-HCI-TAGGING DATABASE 
### Papers using it
More references for this dataset than for eSEEd_v2 - more hope :)
- A Multimodal Database for Affect Recognition and Implicit Tagging 
    - Mohammad Soleymani; Jeroen Lichtenauer; Thierry Pun; Maja Pantic 
    - https://ibug.doc.ic.ac.uk/media/uploads/documents/taffcsi-2010-11-0112-2.pdf 
- Emotion recognition using eye gaze based on shallow CNN with identity mapping 
    - Shan Jin; Chunmei Qing; Xiangmin Xu; Yang Wang 
    - https://link.springer.com/chapter/10.1007/978-3-030-39431-8_7 
- Implicit Affective Video Tagging Using Pupillary Response 
    - Dongdong Gui; Sheng-hua Zhong; Ming Zhong 
    - https://link.springer.com/chapter/10.1007/978-3-319-73600-6_15 
- Multimodal insights into granger causality connectivity: Integrating physiological signals and gated eye-tracking data for emotion recognition using convolutional neural network 
    - Javid Farhadi Sedehi; Nader Jafarnia Dabanloo; Keivan Maghooli; Ali Sheikhani 
    - https://www.cell.com/heliyon/fulltext/S2405-8440%2824%2912442-5 

### EDA
- Ran quick EDA on `data/processed/hci-tagging/emotion-elicitation` (`942` sections, `24` subjects, `3,797,165` rows).
- Label coverage is solid: `3,214,000` rows are labeled (`emotion-derivation-status=ok`), `583,165` are unlabeled baseline/neutral periods (`not-reported`).
- Emotion labels are distributed across 9 classes (`0,1,2,3,4,5,6,11,12`), with largest classes: Neutral (`726,659`) and Amusement (`669,438`).
- Signal quality is good after robust filtering: outlier removal in sampled signal plots stayed low overall (~`1.0%` for `x`, `3.08%` for `y`, ~`1.6%` for pupil sizes).
- Pupil left/right consistency is strong (`corr=0.902`), with moderate subject-specific asymmetry outliers (largest mean abs diff: `P28=0.590`).
- Overall, this HCI emotion data looks much cleaner and more usable than eSEEd_v2 (better label availability and fewer severe preprocessing/pathology issues).


### Binary classifications
First, we will test the dataset with simple binary classification problems.
    [x] Binary: predict high vs low valence (emotion-valence)
    [x] Binary: predict high vs low arousal (emotion-arousaly)
    [x] Binary: emotion-predictability
    [x] Binary: emotion-control

#### Binary: predict high vs low valence
Looks like something finally works! :D
All the models perform similarly well, all beating the majority classifier, meaning some patterns can be learned. The GNN does not collapse, which is good. Confusion matrices "look OK" now. The numbers match between models and the diagonals are relatively strong.

| Model   | Accuracy | Precision | Recall | F1    | AUC   | Strategy    |
|---------|----------|-----------|--------|-------|-------|-------------|
| Mean    | 0.533    | 0.533     | 1.000  | 0.691 | 0.500 | subject_loo |
| SVM     | 0.640    | 0.662     | 0.767  | 0.674 | 0.694 | subject_loo |
| LightGBM| 0.628    | 0.652     | 0.729  | 0.658 | 0.702 | subject_loo |
| MLP     | 0.609    | 0.618     | 0.699  | 0.636 | 0.670 | subject_loo |
| GNN     | 0.643    | 0.702     | 0.695  | 0.646 | 0.738 | subject_loo |
||||||||
| Mean     | 0.405    | 0.397     | 0.750  | 0.462 | 0.400 | recording_loo  |
| SVM      | 0.592    | 0.548     | 0.581  | 0.521 | 0.435 | recording_loo  |
| LightGBM | 0.599    | 0.551     | 0.575  | 0.520 | 0.443 | recording_loo  |
| MLP      | 0.592    | 0.550     | 0.562  | 0.514 | 0.425 | recording_loo  |
| GNN      | 0.598    | 0.552     | 0.582  | 0.530 | 0.430 | recording_loo  |

#### Comparison of binary classifications 
We compared the aforementioned binary classifications: high vs low valence/arousal/predictability/control.
We experimented with a subset of data - only participants [P1, P8, P5, P4, P28, P2, P27].
We conducted only recording LOO.

Results (confusion matrices) show that control is the easiest to infer.

See results in `RETAIN_2026-03-04_13-42-52`, `RETAIN_2026-03-04_13-42-57`, `RETAIN_2026-03-04_13-46-09`, `RETAIN_2026-03-04_13-46-24`.

**Edge weights don't make a difference?**
We compared GNNs with and without edge weights on the valence and arousal binary classifications. There is no difference in confusion matrices between GNNs with and without edge weights. See results `RETAIN_2026-03-04_13-23-58` vs `RETAIN_2026-03-04_13-42-52` and `RETAIN_2026-03-04_13-30-17` vs `RETAIN_2026-03-04_13-42-57`.


