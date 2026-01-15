# Road to GFM4ET Journal

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