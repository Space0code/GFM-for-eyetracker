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

**Best params**


#### Classical ML baselines to beat

| Strategy      | Baseline      | MSE     | MAE    | SD_Error | R²       | Pearson R |
|---------------|---------------|---------|--------|----------|----------|-----------|
| subject_loo   | MeanEstimator | 12.7008 | 3.1039 | 3.2169   | -0.6298  | 0.2611    |
| subject_loo   | SVM           | 14.6872 | 2.7996 | 3.2816   | -0.5008  | 0.2699    |
| subject_loo   | GaussianNB    | 18.8645 | 3.7430 | 3.8806   | -2.0684  | 0.0011    |
| subject_loo   | MLP           | 13.9244 | 3.0553 | 3.3636   | -1.0275  | 0.2614    |
| subject_loo   | LightGBM      | 13.0676 | 3.0373 | 3.2700   | -0.7352  | 0.2721    |
| recording_loo | MeanEstimator | 13.7904 | 3.2620 | 3.2170   | -1.2703  | 0.0715    |
| recording_loo | SVM           | 16.5039 | 3.0595 | 3.4087   | -1.2817  | 0.0521    |
| recording_loo | GaussianNB    | 19.5374 | 3.7891 | 3.7776   | -2.5449  | -0.0273   |
| recording_loo | MLP           | 15.2606 | 3.2298 | 3.4546   | -1.6540  | 0.0484    |
| recording_loo | LightGBM      | 14.8453 | 3.2624 | 3.3680   | -1.5904  | 0.0556    |
| combined_loo  | MeanEstimator | 13.6940 | 3.2437 | 2.5945   | -4.4296  | 0.0825    |
| combined_loo  | SVM           | 17.0358 | 3.0881 | 2.8063   | -3.2391  | 0.0079    |
| combined_loo  | GaussianNB    | 19.7463 | 3.8350 | 3.1807   | -10.1552 | -0.0188   |
| combined_loo  | MLP           | 16.4226 | 3.3535 | 2.9072   | -5.4677  | -0.0330   |
| combined_loo  | LightGBM      | 15.4480 | 3.3348 | 2.8231   | -5.2698  | -0.0121   |

We need to beat MAE of 3 for all CVs, 0.26 pearson r for LOSO and 0.10 for LORO and combined LOO.

- should have normalized feats for SVM, forgot