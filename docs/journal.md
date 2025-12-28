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