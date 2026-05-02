# Načrt zaključevanja diplomske naloge

**Naslov:** Grafovska nevronska mreža za prepoznavo čustev iz sledilnika pogleda

**Raziskovalno vprašanje:**  
Ali lahko spatio-temporalna grafovska predstavitev eye-tracking oken izboljša klasifikacijo afektivnih stanj v primerjavi z ne-grafovskimi modeli na enakih vhodnih oknih?

**Stil naloge:**  
Mešanica raziskovalnega članka in inženirskega projekta, z večjo težo raziskovalnemu članku.

**Glavni dataset:**  
MAHNOB-HCI-TAGGING, emotion elicitation del.

**Zgodovinska motivacija:**  
eSEEd_v2 se omeni kot začetni poskus, kjer se je pokazalo, da so podatki/problem manj primerni. Ne sme postati osrednji del diplome.

**Glavne naloge:**

- 3-class arousal
- 3-class valence

Nalogi sta enakovredni. Izbrani sta zaradi primerljivosti z MAHNOB-HCI paperjem.

**Glavni evalvacijski protokoli:**

- subject LOO
- recording LOO

Oba sta enakovredna. `combined_loo` se izpusti.

**Vhodni signali:**

- gaze x
- gaze y
- left pupil size
- right pupil size

Ne uporabljamo EEG, GSR, ECG, respiration, video ali drugih fizioloških signalov.

**Glavne primerjave:**

- Majority / Mean baseline
- SVM
- LightGBM
- MLP
- GCN
- GAT
- finalni SpatioTemporalHeteroGNN
- GazeMAE + MLP classifier

**GazeMAE pipeline:**

```text
raw/windowed eye-tracking data
→ GazeMAE encoder
→ embedding
→ MLP classifier
→ 3-class arousal / 3-class valence