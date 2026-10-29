# Lab 1 — ECG signals to interpretable features and a classifier

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junjiechen-chris/CDT-HDS-ECG-Lab-1/blob/main/ECG_Preprocessing.ipynb)

Open **`ECG_Preprocessing.ipynb`**, the lab notebook, and read **`note/Lab1_Lab_Note.pdf`**, the lab
note, alongside it. The notebook follows the note's five questions, and both number their
sections 1 to 5:

1. **What is an ECG signal?** Label the P, Q, R, S and T dots on five excerpts from N recordings,
   with one labelled example.
2. **What is measured from it?** Code heart rate, SDRR and RMSSD from the RR sequences that
   NeuroKit2 extracts from three recordings; checked against `hrv_time`.
3. **Can we extract them ourselves?** The recipe is a peak finder, median imputation and the
   arithmetic of Section 2. Code detector v1: a threshold, one beat per region above it.
4. **Finding the beats.** Code the four steps of detector v2 (4A filter; 4B differentiate, square
   and integrate; 4C locate; 4D median imputation). Each is scored on a benchmark across ~2,000
   recordings against a fixed **NeuroKit2 reference**: NeuroKit2 as it is (`ecg_clean` +
   `ecg_peaks`, no correction, and `hrv_time`). Scores: beat agreement, and the MAE and P95
   absolute error of heart rate and RMSSD. 4E compares the final detector with the reference
   (Bland–Altman, r, CCC); a supplied function removes one step at a time.
5. **What are the features for?** NeuroKit2 extracts 54 features per recording (a precomputed
   table). Code the classifier: a scaler fitted on the training set and a RandomForest of
   100 trees, on Lab 2's split; keep the test macro F1 for Lab 2.

Every task is a function written from its specification (what is given, what may be used, what
to return, how it is checked). Each is checked on an invented input and against a reference
implementation in `code/shared_ecg.py`, then plotted or scored by supplied code. Unfinished
tasks print a reminder; in Section 4 a blank step falls back to the reference so that later
cells still run, but it is not scored.

## Run on Colab

1. Click the badge above (or open the notebook from this repository in Colab).
2. **File → Save a copy in Drive**, so that your work is kept.
3. Run the Setup cell. It clones this repository next to the notebook, installs the
   requirements, downloads the data zip and asks for its **password, which is given in the lab**.

A CPU runtime is enough. The whole notebook runs in about ten minutes; each benchmark score
takes under a minute.

## Run locally

Python 3.12 has been tested.

```bash
git clone https://github.com/junjiechen-chris/CDT-HDS-ECG-Lab-1.git
cd CDT-HDS-ECG-Lab-1
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt -r requirements-comparison.txt
python -m unittest discover -s tests -v
jupyter lab ECG_Preprocessing.ipynb
# On the solution branch: rebuild and execute the solution after editing the lab notebook
python solution/build_solution.py --execute
```

Download `ECG_lab_data.zip` into `data/` (the Setup cell does this too) and run the Setup cell,
which asks for the password and unpacks `ECG.pkl` and `ECG_feature.pkl` there. See
`data/README.md`. No GPU is needed.

## Files

| Path | Purpose |
|---|---|
| `ECG_Preprocessing.ipynb` | The lab notebook, executed blank: demonstrations shown, tasks printing their reminders. |
| `solution/ECG_Preprocessing_Solution.ipynb` | On this `solution` branch only: every placeholder filled and executed. Generated from the lab notebook by `solution/build_solution.py`; do not edit by hand. |
| `note/Lab1_Lab_Note.pdf` | The lab note: the ideas behind each section, the figures, and the specification of every task function. |
| `code/shared_ecg.py` | Dataset loading; `run_detector`, which chains the notebook's step functions; the reference implementations used by the self-checks; scoring, peak matching and agreement statistics; plotting and feature-table helpers. |
| `code/benchmark.py` | NeuroKit2's beats and `hrv_time` values (the reference), and `Benchmark`: scores a detector on every recording, keeps the leaderboard, compares heart rate and RMSSD (4E) and runs the take-one-out test. |
| `code/wave_labels.py` | Task 1: six N excerpts with stored wave-point positions, the labelled example, and a check against salted hashes of the answers. |
| `code/ECG_feature_utils.py` | Lab 2's 54-feature extractor, copied unchanged (MIT). |
| `tests/test_shared_ecg.py` | Unit tests for the helpers (synthetic data; no detector validation). |
| `requirements.txt`, `requirements-comparison.txt` | Dependencies; the second adds scikit-learn (Section 5). |
| `data/README.md` | What the data zip contains and where it comes from. |
| `LICENSE`, `LICENSES/` | GPL-2.0 for the lab; the MIT notice of `ECG_feature_utils.py`. |

## Processing notes

- Full single-lead recordings, 300 Hz; stored values / 1000 = mV, following Lab 2.
- Teaching detector: 0.5–40 Hz Butterworth (designed for 300 Hz), successive differences,
  square, 120 ms centred moving mean (37 samples), 20% middle-half-maximum threshold, ECG
  maximum per region with a 250 ms minimum separation, then median-RR imputation (drop below
  0.6 × median, fill above 1.6 × median, unchanged outside 30–200 bpm). The reference is never
  corrected.
- Leaderboard (beat agreement; HR MAE / P95 in bpm; RMSSD MAE / P95 in ms), 1,996 recordings:
  naive 0.71, 51.7 / 129, 319 / 584; + filter 0.72, 52.6 / 147, 288 / 516; + envelope 0.92,
  6.4 / 33, 218 / 648; + locate 0.91, 5.1 / 32, 194 / 604; + correct 0.90, 5.0 / 28.5, 146 / 393.
- Neither detector supplies ground truth: agreement is not accuracy. RR intervals from a
  detector are not normal-to-normal intervals, and short-record SDRR/RMSSD are descriptive.
- Classifier (Section 5): macro F1 0.68 on the validation set and 0.64 on the test set, the
  number Lab 2 compares against.

## Provenance and licence

The learning sequence derives from the CWM Wearable Technology practical by Mauricio Villarroel,
with detector material credited to Gari Clifford and Mauricio Villarroel
(https://github.com/maurovm/cwm_laboratory_code). The 300 Hz implementation is a teaching
adaptation, not a numerical reproduction of that detector.

The lab code and notebook are licensed under **GPL-2.0-only** (`LICENSE`), except
`code/ECG_feature_utils.py`, which keeps Lab 2's MIT licence (`LICENSES/`). The data retain the
PhysioNet/CinC 2017 terms; the lab note's template is not distributed.
