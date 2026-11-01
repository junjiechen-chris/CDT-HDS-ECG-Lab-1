# Practical 1 (ECG Lab 1) — ECG signals to features and a classifier

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junjiechen-chris/CDT-HDS-ECG-Lab-1/blob/main-v2/ECG_Preprocessing.ipynb)

Click the badge to open the lab notebook in Colab, then **File → Save a copy in Drive** so that
your work is kept. Run the Setup cell: it fetches the code, installs the packages, downloads the
data and asks for the data password, which is given in the lab. A CPU runtime is enough. Read
the lab note, handed out in the lab, alongside the notebook.

| Path | What it is |
|---|---|
| `ECG_Preprocessing.ipynb` | The lab notebook: Tasks 1–5, with the tasks left for you to code. |
| `solution/ECG_Preprocessing_Solution.ipynb` | The same notebook with every task filled in and executed ([open in Colab](https://colab.research.google.com/github/junjiechen-chris/CDT-HDS-ECG-Lab-1/blob/main-v2/solution/ECG_Preprocessing_Solution.ipynb)). |
| `code/` | The helpers the notebook imports: data loading, the NeuroKit2 reference and benchmark, the Task 1 diagrams, the 54-feature extractor. |

To run locally: clone this branch, `pip install -r requirements.txt -r requirements-comparison.txt`,
open the notebook and run the Setup cell, which downloads and unpacks the data into `data/`.

The data come from the PhysioNet/CinC Challenge 2017 and are for educational use only. The code
is GPL-2.0 (`LICENSE`), except `code/ECG_feature_utils.py` (MIT, see `LICENSES/`).
