# Data

The lab uses two pickle files, distributed together as a **password-protected zip**
(`ECG_lab_data.zip`) on Hugging Face. The password is given in the lab. The notebook's
Setup cell downloads the zip, asks for the password and unpacks it here; locally you can
also download it by hand into this folder.

- `ECG.pkl`: 2,000 single-lead ECG recordings from the PhysioNet/CinC Challenge 2017
  (https://physionet.org/content/challenge-2017/1.0.0/), 300 Hz; stored values / 1000 = mV.
  A dictionary: recording ID → `data` (1-D waveform), `label`.
  Labels: N = normal sinus rhythm; A = atrial fibrillation; O = other rhythm; ~ = noise.
  They describe recordings, not individual beats; no beat annotations are supplied.
  Counts: N 1187, A 196, O 553, ~ 64.
- `ECG_feature.pkl`: the 54 NeuroKit2 features of every recording, in the same order
  (Section 5).

**Only unpickle trusted files:** pickle can execute code. The loader requires an explicit
trust flag, which the notebook enables for the lab's own zip.

Respect the dataset's attribution and reuse terms; the code licence does not license the data.
Nothing in this folder except this file is committed.
