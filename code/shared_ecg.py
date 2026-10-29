"""Lab 1 helpers for Lab 2's PhysioNet 2017 teaching export.

Pickle can execute code: load only a file whose source you trust.
Amplitude conversion follows Lab 2 (stored values / 1000 -> mV).
"""
from pathlib import Path
import pickle

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 300
LABELS = {"N": "Normal sinus rhythm", "A": "Atrial fibrillation",
          "O": "Other rhythm", "~": "Noise"}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "ECG.pkl"  # From the lab's data zip; see data/README.md.


def load_dataset(path=DEFAULT_DATA, *, trusted=False):
    """Explicit trust opt-in is required before unpickling."""
    if not trusted:
        raise ValueError("Only load trusted pickle files; pass trusted=True after checking the source")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download ECG.pkl using the link in data/README.md; "
            "ECG_feature.pkl is not needed for Lab 1.")
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("Expected a nonempty recording dictionary")
    for key, record in dataset.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ValueError("Expected recording IDs mapped to dictionaries")
        if record.get("label") not in LABELS:
            raise ValueError(f"Unknown label for {key}")
        signal_mv(record)
    return dataset


def signal_mv(record):
    """Return the complete single-lead recording in mV, without cropping."""
    x = np.asarray(record["data"], dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite single-lead waveform")
    return x / 1000.0


def example_ids(dataset):
    """Reproducible starting candidates, NOT curated/quality-verified examples."""
    return {label: next((key for key in sorted(dataset)
                         if dataset[key]["label"] == label), None)
            for label in LABELS}


def filter_ecg(x, fs=FS, low=0.5, high=40.0):
    """Teaching bandpass: fourth-order Butterworth, offline zero-phase SOS.

    This is not the historical 1 kHz filter. Forward/backward filtering is
    noncausal and doubles the effective order; inspect boundary distortion.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite single-lead waveform")
    if not 0 < low < high < fs / 2:
        raise ValueError("Require 0 < low < high < Nyquist frequency")
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def moving_average(values, fs=FS, window_s=0.12):
    """Centred moving mean: combine nearby squared differences, offline.

    Round to an odd sample count so each output has an unambiguous centre.
    Zero padding affects the first/last half-window; inspect these boundaries.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or not np.isfinite(fs) or fs <= 0
            or not np.isfinite(window_s) or window_s <= 0):
        raise ValueError("Need finite 1D values, positive fs and positive window_s")
    width = max(1, int(round(window_s * fs)))
    if width % 2 == 0:
        width += 1
    if width > values.size:
        raise ValueError("Recording is shorter than the integration window")
    return np.convolve(values, np.ones(width) / width, mode="same")


def candidate_regions(envelope, threshold_fraction=0.2):
    """Threshold against the middle-half maximum; return [start, stop) regions.

    This simple global threshold can fail with noise/amplitude changes. No
    beat detector library is used. The notebook displays every intermediate.
    """
    envelope = np.asarray(envelope, dtype=float)
    if (envelope.ndim != 1 or envelope.size < 4
            or not np.isfinite(envelope).all() or np.any(envelope < 0)):
        raise ValueError("Need at least four finite, nonnegative envelope samples")
    if not np.isfinite(threshold_fraction) or not 0 < threshold_fraction < 1:
        raise ValueError("threshold_fraction must be between zero and one")
    n = len(envelope)
    threshold = float(threshold_fraction * np.max(envelope[n // 4:3 * n // 4]))
    above = envelope > threshold
    edges = np.diff(np.r_[False, above, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)  # exclusive, including len(signal) at end
    return threshold, starts, stops


def locate_rpeaks(filtered, starts, stops, fs=FS, min_separation_s=0.25,
                  polarity="positive"):
    """Locate ECG extrema in regions; retain stronger candidates if too close.

    positive: local maxima; negative: local minima. Polarity must be chosen
    after looking at the ECG. No claim that every extremum is a true R peak.
    Returns all region candidates and the retained indices for visual review.
    """
    x = np.asarray(filtered, dtype=float)
    starts, stops = np.asarray(starts), np.asarray(stops)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a finite single-lead waveform")
    if (not np.isfinite(fs) or fs <= 0 or not np.isfinite(min_separation_s)
            or min_separation_s <= 0):
        raise ValueError("fs and min_separation_s must be positive")
    if (starts.ndim != 1 or starts.shape != stops.shape
            or not np.isfinite(starts).all() or not np.isfinite(stops).all()
            or np.any(starts != np.floor(starts)) or np.any(stops != np.floor(stops))
            or np.any(starts < 0) or np.any(stops > x.size) or np.any(starts >= stops)):
        raise ValueError("Expected valid start/stop region pairs")
    if polarity not in {"positive", "negative"}:
        raise ValueError("Choose positive or negative polarity after inspecting the ECG")
    score = x if polarity == "positive" else -x
    starts, stops = starts.astype(int), stops.astype(int)
    candidates = np.array([a + np.argmax(score[a:b]) for a, b in zip(starts, stops)], dtype=int)
    distance = max(1, int(np.ceil(min_separation_s * fs)))
    kept = []
    for p in sorted(set(candidates.tolist()), key=lambda p: (-score[p], p)):
        if all(abs(p - other) >= distance for other in kept):
            kept.append(p)
    return candidates, np.array(sorted(kept), dtype=int)


def beats_above(signal, threshold):
    """The naive detector: one beat per region where signal > threshold, at the region's maximum.

    Reference for the notebook's Task 3 self-check. Returns sample indices.
    """
    signal = np.asarray(signal, dtype=float)
    edges = np.diff(np.r_[False, signal > threshold, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return np.array([a + np.argmax(signal[a:b]) for a, b in zip(starts, stops)], dtype=int)


def detect_rpeaks(filtered, fs=FS, *, window_s=0.12, threshold_fraction=0.2,
                  min_separation_s=0.25, polarity="positive"):
    """Convenience wrapper around the same visible NumPy steps in the notebook.

    Teaching algorithm for the 300 Hz worksheet, NOT the historical detector
    or a validated clinical algorithm. Input should already be filtered.
    """
    x = np.asarray(filtered, dtype=float)
    if x.ndim != 1 or x.size < 4 or not np.isfinite(x).all():
        raise ValueError("Expected at least four finite single-lead samples")
    difference = np.diff(x, prepend=x[0])
    squared = difference ** 2
    envelope = moving_average(squared, fs, window_s)
    _, starts, stops = candidate_regions(envelope, threshold_fraction)
    _, peaks = locate_rpeaks(x, starts, stops, fs, min_separation_s, polarity)
    return peaks


STEPS = ("filter", "differentiate", "envelope", "locate", "correct")


def detect_beats(x, steps=STEPS, fs=FS, *, low=0.5, high=40.0, window_s=0.12,
                 threshold_fraction=0.2, min_separation_s=0.25, polarity="positive"):
    """Section 2's detector with only the listed steps switched on, for scoring.

    With no steps, threshold the raw ECG relative to its middle-half maximum and
    take one beat per region. "envelope" is square + moving average. Without
    "locate", each beat is the maximum of the processed signal in its region and
    no minimum separation is applied. "correct" imputes the beats with
    `correct_beats`. Teaching algorithm, not a validated detector.
    """
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}")
    ecg = filter_ecg(x, fs, low, high) if "filter" in steps else np.asarray(x, dtype=float)
    if ecg.ndim != 1 or ecg.size < 4 or not np.isfinite(ecg).all():
        raise ValueError("Expected at least four finite single-lead samples")
    activity = np.diff(ecg, prepend=ecg[0]) if "differentiate" in steps else ecg
    if "envelope" in steps:
        activity = moving_average(activity ** 2, fs, window_s)
    n = len(activity)
    above = activity > threshold_fraction * np.max(activity[n // 4:3 * n // 4])
    edges = np.diff(np.r_[False, above, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if "locate" in steps:
        beats = locate_rpeaks(ecg, starts, stops, fs, min_separation_s, polarity)[1]
    else:
        beats = np.array([a + np.argmax(activity[a:b]) for a, b in zip(starts, stops)], dtype=int)
    return correct_beats(beats, fs) if "correct" in steps else beats


def median_rate_bpm(peaks, fs=FS):
    """Heart rate from the median RR interval; NaN with fewer than three beats."""
    peaks = np.asarray(peaks)
    return 60 * fs / np.median(np.diff(peaks)) if len(peaks) >= 3 else np.nan


def correct_beats(peaks, fs=FS, short=0.6, long=1.6, rate_bpm=(30, 200)):
    """Median-RR imputation: drop extra beats, fill gaps from missed beats.

    In time order, a beat closer than short x median RR to the last kept beat is
    dropped; a gap longer than long x median RR is split evenly into
    round(gap / median) intervals, inserting invented beats. Unchanged if the
    median rate is outside rate_bpm: then most beats are wrong and the median
    cannot be trusted. Also removes genuine irregularity (AF, ectopic beats).
    """
    peaks = np.asarray(peaks, dtype=int)
    rate = median_rate_bpm(peaks, fs)
    if not rate_bpm[0] <= rate <= rate_bpm[1]:
        return peaks
    median_rr = np.median(np.diff(peaks))
    kept = [int(peaks[0])]
    for p in peaks[1:]:
        gap = p - kept[-1]
        if gap < short * median_rr:
            continue
        if gap > long * median_rr:
            k = int(round(gap / median_rr))
            kept += [int(kept[-1] + gap * j / k) for j in range(1, k)]
        kept.append(int(p))
    return np.array(kept, dtype=int)


def gap_to_reference(beats, reference, fs=FS, tolerance_s=0.1):
    """Score one recording's beats against a reference detector's beats.

    Beat agreement is 2 x matched / (our count + reference count): 1 when every
    beat pairs within the tolerance; missed and extra beats both lower it.
    Heart rate is 60 / mean RR for each detector (NaN with fewer than two beats);
    RMSSD is in ms (NaN with fewer than three beats).
    The reference is another algorithm's output (NeuroKit2), not beat annotations.
    """
    beats, reference = np.asarray(beats, dtype=int), np.asarray(reference, dtype=int)
    total = len(beats) + len(reference)
    matched = len(compare_peak_locations(beats, reference, fs, tolerance_s)["pairs"])

    def rate(p):
        return 60 * fs / np.mean(np.diff(p)) if len(p) >= 2 else np.nan

    def rmssd(p):
        return 1000 * np.sqrt(np.mean(np.diff(np.diff(p) / fs) ** 2)) if len(p) >= 3 else np.nan
    return {"agreement": 2 * matched / total if total else 1.0,
            "hr_bpm": rate(beats), "reference_hr_bpm": rate(reference),
            "rmssd_ms": rmssd(beats), "reference_rmssd_ms": rmssd(reference)}


def compare_peak_locations(first, second, fs=FS, tolerance_s=0.1):
    """Chronological one-to-one matching, not comparison against ground truth.

    Pair the earliest remaining indices if within tolerance; otherwise mark
    the earlier one unmatched. This does not minimise timing differences in
    ambiguous clusters. Positive offsets mean the second detector is later.
    """
    if (not np.isfinite(fs) or fs <= 0 or not np.isfinite(tolerance_s)
            or tolerance_s < 0):
        raise ValueError("Require positive fs and nonnegative finite tolerance")
    arrays = []
    for values in (first, second):
        x = np.asarray(values, dtype=float)
        if (x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0)
                or np.any(x != np.floor(x)) or np.any(np.diff(x) <= 0)):
            raise ValueError("Peak arrays must contain increasing nonnegative integer indices")
        arrays.append(x.astype(int))
    first, second = arrays
    pairs, first_only, second_only = [], [], []
    i = j = 0
    while i < len(first) and j < len(second):
        delta = second[j] - first[i]
        if abs(delta) <= tolerance_s * fs:
            pairs.append((first[i], second[j]))
            i += 1
            j += 1
        elif first[i] < second[j]:
            first_only.append(first[i])
            i += 1
        else:
            second_only.append(second[j])
            j += 1
    first_only.extend(first[i:])
    second_only.extend(second[j:])
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return {"pairs": pairs,
            "first_only": np.asarray(first_only, dtype=int),
            "second_only": np.asarray(second_only, dtype=int),
            "offset_ms": (pairs[:, 1] - pairs[:, 0]) * 1000.0 / fs}


def interval_summary(peaks, fs=FS):
    """Descriptive RR metrics only; does not classify intervals as NN."""
    peaks = np.asarray(peaks, dtype=float)
    if (not np.isfinite(fs) or fs <= 0 or peaks.ndim != 1 or peaks.size < 3
            or not np.isfinite(peaks).all() or np.any(peaks < 0)
            or np.any(peaks != np.floor(peaks)) or not np.all(np.diff(peaks) > 0)):
        raise ValueError("Need at least three nonnegative, increasing integer beat indices and positive fs")
    rr = np.diff(peaks) / fs
    return {"rr_s": rr, "heart_rate_bpm": 60 / rr,
            "rate_from_mean_rr_bpm": 60 / rr.mean(),
            "sdrr_ms": np.std(rr, ddof=1) * 1000,
            "rmssd_ms": np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000}


def agreement_stats(a, b):
    """Agreement of paired measurements: Bland-Altman, Pearson r and Lin's CCC.

    Differences are b - a. Limits of agreement are bias +/- 1.96 SD of the
    differences. Pearson r measures association only; Lin's concordance
    correlation coefficient (CCC) also penalises bias and scale differences.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Need two equal-length 1D finite arrays with at least three pairs")
    diff = b - a
    bias, sd = diff.mean(), diff.std(ddof=1)
    cov = np.mean((a - a.mean()) * (b - b.mean()))
    denom = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    spread = a.std() * b.std()
    return {"n": a.size, "bias": bias, "sd_diff": sd,
            "loa": (bias - 1.96 * sd, bias + 1.96 * sd),
            "pearson_r": cov / spread if spread > 0 else np.nan,
            "ccc": 2 * cov / denom if denom > 0 else np.nan}


def run_detector(x, steps):
    """Chain your own step functions, in the fixed order of `detect_beats`.

    `steps` maps step names to functions; only the steps present are applied:
      "filter"(x) -> ecg, "differentiate"(signal) -> signal, "envelope"(difference) -> envelope,
      "regions"(signal) -> (threshold, starts, stops), required,
      "locate"(ecg, starts, stops) -> beats, or else "pick"(signal, starts, stops) -> beats,
      "correct"(beats) -> beats.
    The regions are found on the last processed signal; "pick" takes beats from that signal,
    "locate" from the (filtered) ECG.
    """
    unknown = set(steps) - {"regions", "pick", *STEPS}
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}")
    if "regions" not in steps or not {"pick", "locate"} & set(steps):
        raise ValueError('Need "regions", and "pick" or "locate"')
    ecg = steps["filter"](np.asarray(x, dtype=float)) if "filter" in steps else np.asarray(x, dtype=float)
    signal = ecg
    for name in ("differentiate", "envelope"):
        if name in steps:
            signal = steps[name](signal)
    _, starts, stops = steps["regions"](signal)
    if "locate" in steps:
        beats = steps["locate"](ecg, starts, stops)
    else:
        beats = steps["pick"](signal, starts, stops)
    if "correct" in steps:
        beats = steps["correct"](np.asarray(beats, dtype=int))
    return np.asarray(beats, dtype=int)


# Marker styles for plot_beats, one per kind of beat.
BEAT_MARKERS = {"neurokit_reference": dict(s=90, facecolors="none", edgecolors="tab:green", lw=1.5),
                "candidates": dict(s=25, color="tab:orange", zorder=2),
                "ours": dict(marker="x", color="tab:red", zorder=3),
                "dropped": dict(s=120, marker="s", facecolors="none", edgecolors="black"),
                "inserted": dict(s=80, marker="^", color="tab:purple", zorder=3)}


def plot_beats(ax, x, fs=FS, xlim=None, title=None, labels=None, **beats):
    """Plot an ECG in mV and mark beats: neurokit_reference=, candidates=, ours=, dropped=, inserted=.

    Each keyword takes sample indices; `labels` renames them in the legend.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x)) / fs
    ax.plot(t, x, lw=0.6, color="tab:blue")
    for kind, indices in beats.items():
        indices = np.asarray(indices, dtype=int)
        name = (labels or {}).get(kind, "NeuroKit2 reference" if kind == "neurokit_reference" else kind.capitalize())
        ax.scatter(t[indices], x[indices], label=f"{name}: {len(indices)} beat{'' if len(indices) == 1 else 's'}",
                   **BEAT_MARKERS[kind])
    ax.set(xlabel="Time (s)", ylabel="ECG (mV)", title=title or "")
    if xlim is not None:
        ax.set_xlim(xlim)
    if beats:
        ax.legend(loc="lower right")
    return ax


def plot_traces(traces, fs=FS, xlim=None):
    """Stack signals that share a time axis: {y-axis label: signal}. Returns the axes."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 2.4 * len(traces)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (label, signal) in zip(axes, traces.items()):
        ax.plot(np.arange(len(signal)) / fs, signal, lw=0.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time (s)")
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    fig.tight_layout()
    return axes


def shade_regions(ax, starts, stops, fs=FS):
    """Shade [start, stop) sample regions, such as the regions above a threshold."""
    for a, b in zip(starts, stops):
        ax.axvspan(a / fs, b / fs, color="orange", alpha=0.2)


def check_review(peaks, start_s, end_s, n_samples, fs=FS):
    """Validate reviewed beats: at least three increasing integer indices inside [start_s, end_s)."""
    if not 0 <= start_s < end_s <= n_samples / fs:
        raise ValueError("Choose a segment within the current recording.")
    checked = np.asarray(peaks, dtype=float)
    if (checked.ndim != 1 or checked.size < 3 or not np.isfinite(checked).all()
            or np.any(checked != np.floor(checked)) or np.any(checked < 0)
            or np.any(checked >= n_samples) or np.any(np.diff(checked) <= 0)):
        raise ValueError("Supply at least three increasing, valid integer sample indices.")
    if np.any(checked / fs < start_s) or np.any(checked / fs >= end_s):
        raise ValueError("All reviewed peaks must lie within the selected [start, end) segment.")
    return checked.astype(int)


def list_feature_names(stored_signal, fs=FS):
    """Names of the 54 features, in the order `extract_feature` returns them."""
    import neurokit2 as nk
    from ECG_feature_utils import from_Rpeaks, from_waves
    sig = nk.ecg_clean(stored_signal, sampling_rate=fs)
    _, rp = nk.ecg_peaks(sig, sampling_rate=fs)
    _, waves = nk.ecg_delineate(sig, rp, sampling_rate=fs, method="peak")
    return (list(from_Rpeaks(sig, rp["ECG_R_Peaks"], sampling_rate=fs, average=True))
            + list(from_waves(sig, rp["ECG_R_Peaks"], waves, sampling_rate=fs, average=True))
            + list(nk.hrv_time(rp["ECG_R_Peaks"], sampling_rate=fs).columns) + ["SQI"])


def load_feature_table(path, dataset, *, trusted=False):
    """Lab 2's precomputed feature table: (features, labels), or (None, None) with a message."""
    path = Path(path)
    if not path.exists():
        print(f"Place ECG_feature.pkl next to ECG.pkl ({path}) to run 5B and 5C.")
        return None, None
    if not trusted:
        print("Feature table not loaded: enable the trust flag only for a trusted source.")
        return None, None
    with path.open("rb") as stream:
        table = pickle.load(stream)
    features, labels = np.asarray(table["features"]), np.asarray(table["labels"])
    if list(labels) != [r["label"] for r in dataset.values()]:
        raise ValueError("Feature rows do not follow the order of ECG.pkl")
    return features, labels


def train_reference_forest(x_train, y_train, n_estimators=100, random_state=42):
    """Reference for the notebook's Task 5 self-check: standardise with the training set only,
    then fit a RandomForest. Returns (scaler, model). Needs scikit-learn."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(scaler.transform(x_train), y_train)
    return scaler, model
