"""Lab 1 benchmark: score a beat detector against NeuroKit2's result, as it is.

The NeuroKit2 reference is another algorithm's output, not beat annotations.
"""
import numpy as np
import matplotlib.pyplot as plt

from shared_ecg import (FS, LABELS, signal_mv, run_detector, gap_to_reference, median_rate_bpm,
                        interval_summary, agreement_stats, plot_beats)


def _neurokit():
    try:
        import neurokit2 as nk
    except ImportError as error:
        raise ImportError("Sections 2, 3 and 5 need NeuroKit2: install requirements.txt "
                          "(Colab: %pip install neurokit2), then rerun.") from error
    return nk


def neurokit_beats(x, fs=FS):
    """NeuroKit2's beats as it is: ecg_clean, then ecg_peaks without artefact correction."""
    nk = _neurokit()
    _, info = nk.ecg_peaks(nk.ecg_clean(x, sampling_rate=fs), sampling_rate=fs, correct_artifacts=False)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def neurokit_hrv(beats, fs=FS):
    """NeuroKit2's hrv_time: heart rate (60000 / HRV_MeanNN) in bpm, HRV_SDNN and HRV_RMSSD in ms."""
    hrv = _neurokit().hrv_time(beats, sampling_rate=fs)
    return {"hr_bpm": 60000 / hrv["HRV_MeanNN"].iloc[0],
            "sdnn_ms": hrv["HRV_SDNN"].iloc[0], "rmssd_ms": hrv["HRV_RMSSD"].iloc[0]}


def mae_p95(errors):
    """Mean and 95th percentile of the absolute errors, skipping recordings without a value."""
    errors = np.abs(np.asarray(errors, dtype=float))
    errors = errors[np.isfinite(errors)]
    return errors.mean(), np.percentile(errors, 95)


class Benchmark:
    """Score detectors on every recording against NeuroKit2's beats, heart rate and RMSSD.

    Scores: beat agreement (2 x matched / (ours + reference), pairing within tolerance_s), and
    the MAE and P95 of the absolute heart rate and RMSSD errors. Recordings where the reference
    has fewer than three beats are not scored; where ours has too few, the error is skipped.
    """

    def __init__(self, dataset, show, fs=FS, tolerance_s=0.1):
        self.data, self.show, self.fs, self.tolerance_s = dataset, tuple(show), fs, tolerance_s
        self.neurokit_reference = {rid: neurokit_beats(signal_mv(record), fs) for rid, record in dataset.items()}
        self.scored_ids = [rid for rid in dataset if len(self.neurokit_reference[rid]) >= 3]
        self.neurokit_reference_hrv = {rid: neurokit_hrv(self.neurokit_reference[rid], fs) for rid in self.scored_ids}
        self.scoreboard = {}  # Leaderboard: name -> (agreement, HR MAE, HR P95, RMSSD MAE, RMSSD P95).
        print(f"NeuroKit2 reference beats for {len(self.neurokit_reference)} recordings; {len(self.scored_ids)} with at least "
              f"three are scored.")

    def score(self, name, steps, show=None, zoom_s=(0, 10), plot=True, record=True):
        """Run `run_detector(x, steps)` on every recording and print its scores.

        record: enter it on the leaderboard. plot: show the recordings and the leaderboard.
        """
        rows, labels = [], []
        for rid in self.scored_ids:
            gap = gap_to_reference(run_detector(signal_mv(self.data[rid]), steps), self.neurokit_reference[rid],
                                self.fs, self.tolerance_s)
            rows.append((gap["agreement"], gap["hr_bpm"] - gap["reference_hr_bpm"],  # NaN: too few beats
                         gap["rmssd_ms"] - gap["reference_rmssd_ms"]))
            labels.append(self.data[rid]["label"])
        rows, labels = np.array(rows, dtype=float), np.array(labels)
        result = (rows[:, 0].mean(), *mae_p95(rows[:, 1]), *mae_p95(rows[:, 2]))
        print(f"{name}: beat agreement {result[0]:.2f} | heart rate error: MAE {result[1]:.1f}, "
              f"P95 {result[2]:.1f} bpm | RMSSD error: MAE {result[3]:.0f}, P95 {result[4]:.0f} ms")
        if record:
            self.scoreboard[name] = result
        if plot:
            print(f"  {np.isnan(rows[:, 2]).sum()} recordings with fewer than three of our beats have no RMSSD error.")
            print("  by label (agreement / HR MAE / RMSSD MAE): " + " | ".join(
                f"{lab} {rows[labels == lab, 0].mean():.2f} / {mae_p95(rows[labels == lab, 1])[0]:.1f} / "
                f"{mae_p95(rows[labels == lab, 2])[0]:.0f}" for lab in LABELS))
            self._plot(name, steps, self.show if show is None else show, zoom_s)
        return result

    def _plot(self, name, steps, show, zoom_s):
        fig = plt.figure(figsize=(12, 3 * len(show) + 3.5))
        grid = fig.add_gridspec(len(show) + 1, 3, height_ratios=[1] * len(show) + [1.2])
        for row, rid in enumerate(show):
            x = signal_mv(self.data[rid])
            plot_beats(fig.add_subplot(grid[row, :]), x, self.fs, xlim=zoom_s,
                       title=f"{rid} ({self.data[rid]['label']}), {name}",
                       neurokit_reference=self.neurokit_reference[rid], ours=run_detector(x, steps))
        names = list(self.scoreboard)
        xs = np.arange(len(names))
        panels = [("Beat agreement (reference: 1)", [0], 1.0, "%.2f"),
                  ("Heart rate error (bpm, reference: 0)", [1, 2], 0.0, "%.0f"),
                  ("RMSSD error (ms, reference: 0)", [3, 4], 0.0, "%.0f")]
        for col, (metric, cols, goal_value, fmt) in enumerate(panels):
            ax = fig.add_subplot(grid[-1, col])
            width = 0.6 / len(cols)
            for j, (c, kind) in enumerate(zip(cols, ["MAE", "P95"] if len(cols) == 2 else [None])):
                offset = (j - (len(cols) - 1) / 2) * width
                colors = ["tab:red" if n == name else "tab:grey" for n in names]
                bars = ax.bar(xs + offset, [self.scoreboard[n][c] for n in names], width=width,
                              color=colors, alpha=1.0 if j == 0 else 0.45, label=kind)
                ax.bar_label(bars, fmt=fmt, fontsize=7)
            ax.axhline(goal_value, color="tab:green", ls="--", lw=1)
            ax.set_xticks(xs, names)
            ax.set(xlim=(-0.5, max(len(names), 6) - 0.5), title=metric)
            ax.set_ylim(0, 1.1 if goal_value == 1 else None)
            if len(cols) == 2:
                ax.legend(fontsize=8)  # Dark: MAE; light: P95.
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", labelsize=8, rotation=35)
        plt.tight_layout()
        plt.show()

    def take_one_out(self, steps, groups=None):
        """Score the full detector and the detector with one step removed, and plot the gaps.

        Adding steps one at a time shows what each adds on top of the ones before it; this is the
        reverse test: is each step still needed once all the others are there? Nothing is entered
        on the leaderboard. Returns {name: scores}.
        """
        if groups is None:
            groups = {"filter": {"filter"}, "diff, square & integrate": {"differentiate", "envelope"},
                      "locate": {"locate"}, "correct": {"correct"}}
        results = {"All steps": self.score("All steps", steps, plot=False, record=False)}
        for name, skip in groups.items():
            if skip & set(steps):
                results[f"Without {name}"] = self.score(
                    f"Without {name}", {k: f for k, f in steps.items() if k not in skip},
                    plot=False, record=False)
        names = list(results)
        xs = np.arange(len(names))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        panels = [("Beat agreement (reference: 1)", [0], 1.0, "%.2f"),
                  ("Heart rate error (bpm, reference: 0)", [1, 2], 0.0, "%.0f"),
                  ("RMSSD error (ms, reference: 0)", [3, 4], 0.0, "%.0f")]
        for ax, (metric, cols, goal_value, fmt) in zip(axes, panels):
            width = 0.6 / len(cols)
            for j, (c, kind) in enumerate(zip(cols, ["MAE", "P95"] if len(cols) == 2 else [None])):
                offset = (j - (len(cols) - 1) / 2) * width
                colors = ["tab:red" if n == "All steps" else "tab:grey" for n in names]
                bars = ax.bar(xs + offset, [results[n][c] for n in names], width=width,
                              color=colors, alpha=1.0 if j == 0 else 0.45, label=kind)
                ax.bar_label(bars, fmt=fmt, fontsize=7)
            ax.axhline(goal_value, color="tab:green", ls="--", lw=1)
            ax.set_xticks(xs, names)
            ax.set(title=metric)
            ax.set_ylim(0, 1.1 if goal_value == 1 else None)
            if len(cols) == 2:
                ax.legend(fontsize=8)
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", labelsize=8, rotation=35)
        fig.suptitle("Take one out: the full detector (red) and the detector with one step removed", fontsize=10)
        plt.tight_layout()
        plt.show()
        return results

    def show_neurokit_reference(self):
        """The NeuroKit2 reference on the recordings shown at every step, and its values for every recording."""
        fig, axes = plt.subplots(len(self.show), 1, figsize=(12, 3 * len(self.show)), squeeze=False)
        for ax, rid in zip(axes[:, 0], self.show):
            hrv = self.neurokit_reference_hrv[rid]
            plot_beats(ax, signal_mv(self.data[rid]), self.fs, xlim=(0, 10), neurokit_reference=self.neurokit_reference[rid],
                       title=f"{rid} ({self.data[rid]['label']}): reference heart rate {hrv['hr_bpm']:.0f} bpm, "
                             f"RMSSD {hrv['rmssd_ms']:.0f} ms")
        plt.tight_layout()
        plt.show()
        values = np.array([(self.neurokit_reference_hrv[rid]["hr_bpm"], self.neurokit_reference_hrv[rid]["rmssd_ms"])
                           for rid in self.scored_ids])
        labels = np.array([self.data[rid]["label"] for rid in self.scored_ids])
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax, col, name, bins in [(axes[0], 0, "Reference heart rate (bpm)", np.linspace(20, 200, 46)),
                                    (axes[1], 1, "Reference RMSSD (ms)", np.geomspace(1, 2000, 46))]:
            for lab in LABELS:
                ax.hist(values[labels == lab, col], bins=bins, histtype="step", lw=1.5, label=lab)
            ax.set(xlabel=name, ylabel="Recordings")
            ax.legend()
        axes[1].set_xscale("log")
        plt.tight_layout()
        plt.show()
        print("Median reference values by label (heart rate bpm, RMSSD ms): " + " | ".join(
            f"{lab} {np.median(values[labels == lab, 0]):.0f}, {np.median(values[labels == lab, 1]):.0f}"
            for lab in LABELS))

    def compare(self, steps):
        """Per-recording heart rate and RMSSD, ours vs the NeuroKit2 reference: statistics and plots.

        Recordings where either median rate is outside 30-200 bpm are detection failures.
        Returns the labels and the columns (our HR, reference HR, our RMSSD, reference RMSSD).
        """
        rows, failed = [], 0
        for rid in self.scored_ids:
            ours_beats, their_beats = run_detector(signal_mv(self.data[rid]), steps), self.neurokit_reference[rid]
            if not all(30 <= median_rate_bpm(b, self.fs) <= 200 for b in (ours_beats, their_beats)):
                failed += 1  # NaN with fewer than three beats also counts as a failure.
                continue
            ours, theirs = interval_summary(ours_beats, self.fs), interval_summary(their_beats, self.fs)
            rows.append((self.data[rid]["label"], ours["rate_from_mean_rr_bpm"],
                         theirs["rate_from_mean_rr_bpm"], ours["rmssd_ms"], theirs["rmssd_ms"]))
        labels = np.array([r[0] for r in rows])
        values = np.array([r[1:] for r in rows])
        print(f"Recordings compared: {len(rows)}; detection failures (median rate outside 30–200 bpm): {failed}")
        measures = [("Heart rate", "bpm", 0, 1), ("RMSSD", "ms", 2, 3)]
        for name, unit, a, b in measures:
            print(f"\n{name} ({unit}), reference − ours:")
            for label in ["all", *LABELS]:
                keep = np.ones(len(labels), bool) if label == "all" else labels == label
                s = agreement_stats(values[keep, a], values[keep, b])
                diff = np.abs(values[keep, b] - values[keep, a])
                print(f"  {label:>3}: n={s['n']:4d} | bias {s['bias']:7.1f} | limits {s['loa'][0]:8.1f} to "
                      f"{s['loa'][1]:7.1f} | r={s['pearson_r']:.2f} CCC={s['ccc']:.2f} | "
                      f"median |diff| {np.median(diff):5.1f} | MAE {diff.mean():5.1f} | P95 {np.percentile(diff, 95):6.1f}")
        colors = dict(zip(LABELS, ["tab:blue", "tab:red", "tab:orange", "tab:grey"]))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for row, (name, unit, a, b) in zip(axes, measures):
            s = agreement_stats(values[:, a], values[:, b])
            for label, color in colors.items():
                keep = labels == label
                row[0].scatter(values[keep, a], values[keep, b], s=6, alpha=0.5, color=color, label=label)
                row[1].scatter((values[keep, a] + values[keep, b]) / 2, values[keep, b] - values[keep, a],
                               s=6, alpha=0.5, color=color, label=label)
            lo, hi = values[:, [a, b]].min(), values[:, [a, b]].max()
            row[0].plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="Identity")
            row[0].set(xlabel=f"Our {name} ({unit})", ylabel=f"Reference {name} ({unit})",
                       title=f"{name}: r = {s['pearson_r']:.2f}, CCC = {s['ccc']:.2f}")
            row[1].axhline(s["bias"], color="black", label=f"Bias {s['bias']:.1f} {unit}")
            for limit in s["loa"]:
                row[1].axhline(limit, color="tab:red", ls="--")
            row[1].plot([], [], color="tab:red", ls="--", label="95% limits of agreement")
            row[1].set(xlabel=f"Mean of the two ({unit})", ylabel=f"Reference − ours ({unit})",
                       title=f"{name}: Bland–Altman, {len(rows)} recordings")
            if name == "RMSSD":  # RMSSD spans several orders of magnitude.
                row[0].set(xscale="log", yscale="log")
                row[1].set_xscale("log")
                row[1].set_yscale("symlog", linthresh=10)
            for ax in row:
                ax.legend(loc="best", fontsize=8, markerscale=2)
        plt.tight_layout()
        plt.show()
        return labels, values
