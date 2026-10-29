"""Synthetic checks, not detector validation on annotated ECG."""
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from shared_ecg import (load_dataset, signal_mv, example_ids,
                        filter_ecg, interval_summary, moving_average,
                        candidate_regions, locate_rpeaks, detect_rpeaks,
                        compare_peak_locations, agreement_stats,
                        STEPS, detect_beats, gap_to_reference,
                        correct_beats, median_rate_bpm, run_detector,
                        check_review, plot_beats, beats_above,
                        train_reference_forest)


class SharedECGTests(unittest.TestCase):
    def test_loader_and_units(self):
        data = {'A00002': {'data': np.array([1000, -1000]), 'label': 'N'}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ECG.pkl'
            path.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                load_dataset(path)
            loaded = load_dataset(path, trusted=True)
            np.testing.assert_equal(signal_mv(loaded['A00002']), [1, -1])
            self.assertEqual(example_ids(loaded)['N'], 'A00002')
            self.assertIsNone(example_ids(loaded)['A'])
            with self.assertRaises(FileNotFoundError):
                load_dataset(Path(folder) / 'missing.pkl', trusted=True)

    def test_invalid_waveform(self):
        for x in ([[1, 2], [3, 4]], [1, np.nan], [1]):
            with self.assertRaises(ValueError):
                signal_mv({'data': x})

    def test_filter(self):
        t = np.arange(6000) / 300
        desired = np.sin(2 * np.pi * 10 * t)
        noise = np.sin(2 * np.pi * 100 * t)
        result = filter_ecg(desired + noise)
        self.assertEqual(result.shape, t.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(np.std((result - desired)[600:-600]), 0.1)
        with self.assertRaises(ValueError):
            filter_ecg(desired, high=150)

    def test_integration_and_threshold_regions(self):
        values = np.arange(100, dtype=float)
        # 0.12 s at 300 Hz rounds to 37 samples, centred, zero-padded.
        expected = np.convolve(values, np.ones(37) / 37, mode='same')
        np.testing.assert_allclose(moving_average(values), expected)
        threshold, starts, stops = candidate_regions([0, 0, 4, 4, 0, 4, 0, 0])
        self.assertEqual(threshold, 0.8)
        np.testing.assert_array_equal(starts, [2, 5])
        np.testing.assert_array_equal(stops, [4, 6])
        _, starts, stops = candidate_regions([4, 0, 4, 0, 0, 0, 4, 4])
        np.testing.assert_array_equal(starts, [0, 2, 6])
        np.testing.assert_array_equal(stops, [1, 3, 8])
        _, starts, stops = candidate_regions(np.zeros(100))
        self.assertEqual(len(starts), 0)
        self.assertEqual(len(stops), 0)
        with self.assertRaises(ValueError):
            moving_average([1, 2], window_s=1)
        with self.assertRaises(ValueError):
            candidate_regions(np.ones(20), threshold_fraction=1)

    def test_candidate_localisation_and_separation(self):
        x = np.zeros(1000)
        x[[100, 130, 500]] = [1, 2, 1]
        candidates, peaks = locate_rpeaks(x, [90, 120, 490], [110, 140, 510])
        np.testing.assert_array_equal(candidates, [100, 130, 500])
        np.testing.assert_array_equal(peaks, [130, 500])
        _, inverted = locate_rpeaks(-x, [90, 120, 490], [110, 140, 510], polarity='negative')
        np.testing.assert_array_equal(peaks, inverted)
        self.assertEqual(len(locate_rpeaks(x, [], [])[1]), 0)
        for starts, stops in [([-1], [2]), ([1], [1001]), ([4], [2]), ([1.2], [3])]:
            with self.assertRaises(ValueError):
                locate_rpeaks(x, starts, stops)

    def test_synthetic_detector_and_visible_stages(self):
        fs = 300
        t = np.arange(3000) / fs
        expected_times = np.arange(1, 10)
        x = sum(np.exp(-0.5 * ((t - beat) / 0.025) ** 2) for beat in expected_times)
        clean = filter_ecg(x)
        peaks = detect_rpeaks(clean)
        np.testing.assert_allclose(peaks / fs, expected_times, atol=2 / fs)
        np.testing.assert_array_equal(detect_rpeaks(-clean, polarity='negative'), peaks)
        self.assertEqual(len(detect_rpeaks(np.zeros(3000))), 0)
        # Same explicit operations used in the notebook, without any beat library.
        difference = np.diff(clean, prepend=clean[0])
        width = int(round(0.12 * fs))
        width += (width % 2 == 0)
        envelope = np.convolve(difference ** 2, np.ones(width) / width, mode='same')
        threshold = 0.2 * np.max(envelope[len(envelope)//4:3*len(envelope)//4])
        edges = np.diff(np.r_[False, envelope > threshold, False].astype(int))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        _, visible_peaks = locate_rpeaks(clean, starts, stops)
        np.testing.assert_array_equal(peaks, visible_peaks)

    def test_stage_switchable_detector(self):
        fs = 300
        t = np.arange(3000) / fs
        x = sum(np.exp(-0.5 * ((t - beat) / 0.025) ** 2) for beat in np.arange(1, 10))
        # Steps up to locate reproduce the full detector on its own filtered input.
        np.testing.assert_array_equal(detect_beats(x, STEPS[:4]), detect_rpeaks(filter_ecg(x)))
        np.testing.assert_array_equal(detect_beats(x), detect_beats(x, STEPS[:4]))
        # No steps: one beat per region of the raw signal above 20% of its middle-half maximum.
        np.testing.assert_allclose(detect_beats(x, ()) / fs, np.arange(1, 10), atol=1 / fs)
        # Signed difference: only the rising edge of each beat crosses the threshold.
        self.assertEqual(len(detect_beats(x, ('differentiate',))), 9)
        self.assertGreater(len(detect_beats(x, STEPS[:3])), 0)
        with self.assertRaises(ValueError):
            detect_beats(x, ('square',))

    def test_run_detector_matches_reference(self):
        # Notebook-style step functions in a dict reproduce detect_beats, step by step.
        fs = 300
        t = np.arange(6000) / fs
        rng = np.random.default_rng(0)
        beats = np.cumsum(rng.uniform(0.6, 1.0, 22))
        x = sum(np.exp(-0.5 * ((t - b) / 0.02) ** 2) for b in beats) + 0.05 * rng.standard_normal(t.size)

        def regions(signal):
            n = len(signal)
            threshold = 0.2 * np.max(signal[n // 4:3 * n // 4])
            edges = np.diff(np.r_[False, signal > threshold, False].astype(int))
            return threshold, np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)

        def pick(signal, starts, stops):
            return np.array([a + np.argmax(signal[a:b]) for a, b in zip(starts, stops)], dtype=int)

        steps = {"regions": regions, "pick": pick, "filter": filter_ecg,
                 "differentiate": lambda v: np.diff(v, prepend=v[0]),
                 "envelope": lambda d: moving_average(d ** 2),
                 "locate": lambda ecg, a, b: locate_rpeaks(ecg, a, b)[1],
                 "correct": correct_beats}
        for k in range(len(STEPS) + 1):
            chosen = {name: steps[name] for name in ("regions", "pick", *STEPS[:k])}
            np.testing.assert_array_equal(run_detector(x, chosen), detect_beats(x, STEPS[:k]))
        without_filter = {n: f for n, f in steps.items() if n != "filter"}
        np.testing.assert_array_equal(run_detector(x, without_filter),
                                      detect_beats(x, STEPS[1:]))
        with self.assertRaises(ValueError):
            run_detector(x, {"regions": regions})
        with self.assertRaises(ValueError):
            run_detector(x, {**steps, "square": pick})

    def test_check_review_and_plot(self):
        np.testing.assert_array_equal(check_review([300, 600, 900], 0.5, 4, 3000), [300, 600, 900])
        for peaks, start, end in [([300, 600], 0, 4), ([300, 300, 900], 0, 4),
                                  ([300, 600, 1500], 0, 4), ([300, 600, 900], 4, 2)]:
            with self.assertRaises(ValueError):
                check_review(peaks, start, end, 3000)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_beats(ax, np.zeros(600), neurokit_reference=[100, 400], ours=[110])
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         ["NeuroKit2 reference: 2 beats", "Ours: 1 beat"])
        plt.close(fig)

    def test_correct_beats(self):
        # Median RR 240 samples (75 bpm): drop the beat at 300, fill the gap 720 -> 1440.
        peaks = [0, 240, 300, 480, 720, 1440, 1680]
        np.testing.assert_array_equal(correct_beats(peaks), [0, 240, 480, 720, 960, 1200, 1440, 1680])
        self.assertEqual(median_rate_bpm(peaks), 75)
        regular = np.arange(0, 3000, 250)
        np.testing.assert_array_equal(correct_beats(regular), regular)
        # Implausible median rate (5 beats in a minute): left unchanged.
        sparse = np.arange(0, 18000, 3600)
        np.testing.assert_array_equal(correct_beats(sparse), sparse)
        self.assertTrue(np.isnan(median_rate_bpm([0, 300])))

    def test_gap_to_reference(self):
        reference = np.array([0, 300, 600, 900])
        same = gap_to_reference(reference, reference)
        self.assertEqual(same['agreement'], 1)
        self.assertEqual(same['hr_bpm'], same['reference_hr_bpm'])
        # One extra beat: 4 matched of 5 + 4 beats.
        extra = gap_to_reference([0, 150, 300, 600, 900], reference)
        self.assertAlmostEqual(extra['agreement'], 8 / 9)
        self.assertEqual(extra['hr_bpm'], 80)
        # RR 0.5, 0.5, 1, 1 s: successive differences 0, 0.5, 0 s.
        self.assertAlmostEqual(extra['rmssd_ms'], 1000 * np.sqrt(0.25 / 3))
        self.assertEqual(extra['reference_rmssd_ms'], 0)
        self.assertTrue(np.isnan(gap_to_reference([0, 300], reference)['rmssd_ms']))
        self.assertTrue(np.isnan(gap_to_reference([10], reference)['hr_bpm']))
        self.assertEqual(gap_to_reference([], [])['agreement'], 1)

    def test_peak_comparison(self):
        result = compare_peak_locations([100, 200, 500], [105, 230, 800], fs=300, tolerance_s=0.1)
        np.testing.assert_array_equal(result['pairs'], [[100, 105], [200, 230]])
        np.testing.assert_array_equal(result['first_only'], [500])
        np.testing.assert_array_equal(result['second_only'], [800])
        np.testing.assert_allclose(result['offset_ms'], [1000 * 5 / 300, 100])
        # One other detection cannot match two manual detections.
        result = compare_peak_locations([100, 120], [110], fs=300)
        np.testing.assert_array_equal(result['pairs'], [[100, 110]])
        np.testing.assert_array_equal(result['first_only'], [120])
        self.assertEqual(compare_peak_locations([], [])['pairs'].shape, (0, 2))
        np.testing.assert_array_equal(compare_peak_locations([], [10])['second_only'], [10])
        result = compare_peak_locations([100], [90], fs=1000)
        np.testing.assert_array_equal(result['offset_ms'], [-10])
        for first in ([1, 1], [-1], [1.5], [np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                compare_peak_locations(first, [])
        with self.assertRaises(ValueError):
            compare_peak_locations([], [], tolerance_s=-1)

    def test_agreement_stats(self):
        a = np.array([800., 850., 900., 950.])
        same = agreement_stats(a, a)
        self.assertAlmostEqual(same['ccc'], 1)
        self.assertAlmostEqual(same['bias'], 0)
        # A constant offset keeps Pearson r at 1 but lowers CCC.
        shifted = agreement_stats(a, a + 50)
        self.assertAlmostEqual(shifted['pearson_r'], 1)
        self.assertAlmostEqual(shifted['bias'], 50)
        self.assertLess(shifted['ccc'], 0.8)
        np.testing.assert_allclose(shifted['loa'], [50, 50])
        result = agreement_stats([1., 2., 3.], [1., 3., 2.])
        np.testing.assert_allclose([result['pearson_r'], result['ccc']], [0.5, 0.5])
        with self.assertRaises(ValueError):
            agreement_stats([1., 2.], [1., 2.])

    def test_intervals(self):
        result = interval_summary([0, 300, 600, 900])
        self.assertEqual(result['rate_from_mean_rr_bpm'], 60)
        self.assertEqual(result['sdrr_ms'], 0)
        self.assertEqual(result['rmssd_ms'], 0)
        for peaks in ([0, 300, 300], [-1, 300, 600], [0, 300.5, 600]):
            with self.assertRaises(ValueError):
                interval_summary(peaks)
        with self.assertRaises(ValueError):
            interval_summary([0, 300, 600], fs=np.nan)



class NaiveDetectorTests(unittest.TestCase):
    def test_beats_above(self):
        signal = np.array([0, 1, 3, 1, 0, 0, 2, 5, 2, 0, 1, 0], dtype=float)
        np.testing.assert_array_equal(beats_above(signal, 1.5), [2, 7])
        np.testing.assert_array_equal(beats_above(signal, 4), [7])
        self.assertEqual(len(beats_above(signal, 10)), 0)


class WaveLabelTests(unittest.TestCase):
    def test_diagrams_are_well_formed(self):
        """Task 1: ten dots per diagram, in time order, inside the window, one hash per dot."""
        import wave_labels

        self.assertEqual(len(wave_labels.DIAGRAMS), 6)
        self.assertEqual(len(wave_labels.EXAMPLE_LABELS), 10)
        for d in wave_labels.DIAGRAMS:
            self.assertEqual(len(d["dots"]), 10)
            self.assertEqual(len(d["hashes"]), 10)
            self.assertEqual(d["dots"], sorted(set(d["dots"])))
            self.assertTrue(d["start"] <= d["dots"][0] and d["dots"][-1] < d["stop"])
        d = wave_labels.DIAGRAMS[0]
        for n, (letter, h) in enumerate(zip(wave_labels.EXAMPLE_LABELS, d["hashes"]), 1):
            self.assertEqual(wave_labels._hash(d["rid"], n, letter), h)
        self.assertNotEqual(wave_labels._hash(d["rid"], 1, "T"), d["hashes"][0])

    def test_reference_forest_is_deterministic(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, 4)); y = (x[:, 0] > 0).astype(int)
        scaler, model = train_reference_forest(x, y, n_estimators=10)
        scaler2, model2 = train_reference_forest(x, y, n_estimators=10)
        np.testing.assert_array_equal(model.predict(scaler.transform(x)), model2.predict(scaler2.transform(x)))
        self.assertEqual(model.predict(scaler.transform(x)).shape, (60,))

    def test_solution_fills_every_placeholder(self):
        import json
        try:
            import build_solution  # Instructor-only; not in the student repository.
        except ImportError:
            self.skipTest("build_solution.py is not present")
        nb = json.loads(build_solution.LAB.read_text())
        total = sum(build_solution.fill("".join(c["source"]))[1] for c in nb["cells"] if c["cell_type"] == "code")
        self.assertEqual(total, len(build_solution.FILLS) + len(build_solution.RETURN_FILLS) + 1)


if __name__ == "__main__":
    unittest.main()
