"""Conversions, Tauc fits, smoothing, normalisation, peaks and the luminous / solar averages."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import uvvis_processing as P


class ConversionTests(unittest.TestCase):
    def test_transmittance_absorbance_round_trip(self):
        t = np.array([10.0, 50.0, 100.0])
        a = P.convert_y(t, "T", True, "A")
        np.testing.assert_allclose(a, [1.0, math.log10(2), 0.0], atol=1e-12)
        np.testing.assert_allclose(P.convert_y(a, "A", False, "T", True), t)
        np.testing.assert_allclose(P.convert_y(np.array([0.5]), "T", False, "T", True), [50.0])
        self.assertTrue(np.isfinite(P.convert_y(np.array([0.0, -1.0]), "T", True, "A")).all(), "T ≤ 0 gives a large finite absorbance")

    def test_reflectance_kubelka_munk_round_trip(self):
        r = np.array([5.0, 50.0, 100.0])
        f = P.kubelka_munk(r)
        np.testing.assert_allclose(f, [(0.95 ** 2) / 0.1, 0.25, 0.0])
        np.testing.assert_allclose(P.convert_y(f, "KM", False, "R", True), r, atol=1e-9)

    def test_transmission_and_reflection_do_not_mix(self):
        with self.assertRaises(ValueError):
            P.convert_y([50.0], "T", True, "R")
        with self.assertRaises(ValueError):
            P.convert_y([50.0], "R", True, "A")
        with self.assertRaises(ValueError):
            P.convert_y([50.0], "X", True, "A")

    def test_units_and_ranges(self):
        np.testing.assert_allclose(P.x_from_nm([400.0, 800.0], "eV"), [3.0996, 1.5498], atol=1e-4)
        np.testing.assert_allclose(P.x_from_nm([400.0], "cm-1"), [25000.0])
        np.testing.assert_allclose(P.x_from_nm([400.0], "um"), [0.4])
        np.testing.assert_allclose(P.x_to_nm(P.x_from_nm([400.0, 2500.0], "eV"), "eV"), [400.0, 2500.0])
        self.assertEqual(P.range_in_unit(300.0, 800.0, "eV"), (P.EV_NM / 800.0, P.EV_NM / 300.0))
        self.assertEqual(P.range_in_unit(300.0, 800.0, "nm"), (300.0, 800.0))
        with self.assertRaises(ValueError):
            P.x_from_nm([1.0], "Å")


class TaucTests(unittest.TestCase):
    def direct_gap_absorbance(self, eg=3.0):
        energy = np.linspace(2.0, 4.5, 251)
        alpha = np.where(energy > eg, np.sqrt(np.clip(energy - eg, 0, None)) / energy, 0.0)   # (αhν)² linear in hν above E_g
        return P.EV_NM / energy, 0.4 * alpha

    def test_direct_gap_is_recovered_by_extrapolation(self):
        x_nm, absorbance = self.direct_gap_absorbance(3.0)
        energy, value = P.tauc_transform(x_nm, absorbance, "A", False, "direct")
        self.assertTrue(np.all(np.diff(energy) > 0), "ordered by increasing energy")
        fit = P.tauc_fit(energy, value, 3.2, 4.2)
        self.assertAlmostEqual(fit["eg"], 3.0, places=3)
        self.assertGreater(fit["r2"], 0.9999)
        self.assertEqual(fit["n"], int(((energy >= 3.2) & (energy <= 4.2)).sum()))

    def test_fit_from_transmittance_and_reflectance_data_gives_the_same_gap(self):
        x_nm, absorbance = self.direct_gap_absorbance(3.0)
        transmittance = 100 * 10 ** (-absorbance)
        energy_t, value_t = P.tauc_transform(x_nm, transmittance, "T", True, "direct")
        self.assertAlmostEqual(P.tauc_fit(energy_t, value_t, 3.2, 4.2)["eg"], 3.0, places=3)
        reflectance = P.convert_y(absorbance, "KM", False, "R", True)              # F(R) ∝ α
        energy_r, value_r = P.tauc_transform(x_nm, reflectance, "R", True, "direct")
        self.assertAlmostEqual(P.tauc_fit(energy_r, value_r, 3.2, 4.2)["eg"], 3.0, places=3)

    def test_exponents_and_failed_fits(self):
        x_nm, absorbance = self.direct_gap_absorbance(3.0)
        energy, direct = P.tauc_transform(x_nm, absorbance, "A", False, "direct")
        _, indirect = P.tauc_transform(x_nm, absorbance, "A", False, "indirect")
        np.testing.assert_allclose(indirect, np.sqrt(np.sqrt(direct)), atol=1e-12)
        self.assertIsNone(P.tauc_fit(energy, direct, 2.0, 2.5), "flat region: no positive slope")
        self.assertIsNone(P.tauc_fit(energy, direct, 4.4, 4.41), "too few points")
        with self.assertRaises(ValueError):
            P.tauc_fit(energy, direct, 4.0, 3.0)
        with self.assertRaises(ValueError):
            P.tauc_transform(x_nm, absorbance, "A", False, "quartic")


class ProcessingTests(unittest.TestCase):
    def test_smoothing_keeps_a_polynomial_and_handles_short_inputs(self):
        x = np.linspace(0, 1, 101)
        y = 2 + 3 * x - x ** 2
        np.testing.assert_allclose(P.smooth(y, 11, 3), y, atol=1e-10)
        short = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(P.smooth(short, 11, 3), short)
        np.testing.assert_allclose(P.smooth(y, 10, 3), y, atol=1e-10)                 # even windows are made odd

    def test_normalisation_modes(self):
        x = np.arange(400.0, 801.0, 100.0)
        y = np.array([2.0, 4.0, 8.0, 4.0, 2.0])
        np.testing.assert_allclose(P.normalize(x, y, "max"), y / 8)
        np.testing.assert_allclose(P.normalize(x, y, "window", window=(400, 500)), y / 4)
        np.testing.assert_allclose(P.normalize(x, y, "minmax"), (y - 2) / 6)
        np.testing.assert_allclose(P.normalize(x, y, "at", at=550.0), y / 6.0)
        self.assertAlmostEqual(float(np.trapz(P.normalize(x, y, "area"), x)) if hasattr(np, "trapz") else float(np.trapezoid(P.normalize(x, y, "area"), x)), 1.0)
        np.testing.assert_array_equal(P.normalize(x, y, "none"), y)
        np.testing.assert_allclose(P.normalize(x, y, "at", at=10.0), y, err_msg="outside the data: unchanged")
        with self.assertRaises(ValueError):
            P.normalize(x, y, "rms")

    def test_peaks_maxima_and_minima(self):
        x = np.arange(10.0)
        y = np.array([0, 1, 0, 0, 3, 0, 0, 2, 0, 0.0])
        self.assertEqual(P.detect_peaks(x, y, 0.1, 2.0).tolist(), [1, 4, 7])
        self.assertEqual(P.detect_peaks(x, y, 0.5, 2.0).tolist(), [4, 7])
        self.assertEqual(P.detect_peaks(x, -y, 0.1, 2.0, kind="min").tolist(), [1, 4, 7])
        self.assertEqual(P.detect_peaks(x, y, 0.1, 5.0).tolist(), [4], "separation keeps the most prominent")
        self.assertEqual(P.detect_peaks(x, np.zeros(10), 0.1, 1.0).tolist(), [])


class WeightingTests(unittest.TestCase):
    def test_tables(self):
        wl, ybar = P.photopic_table()
        self.assertEqual((wl[0], wl[-1], len(wl)), (360.0, 830.0, 471))
        self.assertEqual(float(ybar[np.argmax(ybar)]), 1.0)
        self.assertEqual(float(wl[np.argmax(ybar)]), 555.0)
        wl, irradiance = P.am15g_table()
        self.assertEqual((wl[0], wl[-1], len(wl)), (280.0, 4000.0, 2002))
        trapz = getattr(np, "trapezoid", None) or np.trapz
        self.assertAlmostEqual(float(trapz(irradiance, wl)), 1000.37, delta=0.5, msg="the global tilt spectrum integrates to about 1000 W/m²")

    def test_flat_step_and_partial_spectra(self):
        x = np.arange(200.0, 2600.0, 1.0)
        flat = P.optical_indices(x, np.full_like(x, 50.0), "T", True)
        self.assertEqual(flat["quantity"], "T")
        for name, value in flat["values"].items():
            self.assertAlmostEqual(value, 0.5, places=9, msg=name)
            self.assertEqual(flat["coverage"][name], 1.0)
        step = P.optical_indices(x, np.where(x < 780, 100.0, 0.0), "T", True)
        self.assertAlmostEqual(step["values"]["lum"], 1.0, places=6)
        self.assertAlmostEqual(step["values"]["nir"], 0.0, places=9)
        self.assertAlmostEqual(step["values"]["uv"], 1.0, places=9)
        self.assertLess(step["values"]["sol"], step["values"]["vis"])
        absorbance = P.optical_indices(x, np.full_like(x, math.log10(2)), "A", False)
        self.assertAlmostEqual(absorbance["values"]["sol"], 0.5, places=9, msg="A = log10 2 is T = 50 %")
        reflectance = P.optical_indices(x, np.full_like(x, 0.25), "KM", False)
        self.assertEqual(reflectance["quantity"], "R")
        self.assertAlmostEqual(reflectance["values"]["nir"], 0.5, places=9, msg="F(R) = 0.25 is R = 50 %")
        value, coverage = P.weighted_average(x[:600], np.full(600, 50.0), "am15g", 300.0, 2500.0)
        self.assertAlmostEqual(value, 50.0)
        self.assertAlmostEqual(coverage, (799 - 300) / 2200)
        self.assertTrue(math.isnan(P.weighted_average(x[:2], [1.0, 2.0], "photopic", 380.0, 780.0)[0]))


if __name__ == "__main__":
    unittest.main()
