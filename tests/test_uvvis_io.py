"""Readers: instrument text/CSV layouts, JCAMP-DX, Excel, ordinate and unit detection, and the errors."""

from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from uvvis_io import (UvVisParseError, condition_summary, guess_y_mode, read_spectra, to_nm,
                      y_mode_from_text, _decode_asdf_line)

DATA = Path(__file__).resolve().parent / "data"


def read(name, **kwargs):
    return read_spectra((DATA / name).read_bytes(), name, **kwargs)


class DetectionTests(unittest.TestCase):
    def test_ordinate_named_by_headers(self):
        cases = {"%T": ("T", True), "T%": ("T", True), "T (%)": ("T", True), "Transmittance": ("T", False),
                 "Abs.": ("A", False), '"Abs."': ("A", False), "ABSORBANCE": ("A", False), "OD": ("A", False),
                 "%R": ("R", True), "Reflectance (%)": ("R", True), "R": ("R", False),
                 "F(R)": ("KM", False), "Kubelka-Munk": ("KM", False), "K-M": ("KM", False)}
        for text, expected in cases.items():
            self.assertEqual(y_mode_from_text(text), expected, text)
        for text in ("Probe A", "20 °C", "Sample T1", "Intensity", "", "a.u."):
            self.assertIsNone(y_mode_from_text(text), text)

    def test_values_decide_when_the_header_says_nothing(self):
        self.assertEqual(guess_y_mode(np.array([10.0, 50.0, 99.0])), ("T", True))
        self.assertEqual(guess_y_mode(np.array([0.1, 0.5, 0.99])), ("T", False))
        self.assertEqual(guess_y_mode(np.array([0.1, 1.5, 2.3])), ("A", False))
        self.assertEqual(guess_y_mode(np.array([-0.2, 150.0])), ("A", False))

    def test_wavelength_units_convert_to_nm(self):
        np.testing.assert_allclose(to_nm([1.0], "um"), [1000.0])
        np.testing.assert_allclose(to_nm([2.0], "eV"), [619.920992])
        np.testing.assert_allclose(to_nm([10000.0], "cm-1"), [1000.0])
        with self.assertRaises(UvVisParseError):
            to_nm([1.0], "furlong")


class JascoTests(unittest.TestCase):
    def test_text_export_with_extended_information(self):
        (spectrum,) = read("jasco_v770_pct_T.txt")
        self.assertEqual(spectrum.name, "jasco_v770_pct_T")
        self.assertEqual(len(spectrum.x), 2301)
        self.assertEqual((spectrum.x[0], spectrum.x[-1]), (200.0, 2500.0), "descending scans come back ascending")
        meta = spectrum.metadata
        self.assertEqual((meta["y_mode"], meta["y_percent"], meta["y_mode_guessed"]), ("T", True, False))
        self.assertEqual(meta["format"], "JASCO text/CSV")
        self.assertEqual(meta["instrument"], "V-770")
        self.assertEqual(meta["conditions"]["Scan speed"], "400 nm/min")
        self.assertIn("UV/Vis bandwidth 2 nm", condition_summary(spectrum))
        self.assertTrue(meta["x_descending"])

    def test_csv_export_in_cp932_with_absorbance(self):
        (spectrum,) = read("jasco_abs.csv")
        self.assertEqual(spectrum.metadata["title"], "試料_B")
        self.assertEqual((spectrum.metadata["y_mode"], spectrum.metadata["y_percent"]), ("A", False))
        self.assertGreater(spectrum.y.max(), 0.9)

    def test_npoints_mismatch_is_an_error(self):
        text = (DATA / "jasco_v770_pct_T.txt").read_text(encoding="utf-8").replace("NPOINTS\t2301", "NPOINTS\t2300")
        with self.assertRaises(UvVisParseError):
            read_spectra(text.encode("utf-8"), "x.txt")


class TableTests(unittest.TestCase):
    def test_uvprobe_blocks_become_one_spectrum_each(self):
        spectra = read("uvprobe_two_samples.txt")
        self.assertEqual([s.name for s in spectra], ["uvprobe_two_samples [Au_NP_1]", "uvprobe_two_samples [Au_NP_2]"])
        for s in spectra:
            self.assertEqual((s.metadata["y_mode"], s.metadata["y_percent"]), ("A", False))
            self.assertEqual(len(s.x), 1221)
        self.assertAlmostEqual(float(spectra[0].x[np.argmax(spectra[0].y)]), 520.0)

    def test_cary_paired_columns_with_different_lengths(self):
        spectra = read("cary_pairs.csv")
        self.assertEqual([s.name for s in spectra], ["cary_pairs [Pigment_red]", "cary_pairs [Pigment_blue]"])
        self.assertEqual((len(spectra[0].x), len(spectra[1].x)), (601, 501))
        self.assertEqual(spectra[1].x[0], 300.0)
        self.assertEqual((spectra[0].metadata["y_mode"], spectra[0].metadata["y_percent"]), ("R", True))
        self.assertEqual(spectra[0].metadata["format"], "paired-column table (Cary)")

    def test_header_lines_then_two_columns(self):
        (spectrum,) = read("hitachi_like.txt")
        self.assertEqual((spectrum.metadata["y_mode"], spectrum.metadata["y_percent"]), ("T", True))
        self.assertEqual(spectrum.x[-1], 2600.0)

    def test_semicolon_and_decimal_comma_with_several_columns(self):
        spectra = read("generic_semicolon_decimal_comma.csv")
        self.assertEqual([s.name for s in spectra], [f"generic_semicolon_decimal_comma [Probe {c}]" for c in "ABC"])
        self.assertAlmostEqual(float(spectra[0].y[0]), 0.1164)
        self.assertTrue(all(s.metadata["y_mode_guessed"] for s in spectra), "'Probe A' names no ordinate")
        spectra = read("generic_semicolon_decimal_comma.csv", default_y_mode=("R", True))
        self.assertEqual((spectra[0].metadata["y_mode"], spectra[0].metadata["y_percent"]), ("R", True))

    def test_two_columns_without_header_take_the_default_unit(self):
        (spectrum,) = read("energy_two_columns.txt")
        self.assertEqual(spectrum.metadata["x_unit"], "nm")
        (spectrum,) = read("energy_two_columns.txt", default_x_unit="eV")
        self.assertEqual(spectrum.metadata["x_unit"], "eV")
        self.assertAlmostEqual(float(spectrum.x.min()), 1239.841984 / 4.5, places=4)
        self.assertAlmostEqual(float(spectrum.x.max()), 1239.841984 / 1.5, places=4)

    def test_header_unit_wins_over_the_default(self):
        data = b"Energy (eV),Abs\n2.0,0.1\n3.0,0.5\n4.0,0.9\n"
        (spectrum,) = read_spectra(data, "e.csv", default_x_unit="nm")
        self.assertEqual(spectrum.metadata["x_unit"], "eV")
        self.assertAlmostEqual(float(spectrum.x[0]), 1239.841984 / 4.0, places=4)
        self.assertEqual(spectrum.metadata["y_mode"], "A")

    def test_percent_values_under_a_bare_t_header_are_percent(self):
        data = b"nm\tT\n400\t80.5\n500\t85.0\n600\t90.0\n"
        (spectrum,) = read_spectra(data, "t.txt")
        self.assertEqual((spectrum.metadata["y_mode"], spectrum.metadata["y_percent"]), ("T", True))

    def test_tabs_quotes_and_thousands_of_points(self):
        rows = "\n".join(f"{x:.1f}\t{0.5:.3f}" for x in np.linspace(190, 3300, 6221))
        (spectrum,) = read_spectra(('"Wavelength (nm)"\t"%T"\n' + rows).encode(), "big.txt")
        self.assertEqual(len(spectrum.x), 6221)


class JcampTests(unittest.TestCase):
    def test_asdf_tokens(self):
        self.assertEqual(_decode_asdf_line("500 A0A2A2A2A5A5A1"), (500.0, [10.0, 12.0, 12.0, 12.0, 15.0, 15.0, 11.0], False))
        self.assertEqual(_decode_asdf_line("500A0KU")[1], [10.0, 12.0, 14.0, 16.0])
        self.assertEqual(_decode_asdf_line("1000 123 j45 T"), (1000.0, [123.0, -22.0, -167.0], True))

    def test_affn_sqz_and_dif_files(self):
        (affn,) = read("jcamp_affn.jdx")
        self.assertEqual(affn.name, "glass_sample")
        self.assertEqual(len(affn.x), 301)
        np.testing.assert_allclose(affn.y[:3], 60 + 30 * np.sin(np.arange(400, 403) / 50), atol=0.006)
        self.assertEqual((affn.metadata["y_mode"], affn.metadata["y_percent"]), ("T", True), "TRANSMITTANCE with values up to 90 is percent")
        (sqz,) = read("jcamp_sqz.jdx")
        self.assertEqual(sqz.y.tolist(), [10.0, 12.0, 12.0, 12.0, 15.0, 15.0, 11.0])
        (dif,) = read("jcamp_dif.jdx")
        self.assertEqual(dif.y.tolist(), [10.0, 12.0, 14.0, 16.0, 16.0, 16.0, 16.0, 13.0], "the check value of the next line is dropped")
        self.assertEqual(dif.x.tolist(), [500.0 + k for k in range(8)])

    def test_compound_file_gives_one_spectrum_per_block(self):
        block = "##TITLE={title}\n##XUNITS=NANOMETERS\n##YUNITS=ABSORBANCE\n##XYPOINTS=(XY..XY)\n400,0.1 500,0.2\n600,0.3\n##END=\n"
        data = ("##TITLE=all\n##BLOCKS=2\n" + block.format(title="first") + block.format(title="second") + "##END=\n").encode()
        spectra = read_spectra(data, "two.jdx")
        self.assertEqual([s.name for s in spectra], ["first", "second"])
        self.assertEqual(spectra[1].y.tolist(), [0.1, 0.2, 0.3])


class ExcelTests(unittest.TestCase):
    def test_workbook_columns_and_sheets(self):
        spectra = read("workbook_two_sheets.xlsx")
        self.assertEqual([s.name for s in spectra], ["workbook_two_sheets [20 °C]", "workbook_two_sheets [90 °C]"])
        self.assertEqual(spectra[0].metadata["sheet"], "RT")
        self.assertEqual((spectra[0].metadata["y_mode"], spectra[0].metadata["y_percent"], spectra[0].metadata["y_mode_guessed"]), ("T", True, True))


class ErrorTests(unittest.TestCase):
    def test_binary_empty_and_unknown_inputs(self):
        with self.assertRaises(UvVisParseError):
            read_spectra(b"", "empty.txt")
        with self.assertRaises(UvVisParseError):
            read_spectra(b"\x00\x01\x02binary", "data.spc")
        with self.assertRaises(UvVisParseError):
            read_spectra(b"just some text\nwithout numbers\n", "notes.txt")
        with self.assertRaises(UvVisParseError):
            read_spectra(b"1 2\n", "one_point.txt")
        with self.assertRaises(UvVisParseError):
            read_spectra(b"a,b\n1,2\n3,4\n", "old.xls")
        with self.assertRaises(UvVisParseError):
            read_spectra(b"nm,T\n400,1\n500,2\n", "x.csv", default_x_unit="parsec")

    def test_nonpositive_wavelengths_are_dropped(self):
        (spectrum,) = read_spectra(b"-5 0.1\n0 0.2\n400 0.3\n500 0.4\n", "neg.txt")
        self.assertEqual(spectrum.x.tolist(), [400.0, 500.0])


if __name__ == "__main__":
    unittest.main()
