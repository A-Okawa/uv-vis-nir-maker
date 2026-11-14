"""Projects, style templates and the annotation helpers."""

import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from project_io import (GLOBAL_KEYS, PATTERN_KEY, ProjectError, atomic_save, clean_settings, decode_project, encode_project,
                        project_fingerprint)
from style_templates import TEMPLATE_KEYS, decode_style_template, encode_style_template
from plot_extras import (PeakMark, PreparedCurve, add_mpl_bands, add_mpl_marks, add_mpl_zoom_inset, add_plotly_bands,
                         add_plotly_marks, add_plotly_zoom_inset, mark_text, parse_marks, top_labels_height)

SETTINGS = {
    "xrange": (300.0, 2500.0), "x_unit": "nm", "y_mode": "T", "y_percent": True, "tauc_transition": "direct", "tauc_fit": True,
    "tauc_range": (2.5, 3.2), "show_eg": True, "normalize_mode": "none", "norm_window": (400.0, 600.0), "norm_at": 550.0,
    "stack": False, "global_offset": 1.0, "show_legend": True, "legend_loc": "upper left", "yrange_auto": False, "yrange": (0.0, 100.0),
    "y_major_tick": 20.0, "show_yticks": True, "major_tick": 500.0, "show_minor": True, "minor_tick": 100.0,
    "x_label": "Wavelength (nm)", "y_label": "Transmittance (%)", "input_x_unit": "nm", "input_y_mode": "auto",
    "smooth_on": False, "smooth_window": 11, "smooth_order": 3, "show_peaks": True, "peak_kind": "min", "peak_prom": 0.05,
    "peak_distance": 10.0, "peak_decimals": 0, "peak_label_rotate": False, "marks_text": "550 T_{lum}\n780-2500 NIR",
    "marks_style": "dashed", "marks_color": "#555555", "marks_label_mode": "rotated", "marks_show_value": True, "marks_decimals": 0,
    "marks_fontsize": 7, "bands_color": "#BBBBBB", "bands_alpha": 0.25, "bands_label": True, "show_zoom": False,
    "zoom_range": (400.0, 800.0), "zoom_location": "upper right", "annotation_font_size": 7, "show_indices": True, "indices_decimals": 1,
    "show_side_labels": True, "label_side": "左", "label_fontsize": 7, "label_offset_x": 2.0, "label_offset_y": 0.5,
    "fig_width": 3.46, "fig_height": 2.76, "dpi_export": 300, "font_size": 7.0, "tick_font_size": 6.5, "legend_font_size": 6.5,
    "show_panel": True, "lang_radio": "日本語", "palette": "blues", "palette_reverse": False, "palette_steps": 0, "palette_from": 2,
    "palette_to": 0, "palette_base_color": "#000000", "font_family": "Arial", "line_width": 0.9, "axes_box": False,
    "vis_0": True, "ord_0": 2, "_val_lbl_0": "VO_{2} 20 °C", "sp_color_0": "#0072B2", "ls_0": "dashed", "extra_offset_0": 0.2,
    "yoff_0": 5.0, "scale_0": 2.0,
}


class ProjectTests(unittest.TestCase):
    def test_round_trip_keeps_files_and_settings(self):
        files = {"a.csv": b"nm,%T\n400,50\n500,60\n", "b.txt": "試料\t%T\n400\t10\n500\t20\n".encode("cp932")}
        raw = encode_project(files, {**SETTINGS, "_private": 1, "unknown_key": 2})
        document = json.loads(raw)
        self.assertEqual(document["format"], "uvvisnir-maker-project")
        self.assertNotIn("_private", document["settings"])
        self.assertNotIn("unknown_key", document["settings"])
        project = decode_project(raw)
        self.assertEqual(project["spectra_files"], files)
        for key, value in SETTINGS.items():
            restored = project["settings"][key]
            self.assertEqual(tuple(restored) if isinstance(restored, list) else restored, value, key)

    def test_invalid_settings_are_rejected_before_anything_changes(self):
        bad = [{"y_mode": "reflectivity"}, {"xrange": (900.0, 300.0)}, {"xrange": (0.0, 300.0)}, {"tauc_range": (3.0, 2.0)},
               {"sp_color_0": "blue"}, {"ls_0": "wavy"}, {"ord_0": 0}, {"scale_0": 0.0}, {"legend_loc": "nowhere"},
               {"marks_text": "x" * 20001}, {"input_y_mode": "%A"}, {"dpi_export": 450}, {"palette": "rainbow"},
               {"label_side": "上"}, {"lang_radio": "Deutsch"}, {"yrange": (1.0, 1.0)}, {"peak_kind": "both"},
               {"smooth_window": 5.5}, {"margin_left": 0.9, "margin_right": 0.5}]
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(ProjectError):
                clean_settings({**SETTINGS, **settings})

    def test_damaged_projects(self):
        raw = encode_project({"a.csv": b"nm,%T\n400,50\n500,60\n"}, SETTINGS)
        with self.assertRaises(ProjectError):
            decode_project(b"not json")
        with self.assertRaises(ProjectError):
            decode_project(json.dumps({"format": "xrd-maker-project", "version": 3, "settings": {}}).encode())
        document = json.loads(raw)
        document["checksums"]["spectra_files"]["a.csv"] = "0" * 64
        with self.assertRaises(ProjectError):
            decode_project(json.dumps(document).encode())
        document = json.loads(raw)
        document["spectra_files"]["a.csv"] = "***"
        with self.assertRaises(ProjectError):
            decode_project(json.dumps(document).encode())
        with self.assertRaises(ProjectError):
            encode_project({"": b"x"}, SETTINGS)
        with self.assertRaises(ProjectError):
            encode_project({"a.csv": "text, not bytes"}, SETTINGS)

    def test_fingerprint_ignores_the_save_time_and_tracks_settings(self):
        a = encode_project({"a.csv": b"1 2\n3 4\n"}, SETTINGS)
        b = encode_project({"a.csv": b"1 2\n3 4\n"}, SETTINGS)
        self.assertEqual(project_fingerprint(a), project_fingerprint(b))
        c = encode_project({"a.csv": b"1 2\n3 4\n"}, {**SETTINGS, "global_offset": 1.5})
        self.assertNotEqual(project_fingerprint(a), project_fingerprint(c))

    def test_atomic_save_replaces_the_file_completely(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sub" / "autosave.json"
            atomic_save(path, b"first")
            atomic_save(path, b"second")
            self.assertEqual(path.read_bytes(), b"second")
            self.assertEqual([p.name for p in path.parent.iterdir()], ["autosave.json"])

    def test_every_global_key_has_a_rule(self):
        for key in GLOBAL_KEYS:
            if key in SETTINGS:
                continue
            self.assertIn(key, ("size_unit", "figure_preset", "reference_line_width", "axes_line_width", "tick_length",
                                "tick_direction", "margin_left", "margin_right", "margin_bottom", "margin_top",
                                "transparent_background", "svg_text_editable"), key)
        self.assertTrue(PATTERN_KEY.fullmatch("ls_12"))
        self.assertFalse(PATTERN_KEY.fullmatch("ls_x"))


class StyleTemplateTests(unittest.TestCase):
    def test_round_trip_keeps_only_appearance(self):
        raw = encode_style_template(SETTINGS, "Paper")
        document = json.loads(raw)
        self.assertEqual(document["format"], "uvvisnir-maker-style")
        for key in ("marks_text", "normalize_mode", "y_mode", "x_unit", "sp_color_0", "palette"):
            self.assertNotIn(key, document["settings"])
        restored = decode_style_template(raw)
        self.assertEqual(restored["legend_loc"], "upper left")
        self.assertEqual(restored["marks_label_mode"], "rotated")
        self.assertEqual(restored["y_major_tick"], 20.0)
        self.assertTrue(set(restored) <= TEMPLATE_KEYS)

    def test_sister_styles_open_without_their_axis_settings(self):
        for fmt in ("xrd-maker-style", "raman-maker-style"):
            sister = {"format": fmt, "version": 1, "name": "x",
                      "settings": {"font_size": 9.0, "x_label": "2θ", "major_tick": 10.0, "line_width": 1.0, "axes_box": False, "cif_label_fontsize": 6}}
            restored = decode_style_template(json.dumps(sister).encode())
            self.assertEqual((restored["font_size"], restored["line_width"], restored["axes_box"]), (9.0, 1.0, False))
            self.assertNotIn("x_label", restored)
            self.assertNotIn("major_tick", restored)
            self.assertNotIn("cif_label_fontsize", restored)

    def test_rejects_other_documents(self):
        with self.assertRaises(ProjectError):
            decode_style_template(b"[]")
        with self.assertRaises(ProjectError):
            decode_style_template(json.dumps({"format": "uvvisnir-maker-style", "version": 1, "settings": {"marks_text": "x"}}).encode())
        with self.assertRaises(ProjectError):
            decode_style_template(json.dumps({"format": "uvvisnir-maker-style", "version": 2, "settings": {}}).encode())


class MarkTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_parse(self):
        marks, bands = parse_marks("550 <i>T</i>_{lum}\n1000, one micron\n780\n780-2500 NIR\n  # a note\nabc\n380–780\n2.5 to 3 eV x\n")
        self.assertEqual([(m.x, m.label) for m in marks], [(550.0, "<i>T</i>_{lum}"), (1000.0, "one micron"), (780.0, "")])
        self.assertEqual([(b.low, b.high, b.label) for b in bands], [(780.0, 2500.0, "NIR"), (380.0, 780.0, ""), (2.5, 3.0, "eV x")])
        self.assertEqual(mark_text(PeakMark(550.0, "T"), True, 1), "550.0 T")
        self.assertEqual(mark_text(PeakMark(550.0, "T"), False, 0), "T")
        self.assertEqual(mark_text(PeakMark(550.0), False, 0), "550")
        self.assertEqual(parse_marks(""), ([], []))

    def test_matplotlib_marks_bands_and_inset_keep_the_limits(self):
        fig, ax = plt.subplots()
        x = np.arange(300.0, 2501.0, 1.0)
        curve = PreparedCurve("s", x, 50 + 40 * np.sin(x / 300), "#0072B2", 0.9)
        ax.plot(curve.x, curve.y)
        ax.set_xlim(300, 2500)
        ax.set_ylim(0, 100)
        marks, bands = parse_marks("550 T_{lum}\n3000 out\n780-2500 NIR\n2600-2700 out")
        line_artists = add_mpl_marks(ax, marks, x_range=(300, 2500), label_mode="rotated", rich=lambda s: s)
        band_artists = add_mpl_bands(ax, bands, x_range=(300, 2500), rich=lambda s: s)
        self.assertEqual(len(line_artists), 2, "one line and one label; the mark outside the range is skipped")
        self.assertEqual(len(band_artists), 2, "one band and one label")
        inset = add_mpl_zoom_inset(ax, [curve], (400.0, 800.0), location="lower right", unit_text="nm")
        self.assertIsNotNone(inset)
        self.assertEqual(ax.get_xlim(), (300.0, 2500.0))
        self.assertEqual(ax.get_ylim(), (0.0, 100.0))
        self.assertIsNone(add_mpl_zoom_inset(ax, [curve], (5000.0, 6000.0)))
        with self.assertRaises(ValueError):
            add_mpl_zoom_inset(ax, [curve], (800.0, 400.0))
        self.assertGreater(top_labels_height(marks, bands, x_range=(300, 2500), label_mode="rotated", plain=lambda s: s), 0)
        self.assertEqual(top_labels_height([], bands, x_range=(300, 2500), plain=lambda s: s), 0.0)

    def test_plotly_marks_bands_and_inset(self):
        fig = go.Figure()
        x = np.arange(300.0, 2501.0, 1.0)
        curve = PreparedCurve("s", x, 50 + 40 * np.sin(x / 300), "#0072B2", 0.9)
        fig.add_trace(go.Scatter(x=curve.x, y=curve.y, mode="lines", name="s"))
        fig.update_xaxes(range=[300, 2500])
        marks, bands = parse_marks("550 T\n780-2500 NIR")
        add_plotly_marks(fig, marks, x_range=(300, 2500))
        add_plotly_bands(fig, bands, x_range=(300, 2500))
        self.assertEqual(len(fig.layout.shapes), 2)
        self.assertEqual([a.text for a in fig.layout.annotations], ["550 T", "NIR"])
        refs = add_plotly_zoom_inset(fig, [curve], (400.0, 800.0), unit_text="nm")
        self.assertEqual(refs, ("x2", "y2"))
        self.assertEqual(len([t for t in fig.data if t.xaxis == "x2"]), 2)


if __name__ == "__main__":
    unittest.main()
