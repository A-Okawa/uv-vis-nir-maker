"""The real UI under AppTest: arrangement, conversions, Tauc, annotations, indices, projects, styles and languages."""

import ast
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import streamlit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness
from palettes import contrast_ratio, palette_sequence
from project_io import decode_project
from style_templates import TEMPLATE_KEYS, decode_style_template, encode_style_template

THREE = ("jasco_v770_pct_T.txt", "hitachi_like.txt", "workbook_two_sheets.xlsx")
EV_NM = 1239.841984


class DefaultRunTests(unittest.TestCase):
    def test_files_become_curves_in_the_graph_app_look(self):
        at = harness.run_app(harness.data_files(*THREE))
        r = harness.report(at)
        self.assertEqual(r["names"], ["jasco_v770_pct_T.txt", "hitachi_like.txt",
                                      "workbook_two_sheets.xlsx [workbook_two_sheets [20 °C]]", "workbook_two_sheets.xlsx [workbook_two_sheets [90 °C]]"])
        self.assertEqual(r["labels"], ["jasco_v770_pct_T", "hitachi_like", "workbook_two_sheets [20 °C]", "workbook_two_sheets [90 °C]"])
        self.assertEqual(r["colors"], ["#0072B2", "#D55E00", "#009E73", "#CC79A7"], "the graph app's paper palette in display order")
        self.assertEqual([c[0] for c in r["curves"]], r["labels"])
        self.assertTrue(all(c[2] == 0.0 for c in r["curves"]), "overlaid by default")
        self.assertEqual(r["mpl_legend"], r["labels"], "overlaid spectra keep the display order in the legend")
        self.assertEqual((r["mpl_xlabel"], r["mpl_ylabel"]), ("Wavelength (nm)", "Transmittance (%)"))
        self.assertEqual((r["preview_xlabel"], r["preview_ylabel"]), ("Wavelength (nm)", "Transmittance (%)"))
        self.assertEqual([round(v) for v in r["mpl_xlim"]], [200, 2600])
        self.assertGreater(len(r["mpl_yticks"]), 2, "y ticks are shown")
        settings = r["session"]["settings"]
        self.assertEqual(settings["figure_preset"], "figapp")
        self.assertEqual((settings["font_size"], settings["tick_font_size"], settings["line_width"], settings["axes_box"]), (7.0, 6.5, 0.9, False))
        self.assertAlmostEqual(settings["fig_width"] * 25.4, 88.0, places=6)
        self.assertEqual(r["png_header"][:4], b"\x89PNG")
        self.assertEqual(r["tiff_header"], b"II*\x00")
        self.assertIn("Transmittance", r["svg"])
        # The preview and the export draw the same numbers.
        for line, trace in zip(r["mpl_lines"], r["preview_lines"]):
            self.assertEqual(line[1], trace[1])
            np.testing.assert_allclose(line[3], trace[3])
            np.testing.assert_allclose(line[4], trace[4])

    def test_no_widget_passes_use_container_width_directly(self):
        tree = ast.parse((harness.APP_DIR / "app.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn("use_container_width", [kw.arg for kw in node.keywords], ast.unparse(node)[:80])
        for name in ("button", "download_button", "popover", "plotly_chart"):
            parameter = __import__("inspect").signature(getattr(streamlit, name)).parameters.get("width")
            new_api = parameter is not None and isinstance(parameter.default, str)
            self.assertIsNotNone(parameter if new_api else True)

    def test_without_files_only_the_prompt_shows(self):
        at = harness.run_app({})
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertNotIn("_regression_report", at.session_state)
        self.assertTrue(any("json" in i.value for i in at.info))


class ArrangementTests(unittest.TestCase):
    def test_order_visibility_stacking_and_absolute_positions(self):
        at = harness.run_app(harness.data_files(*THREE), {"ord_0": 3, "ord_1": 1, "ord_2": 2, "vis_3": False,
                                                           "stack": True, "global_offset": 1.0, "extra_offset_1": 0.5, "palette": "blues"})
        r = harness.report(at)
        self.assertEqual([c[0] for c in r["curves"]], ["hitachi_like", "workbook_two_sheets [20 °C]", "jasco_v770_pct_T"])
        self.assertEqual(r["mpl_legend"], ["jasco_v770_pct_T", "workbook_two_sheets [20 °C]", "hitachi_like"], "stacked: top to bottom")
        first, second, third = r["curves"]
        self.assertAlmostEqual(first[3], 0.5 * (90.0 - 1.648), places=2, msg="extra offset 0.5 lifts the first curve by half its height, from zero")
        self.assertAlmostEqual(second[3], 90.0 - 1.648, places=2, msg="the next curve starts at the previous one's height (the extra offset nudges one curve only)")
        self.assertGreater(third[3], second[4] - 1e-9)
        shades = r["session"]["settings"]
        self.assertEqual([shades[f"sp_color_{i}"] for i in (1, 2, 0)], palette_sequence("blues", 3), "blue shades over the visible spectra in display order")
        self.assertEqual(shades["sp_color_3"], palette_sequence("blues", 4)[3], "the hidden spectrum continues after them")
        at = harness.rerun(at, {"stack": False, "yoff_0": 100.0, "scale_0": 2.0, "vis_3": True})
        r = harness.report(at)
        by_name = {c[0]: c for c in r["curves"]}
        self.assertEqual(by_name["jasco_v770_pct_T"][2], 100.0)
        self.assertAlmostEqual(by_name["jasco_v770_pct_T"][4], 100.0 + 2 * 90.0, places=3, msg="scaled by 2 and shifted by 100")
        self.assertEqual(len(r["curves"]), 4)

    def test_line_styles_legend_position_y_range_and_hidden_ticks(self):
        at = harness.run_app(harness.data_files("workbook_two_sheets.xlsx"), {"ls_1": "dashed", "legend_loc": "outside right", "yrange_auto": False,
                                                                               "yrange": (0.0, 100.0), "y_major_tick": 25.0, "show_yticks": False, "show_minor": False})
        r = harness.report(at)
        self.assertEqual([l[2] for l in r["mpl_lines"]], ["-", "--"])
        self.assertEqual([t[2] for t in r["preview_lines"]], ["solid", "dash"])
        self.assertEqual(r["mpl_ylim"], [0.0, 100.0])
        self.assertEqual(r["mpl_yticks"], [])
        self.assertEqual(r["mpl_legend"], ["workbook_two_sheets [20 °C]", "workbook_two_sheets [90 °C]"])
        at = harness.rerun(at, {"show_yticks": True, "y_major_tick": 25.0})
        ticks = harness.report(at)["mpl_yticks"]
        self.assertEqual([t for t in ticks if 0.0 <= t <= 100.0], [0.0, 25.0, 50.0, 75.0, 100.0], "the locator's ticks inside the axis")
        at.number_input(key="_yrange_hi").set_value(60.0).run()
        self.assertEqual(harness.report(at)["mpl_ylim"], [0.0, 60.0])
        self.assertEqual(at.session_state["yrange"], (0.0, 60.0))
        at.number_input(key="_yrange_lo").set_value(80.0).run()
        self.assertEqual(at.session_state["yrange"], (0.0, 60.0), "a minimum above the maximum is not applied")
        self.assertTrue(any("上端" in w.value for w in at.warning))


class ConversionTests(unittest.TestCase):
    def test_transmittance_absorbance_and_fractions(self):
        at = harness.run_app(harness.data_files("jasco_v770_pct_T.txt"), {"y_mode": "A"})
        r = harness.report(at)
        self.assertEqual(r["mpl_ylabel"], "Absorbance")
        self.assertAlmostEqual(r["curves"][0][3], -np.log10(0.9), places=6)
        at = harness.rerun(at, {"y_mode": "T", "y_percent": False})
        r = harness.report(at)
        self.assertEqual(r["mpl_ylabel"], "Transmittance")
        self.assertAlmostEqual(r["curves"][0][4], 0.9, places=9)
        at = harness.rerun(at, {"y_mode": "as_measured"})
        self.assertEqual(harness.report(at)["mpl_ylabel"], "Transmittance (%)", "as measured: the data's own kind")

    def test_reflectance_data_cannot_become_transmittance(self):
        at = harness.run_app(harness.data_files("cary_pairs.csv", "jasco_v770_pct_T.txt"), {"y_mode": "KM"})
        r = harness.report(at)
        self.assertEqual([c[0] for c in r["curves"]], ["cary_pairs [Pigment_red]", "cary_pairs [Pigment_blue]"])
        self.assertEqual(r["skipped"], [["jasco_v770_pct_T", "T"]] if isinstance(r["skipped"][0], list) else [("jasco_v770_pct_T", "T")])
        self.assertTrue(any("jasco_v770_pct_T" in w.value for w in at.warning))
        self.assertIn("Kubelka", r["mpl_ylabel"])
        self.assertAlmostEqual(r["curves"][0][3], 0.025, places=4, msg="F(R) of 80 % reflectance is (0.2)²/1.6 = 0.025")

    def test_x_units_reverse_the_range_and_reset_the_ticks(self):
        at = harness.run_app(harness.data_files("hitachi_like.txt"), {"xrange": (400.0, 800.0)})
        at.selectbox(key="x_unit").set_value("eV").run()
        r = harness.report(at)
        self.assertEqual(r["mpl_xlabel"], "Photon energy (eV)")
        self.assertAlmostEqual(r["mpl_xlim"][0], EV_NM / 800.0, places=6)
        self.assertAlmostEqual(r["mpl_xlim"][1], EV_NM / 400.0, places=6)
        self.assertEqual((at.session_state["major_tick"], at.session_state["minor_tick"]), (0.5, 0.1))
        self.assertAlmostEqual(r["curves"][0][5], EV_NM / 800.0, places=6, msg="points are ordered by the displayed axis")
        at.selectbox(key="x_unit").set_value("cm-1").run()
        r = harness.report(at)
        self.assertEqual(r["mpl_xlim"], [12500.0, 25000.0])
        self.assertEqual(at.session_state["major_tick"], 5000.0)

    def test_labels_follow_the_quantity_unless_edited(self):
        at = harness.run_app(harness.data_files("jasco_v770_pct_T.txt"))
        at = harness.rerun(at, {"y_mode": "A"})
        self.assertEqual(at.session_state["y_label"], "Absorbance")
        at = harness.rerun(at, {"y_label": "Optical density"})
        at = harness.rerun(at, {"y_mode": "T"})
        self.assertEqual(at.session_state["y_label"], "Optical density", "an edited label stays")
        at = harness.rerun(at, {"x_label": "Wavelength, <i>λ</i> (nm)", "x_unit": "um"})
        r = harness.report(at)
        self.assertEqual(r["mpl_xlabel"], "Wavelength, $\\mathit{λ}$ (nm)")

    def test_input_defaults_for_headerless_files(self):
        at = harness.run_app(harness.data_files("generic_semicolon_decimal_comma.csv"), {"input_y_mode": "R%", "y_mode": "KM"})
        r = harness.report(at)
        self.assertEqual(len(r["curves"]), 3, "values read as %R convert to F(R)")
        at = harness.run_app(harness.data_files("energy_two_columns.txt"), {"input_x_unit": "eV"})
        r = harness.report(at)
        self.assertAlmostEqual(r["curves"][0][5], EV_NM / 4.5, places=3)


class TaucTests(unittest.TestCase):
    def test_band_gap_from_the_fit_and_its_table(self):
        at = harness.run_app(harness.data_files("direct_gap_abs.csv"), {"y_mode": "tauc", "xrange": (280.0, 830.0), "tauc_range": (3.3, 4.2)})
        r = harness.report(at)
        self.assertEqual(r["mpl_xlabel"], "Photon energy (eV)")
        self.assertIn("2", r["mpl_ylabel"])
        self.assertAlmostEqual(r["curves"][0][8], 3.0, places=2)
        self.assertTrue(any("3.00 eV" in t for t in r["mpl_texts"]))
        self.assertTrue(any("3.00 eV" in a for a in r["preview_annotations"]))
        self.assertTrue(any("3.00" in m.value for m in at.markdown), "the E_g table")
        at = harness.rerun(at, {"tauc_transition": "indirect", "show_eg": False})
        r = harness.report(at)
        self.assertIn("1/2", r["mpl_ylabel"])
        self.assertEqual(r["mpl_texts"], [])
        at = harness.rerun(at, {"tauc_fit": False})
        self.assertIsNone(harness.report(at)["curves"][0][8])

    def test_switching_to_tauc_brings_energy_ticks_and_back(self):
        at = harness.run_app(harness.data_files("direct_gap_abs.csv"))
        self.assertEqual((at.session_state["major_tick"], at.session_state["minor_tick"]), (200.0, 100.0))
        at.selectbox(key="y_mode").set_value("tauc").run()
        r = harness.report(at)
        self.assertEqual((at.session_state["major_tick"], at.session_state["minor_tick"], at.session_state["peak_decimals"]), (0.5, 0.1, 2))
        self.assertEqual(r["mpl_xlabel"], "Photon energy (eV)")
        at.selectbox(key="y_mode").set_value("A").run()
        self.assertEqual((at.session_state["major_tick"], at.session_state["minor_tick"]), (200.0, 100.0))
        self.assertEqual(harness.report(at)["mpl_xlabel"], "Wavelength (nm)")


class AnnotationTests(unittest.TestCase):
    def test_marks_bands_peaks_inset_and_side_labels(self):
        at = harness.run_app(harness.data_files("uvprobe_two_samples.txt"), {
            "xrange": (300.0, 800.0), "show_peaks": True, "peak_kind": "max", "marks_text": "520 SPR\n380-780 Vis\n",
            "show_zoom": True, "zoom_range": (480.0, 580.0), "show_side_labels": True, "stack": True, "_val_lbl_0": "Au_{NP} 1"})
        r = harness.report(at)
        self.assertIn("520", r["mpl_texts"])
        self.assertIn("530", r["mpl_texts"])
        self.assertIn("520 SPR", r["mpl_texts"])
        self.assertIn("Vis", r["mpl_texts"])
        self.assertTrue(any("Au$_{\\mathrm{NP}}$ 1" == t for t in r["mpl_texts"]), r["mpl_texts"])
        self.assertEqual(r["mpl_inset"], 1)
        self.assertEqual(r["mpl_patches"], 2, "the band and the inset indicator")
        self.assertEqual(r["preview_shapes"], 3, "band, mark line and inset indicator")
        self.assertIn("Au<sub>NP</sub> 1", [t[0] for t in r["preview_lines"]])
        at = harness.rerun(at, {"marks_label_mode": "none", "bands_label": False, "show_zoom": False})
        r = harness.report(at)
        self.assertNotIn("520 SPR", r["mpl_texts"])
        self.assertNotIn("Vis", r["mpl_texts"])
        self.assertEqual(r["mpl_inset"], 0)

    def test_peaks_are_found_in_nm_whatever_the_axis_unit(self):
        at = harness.run_app(harness.data_files("uvprobe_two_samples.txt"), {"xrange": (300.0, 800.0), "show_peaks": True, "peak_distance": 10.0})
        at.selectbox(key="x_unit").set_value("eV").run()
        r = harness.report(at)
        self.assertEqual(at.session_state["peak_decimals"], 2)
        self.assertIn(f"{EV_NM / 520:.2f}", r["mpl_texts"])
        self.assertIn(f"{EV_NM / 530:.2f}", r["mpl_texts"])

    def test_band_preset_button(self):
        at = harness.run_app(harness.data_files("hitachi_like.txt"))
        at.button(key="_band_preset").click().run()
        self.assertEqual(at.session_state["marks_text"], "300-380 UV\n380-780 Vis\n780-2500 NIR")
        at.button(key="_band_preset").click().run()
        self.assertEqual(at.session_state["marks_text"].count("NIR"), 1, "added once")


class IndexTests(unittest.TestCase):
    def test_table_values_and_partial_coverage(self):
        at = harness.run_app(harness.data_files("jasco_v770_pct_T.txt", "uvprobe_two_samples.txt", "cary_pairs.csv"), {"show_indices": True, "indices_decimals": 2})
        r = harness.report(at)
        rows = r["indices"]
        self.assertEqual(len(rows), 5)
        jasco = rows[0]
        self.assertEqual(jasco["量"], "T")
        for key in ("T_lum (%)", "T_sol (%)", "T_nir (%)"):
            self.assertNotIn("*", jasco[key])
            self.assertTrue(0 <= float(jasco[key]) <= 100)
        au = rows[1]
        self.assertIn("*", au["T_sol (%)"], "190–800 nm data covers the solar band only in part")
        self.assertNotIn("*", au["T_lum (%)"])
        cary = rows[3]
        self.assertEqual(cary["量"], "R")
        self.assertIn("R_nir (%)", cary)


class ProjectTests(unittest.TestCase):
    def test_save_and_reopen_keeps_everything(self):
        extra = {"y_mode": "A", "stack": True, "ord_0": 2, "ord_1": 1, "vis_2": False, "extra_offset_1": 0.3,
                 "_val_lbl_0": "Film <i>A</i>", "sp_color_1": "#8E24AA", "ls_1": "dotted", "marks_text": "550 T_{lum}",
                 "palette": "greens", "fig_width": 180 / 25.4, "font_size": 9.0, "legend_loc": "lower left", "xrange": (300.0, 2000.0)}
        at = harness.run_app(harness.data_files(*THREE), extra)
        before = harness.report(at)
        raw = json.dumps(before["session"]).encode("utf-8")
        project = decode_project(raw)
        self.assertEqual(set(project["spectra_files"]), set(THREE))
        reopened = harness.run_app({}, {"_test_restore": raw, "lang_radio": "English"})
        after = harness.report(reopened)
        self.assertEqual(after["curves"], before["curves"])
        self.assertEqual(after["labels"], before["labels"])
        self.assertEqual(after["colors"], before["colors"])
        self.assertEqual(after["line_styles"], before["line_styles"])
        self.assertEqual(after["mpl_lines"], before["mpl_lines"])
        self.assertEqual(reopened.session_state["lang_radio"], "English", "the screen language is the viewer's own")
        self.assertEqual(reopened.session_state["legend_loc"], "lower left")
        self.assertTrue(any(s.value for s in reopened.sidebar.success))

    def test_new_project_clears_the_work(self):
        at = harness.run_app(harness.data_files("hitachi_like.txt"), {"_val_lbl_0": "x", "y_mode": "A"})
        harness.report(at)
        at = harness.rerun(at, {"_test_new_project": True})
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertEqual(at.session_state["_spectra_bytes"], {})
        self.assertNotIn("_val_lbl_0", at.session_state)
        self.assertEqual(at.session_state["y_mode"], "as_measured")

    def test_style_template_applies_appearance_only(self):
        at = harness.run_app(harness.data_files("hitachi_like.txt"), {"font_size": 9.0, "line_width": 1.5, "legend_loc": "lower right", "y_mode": "A"})
        harness.report(at)
        style = encode_style_template({k: at.session_state[k] for k in TEMPLATE_KEYS if k in at.session_state})
        decoded = decode_style_template(style)
        self.assertEqual((decoded["font_size"], decoded["line_width"], decoded["legend_loc"]), (9.0, 1.5, "lower right"))
        self.assertNotIn("y_mode", decoded)
        other = harness.run_app(harness.data_files("jasco_v770_pct_T.txt"), {"_test_style": style})
        self.assertFalse(other.exception, [e.message for e in other.exception])
        self.assertEqual((other.session_state["font_size"], other.session_state["line_width"], other.session_state["legend_loc"]), (9.0, 1.5, "lower right"))
        self.assertEqual(other.session_state["y_mode"], "as_measured")
        self.assertTrue(any("スタイル" in s.value or "style" in s.value for s in other.success))
        broken = harness.run_app(harness.data_files("jasco_v770_pct_T.txt"), {"_test_style": b"{}"})
        self.assertTrue(any(e.value for e in broken.error))

    def test_file_set_changes_keep_settings_by_name(self):
        files = harness.data_files("jasco_v770_pct_T.txt", "hitachi_like.txt")
        at = harness.run_app(files, {"_val_lbl_1": "Hitachi", "sp_color_1": "#8E24AA", "ord_1": 1, "ord_0": 2})
        harness.report(at)
        at = harness.rerun(at, {"_spectra_bytes": {"hitachi_like.txt": files["hitachi_like.txt"]}})
        r = harness.report(at)
        self.assertEqual(r["labels"], ["Hitachi"])
        self.assertEqual(r["colors"], ["#8E24AA"])


NINE = {f"s{g}_{k}.csv": ("Wavelength (nm),%R\n" + "\n".join(f"{x},{20 + 5 * g + 3 * k + 0.02 * (x - 300)}" for x in range(300, 801, 10))).encode()
        for g in range(3) for k in range(3)}


class GroupedShadeTests(unittest.TestCase):
    def test_three_groups_of_three_take_blue_yellow_and_vermilion_shades(self):
        at = harness.run_app(NINE, {"palette_group_size": 3})
        r = harness.report(at)
        blues, yellows, oranges = (palette_sequence(h, 3) for h in ("blues", "yellows", "oranges"))
        self.assertEqual(r["colors"], blues + yellows + oranges)
        for color in r["colors"]:
            self.assertGreaterEqual(contrast_ratio(color), 3.0, color)
        at = harness.rerun(at, {"vis_1": False})
        self.assertEqual(harness.report(at)["colors"], blues + yellows + oranges, "hiding one spectrum keeps the groups in place")
        at.multiselect(key="palette_group_hues").set_value(["greens", "purples"]).run()
        r = harness.report(at)
        self.assertEqual(r["colors"][3:6], palette_sequence("purples", 3))
        self.assertEqual(r["colors"][6:9], palette_sequence("greens", 3), "hues repeat when groups outnumber them")
        self.assertEqual(r["session"]["settings"]["palette_group_hues"], ["greens", "purples"])
        at.number_input(key="palette_group_size").set_value(0).run()
        colors = harness.report(at)["colors"]
        self.assertEqual([colors[0], colors[2], colors[3]], ["#0072B2", "#D55E00", "#009E73"], "back to the single palette over the visible spectra")

    def test_range_leaves_a_reference_outside_the_groups(self):
        at = harness.run_app(NINE, {"palette_group_size": 4, "palette_from": 2})
        colors = harness.report(at)["colors"]
        self.assertEqual(colors[0], "#000000")
        self.assertEqual(colors[1:5], palette_sequence("blues", 4))
        self.assertEqual(colors[5:9], palette_sequence("yellows", 4))


class LegendFitTests(unittest.TestCase):
    LONG = {f"_val_lbl_{i}": f"Sample {i + 1} sieved under 45 µm, annealed at 1200 °C for 10 h in air, second batch" for i in range(9)}

    def inside(self, box, outer, tolerance=1.0):
        return box[0] >= outer[0] - tolerance and box[1] >= outer[1] - tolerance and box[2] <= outer[2] + tolerance and box[3] <= outer[3] + tolerance

    def test_long_legends_stay_inside_the_axes_and_the_figure(self):
        at = harness.run_app(NINE, self.LONG)
        r = harness.report(at)
        box = r["legend_box"]
        self.assertTrue(self.inside(box[:4], box[6:10]), box)
        self.assertLess(r["legend_layout"]["font"], 6.5, "the text shrank to fit")
        at = harness.run_app(NINE, {"legend_loc": "outside right", **{f"_val_lbl_{i}": f"Sample {i + 1}, 25–45 µm, 1200 °C" for i in range(9)}})
        r = harness.report(at)
        box = r["legend_box"]
        self.assertTrue(self.inside(box[:4], [0, 0, box[4], box[5]]), box)
        self.assertGreaterEqual(box[0], box[8] - 1.0, "right of the axes")
        self.assertLess(r["legend_layout"]["right"], 0.9)
        at = harness.rerun(at, {"legend_fit": False})
        r = harness.report(at)
        self.assertEqual(r["legend_layout"], {"font": 6.5, "right": 0.965})

    def test_short_legends_keep_their_size_and_columns_apply(self):
        at = harness.run_app(NINE, {"legend_ncol": 3})
        r = harness.report(at)
        self.assertEqual(r["legend_layout"]["font"], 6.5)
        box = r["legend_box"]
        self.assertGreater(box[2] - box[0], 2.5 * (box[3] - box[1]), "three columns: wider than tall")


class LanguageTests(unittest.TestCase):
    def test_english_screen_and_switching_keeps_the_work(self):
        at = harness.run_app(harness.data_files("hitachi_like.txt"), {"palette": "greens"}, language="English")
        self.assertFalse(at.exception, [e.message for e in at.exception])
        headers = [h.value for h in at.sidebar.header]
        self.assertIn("⚙️ Graph Settings", headers)
        self.assertIn("📐 Figure Size & Export", headers)
        colors = [at.session_state["sp_color_0"]]
        at.radio(key="lang_radio").set_value("日本語").run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertIn("⚙️ グラフ設定", [h.value for h in at.sidebar.header])
        self.assertEqual([at.session_state["sp_color_0"]], colors)
        self.assertEqual(at.session_state["palette"], "greens")


if __name__ == "__main__":
    unittest.main()
