import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
import plotly.graph_objects as go
import inspect
import os
import re
from datetime import datetime
from pathlib import Path

from project_io import (
    GLOBAL_KEYS, PATTERN_KEY, X_LIMITS, ENERGY_LIMITS, X_UNITS, Y_MODES, INPUT_Y_MODES, NORMALIZE_MODES,
    TAUC_TRANSITIONS, LEGEND_LOCATIONS, LINE_STYLES, ProjectError, encode_project, decode_project,
    atomic_save, project_fingerprint, clean_settings,
)
from uvvis_io import read_spectra, UvVisParseError, SUPPORTED_EXTENSIONS, condition_summary
from uvvis_processing import (
    EV_NM, X_UNIT_LABELS, X_UNIT_TEXT, Y_MODE_LABELS, TAUC_LABELS, x_from_nm, range_in_unit, convert_y,
    tauc_transform, tauc_fit, smooth, normalize, detect_peaks, optical_indices, WEIGHTED_BANDS,
)
from figure_style import (
    STYLE_DEFAULTS, STYLE_KEYS, FIGURE_PRESETS_MM, FONT_FAMILIES, METRIC_COMPATIBLE, mm_to_inches, inches_to_mm,
    font_rcparams, apply_mpl_style, export_figure, resolve_font, css_font_stack, validate_style_settings,
)
from palettes import (
    PALETTES, PALETTE_NAMES, DEFAULT_PALETTE, REFERENCE_COLORS, MAX_SHADE_STEPS,
    palette_sequence, swatch_colors, load_colors, color_emoji,
)
from rich_text import rich, to_html, plain
from style_templates import encode_style_template, decode_style_template
from plot_extras import (
    PreparedCurve, parse_marks, add_mpl_marks, add_plotly_marks, add_mpl_bands, add_plotly_bands,
    add_mpl_zoom_inset, add_plotly_zoom_inset, top_labels_height,
)

APP_VERSION = "1.0"
LAST_UPDATED = "2026-10-30"
LOCAL_MODE = os.environ.get("UVVIS_MAKER_LOCAL") == "1"
# Named colours in the graph app's format; copy that app's colors.csv here to share its names.
COLORS_FILE = Path(__file__).resolve().parent / "colors.csv"
LOCAL_DATA_DIR = (Path(os.environ["UVVIS_MAKER_DATA_DIR"]).expanduser()
                  if LOCAL_MODE and os.environ.get("UVVIS_MAKER_DATA_DIR") else None)
LANGUAGES = {"日本語": "ja", "English": "en"}
# Tick intervals that suit each axis unit (major, minor).
UNIT_TICKS = {"nm": (200.0, 100.0), "um": (0.5, 0.1), "eV": (0.5, 0.1), "cm-1": (5000.0, 1000.0)}
PLOTLY_DASH = {"solid": "solid", "dashed": "dash", "dotted": "dot", "dashdot": "dashdot"}
MPL_LINESTYLE = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
BAND_PRESET = "300-380 UV\n380-780 Vis\n780-2500 NIR"
# The graph app's look (the "figapp" preset) is also how a new project starts, as in Raman Maker:
# Arial 7 pt labels, 6.5 pt ticks and legend, 0.5 pt axes, 0.9 pt lines, 2.5 pt outward ticks, no top/right frame.
FIGAPP_STYLE = dict(font_family="Arial", font_size=7.0, tick_font_size=6.5, legend_font_size=6.5,
                    label_fontsize=7, annotation_font_size=6, marks_fontsize=6,
                    line_width=0.9, reference_line_width=0.6, axes_line_width=0.5, tick_length=2.5, tick_direction="out",
                    axes_box=False, margin_left=.15, margin_right=.965, margin_bottom=.14, margin_top=.97)

# ===== 翻訳辞書 / Translation dictionary =====
TRANSLATIONS = {
    "ja": {
        "password_prompt": "パスワードを入力してください",
        "login": "ログイン",
        "wrong_password": "パスワードが違います",
        "app_caption": "複数スペクトル重ね合わせ・%T / Abs / %R / F(R) / Tauc 変換・T_lum / T_sol・論文用 PDF / SVG / TIFF 出力",
        "label": "ラベル",
        "add_italic": "＋斜体",
        "italic_help": "末尾に `<i>TEXT</i>` を追加。TEXT を書き換えて使用。文字を選んで Ctrl+I（Mac は ⌘+I）でも斜体",
        "add_subscript": "＋x₂",
        "subscript_help": "下付き：末尾に `_{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+,（カンマ）でも下付き",
        "add_superscript": "＋x²",
        "superscript_help": "上付き：末尾に `^{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+.（ピリオド）でも上付き",
        "change_color": "🎨 色を変更",
        "session_header": "💾 セッション",
        "load_session": "保存した作業を開く (.json)",
        "project_help": "測定データ、表示順、色、ラベル、変換、図の設定をまとめて保存・再開できます。",
        "project_loaded": "保存した作業を読み込みました。",
        "project_error": "作業ファイルを開けませんでした: ",
        "new_project": "新しい作業を始める",
        "resume_local": "前回の自動保存から再開",
        "autosaved": "PCに自動保存済み · {time}",
        "autosave_error": "自動保存できませんでした。作業ファイルをダウンロードして保存してください: ",
        "local_mode": "PC版 · データと自動保存はこのPC内に保存されます。",
        "data_header": "📂 UV-Vis-NIR データ",
        "data_upload": "スペクトル（JASCO・Shimadzu・Cary・Hitachi・PerkinElmer のテキスト/CSV、JCAMP-DX、xlsx）",
        "native_help": "装置ソフトのテキスト/CSV 出力をそのまま読み込めます。1 ファイルに複数のスペクトル（複数列・複数ブロック）があれば別々に表示します。",
        "read_error": "測定データを読み込めませんでした: ",
        "duplicate_names": "同名のファイルがあります。ファイル名を変更して選び直してください（現在の作業は保持しています）: ",
        "project_save_error": "作業ファイルを保存できません。データ量やラベルを確認してください: ",
        "scan_count": "{files} ファイル · {scans} スペクトル",
        "data_range": "測定データの全範囲を表示",
        "download_desktop": "PC用をダウンロード (.zip)",
        "desktop_help": "Mac・Windows対応。初回セットアップ後はオフラインで使用できます。",
        "restored": "復元済み: ",
        "graph_settings": "⚙️ グラフ設定",
        "xrange": "波長範囲 (nm)",
        "show_legend_cb": "凡例を表示",
        "tick_settings": "目盛り設定",
        "major_tick": "横軸 主目盛り間隔",
        "show_minor_cb": "副目盛りを表示",
        "minor_tick": "横軸 副目盛り間隔",
        "sample_labels": "サンプルラベル（グラフ内）",
        "show_side_labels_cb": "グラフ内にラベルを表示",
        "label_pos": "ラベル位置",
        "left": "左",
        "right": "右",
        "label_fontsize": "ラベル文字サイズ",
        "label_offset_x": "横オフセット（軸幅の %）",
        "label_offset_x_help": "正=グラフ内側、負=グラフ外側",
        "label_offset_y": "縦オフセット（曲線高さ比）",
        "label_offset_y_help": "0=曲線の最小値、1.0=最大値付近",
        "fig_size": "📐 図サイズ・出力",
        "save_session": "💾 作業を保存・ダウンロード (.json)",
        "show_panel": "⚙️ スペクトル設定パネルを表示",
        "spectra_md": "#### スペクトル",
        "display_order": "表示順",
        "show_cb": "表示する",
        "line_style": "線種",
        "offset_adj": "オフセット調整",
        "y_position": "Y位置（絶対値）",
        "scale": "倍率",
        "drag_zoom": "ドラッグで範囲ズーム ／ ダブルクリックでリセット",
        "interactive_preview": "インタラクティブ",
        "export_preview": "保存画像のプレビュー",
        "export_preview_caption": "図サイズ: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI（プレビューは150 DPI）",
        "save_tiff": "📥 TIFF として保存 ({dpi} DPI)",
        "save_png": "📥 PNG として保存 ({dpi} DPI)",
        "upload_prompt": "スペクトルのファイルを選択するか、保存した作業ファイル（.json）を開いてください。",
    },
    "en": {
        "password_prompt": "Enter password",
        "login": "Login",
        "wrong_password": "Incorrect password",
        "app_caption": "Multi-spectrum overlay · %T / Abs / %R / F(R) / Tauc conversion · T_lum / T_sol · Publication-quality PDF / SVG / TIFF export",
        "label": "Label",
        "add_italic": "+Italic",
        "italic_help": "Appends `<i>TEXT</i>`; replace TEXT. Or select text and press Ctrl+I (⌘+I on a Mac).",
        "add_subscript": "+x₂",
        "subscript_help": "Subscript: appends `_{N}`; replace N. Or select text and press Ctrl+, (comma).",
        "add_superscript": "+x²",
        "superscript_help": "Superscript: appends `^{N}`; replace N. Or select text and press Ctrl+. (period).",
        "change_color": "🎨 Change color",
        "session_header": "💾 Session",
        "load_session": "Open saved project (.json)",
        "project_help": "Save and reopen measurements, order, colors, labels, conversions and figure settings together.",
        "project_loaded": "Project restored.",
        "project_error": "Could not open the project: ",
        "new_project": "Start a new project",
        "resume_local": "Resume the previous autosave",
        "autosaved": "Autosaved on this PC · {time}",
        "autosave_error": "Could not autosave. Download the project to keep your work: ",
        "local_mode": "PC edition · Data and autosaves stay on this computer.",
        "data_header": "📂 UV-Vis-NIR Data",
        "data_upload": "Spectra (JASCO, Shimadzu, Cary, Hitachi, PerkinElmer text/CSV, JCAMP-DX, xlsx)",
        "native_help": "Instrument text/CSV exports are read as they are. Several spectra in one file (columns or blocks) are shown separately.",
        "read_error": "Could not read measurement data: ",
        "duplicate_names": "Files have duplicate names. Rename them and select again; your current work is kept: ",
        "project_save_error": "Could not save the project. Check the data size and labels: ",
        "scan_count": "{files} files · {scans} spectra",
        "data_range": "Show the full measured range",
        "download_desktop": "Download PC edition (.zip)",
        "desktop_help": "For Mac and Windows. Works offline after the first setup.",
        "restored": "Restored: ",
        "graph_settings": "⚙️ Graph Settings",
        "xrange": "Wavelength range (nm)",
        "show_legend_cb": "Show legend",
        "tick_settings": "Tick settings",
        "major_tick": "x major tick interval",
        "show_minor_cb": "Show minor ticks",
        "minor_tick": "x minor tick interval",
        "sample_labels": "Sample labels (in plot)",
        "show_side_labels_cb": "Show labels in plot",
        "label_pos": "Label position",
        "left": "Left",
        "right": "Right",
        "label_fontsize": "Label font size",
        "label_offset_x": "Horizontal offset (% of axis width)",
        "label_offset_x_help": "Positive=inside, Negative=outside",
        "label_offset_y": "Vertical offset (curve height ratio)",
        "label_offset_y_help": "0=minimum of the curve, 1.0=near its maximum",
        "fig_size": "📐 Figure Size & Export",
        "save_session": "💾 Save and download project (.json)",
        "show_panel": "⚙️ Show spectrum settings panel",
        "spectra_md": "#### Spectra",
        "display_order": "Display order",
        "show_cb": "Show",
        "line_style": "Line style",
        "offset_adj": "Offset adjustment",
        "y_position": "Y position (absolute)",
        "scale": "Scale factor",
        "drag_zoom": "Drag to zoom / Double-click to reset",
        "interactive_preview": "Interactive",
        "export_preview": "Export preview",
        "export_preview_caption": "Figure size: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI (preview: 150 DPI)",
        "save_tiff": "📥 Save as TIFF ({dpi} DPI)",
        "save_png": "📥 Save as PNG ({dpi} DPI)",
        "upload_prompt": "Choose spectrum files or open a saved project (.json) to continue.",
    },
}

st.set_page_config(page_title="UV-Vis-NIR Maker", page_icon="🌈", layout="wide")
# Full-width preview image on old and new Streamlit (use_column_width now shows a warning box).
_IMAGE_PARAMETERS = inspect.signature(st.image).parameters
FULL_WIDTH_IMAGE = ({"width": "stretch"} if "int" not in str(_IMAGE_PARAMETERS["width"].annotation)
                    else {"use_container_width": True} if "use_container_width" in _IMAGE_PARAMETERS
                    else {"use_column_width": True})


def _full_width(widget):
    """Stretch a widget to its column: width="stretch" where Streamlit has the new width argument
    (use_container_width is deprecated there and is to be removed), use_container_width on 1.32."""
    width = inspect.signature(widget).parameters.get("width")
    return {"width": "stretch"} if width is not None and isinstance(width.default, str) else {"use_container_width": True}


FULL_WIDTH = {name: _full_width(getattr(st, name)) for name in ("button", "download_button", "popover", "plotly_chart")}

# ===== 言語選択 / Language selection =====
_lang_option = st.sidebar.radio(
    "🌐 Language / 言語", list(LANGUAGES),
    horizontal=True, key="lang_radio",
)
_lang_code = LANGUAGES.get(_lang_option, "ja")
T = {**TRANSLATIONS["en"], **TRANSLATIONS[_lang_code]}


def ui(ja, en):
    """Inline screen text in Japanese or English."""
    return ja if _lang_code == "ja" else en


def _widget_default(key, default, checkbox=False, number=False):
    # A restored value has one owner: session state. Passing a second widget
    # default causes Streamlit's duplicate-default warning after project loads.
    if st.session_state.get(key) is None:
        st.session_state[key] = default
    return False if checkbox else "min" if number else None


# ===== パスワード認証 =====
def check_password():
    if LOCAL_MODE:
        return True
    if st.session_state.get("authenticated"):
        return True
    pwd = st.secrets.get("password", "")
    st.title("🌈 UV-Vis-NIR Maker")
    entered = st.text_input(T["password_prompt"], type="password")
    if st.button(T["login"]):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error(T["wrong_password"])
    return False


if not check_password():
    st.stop()

st.title("UV-Vis-NIR Maker")
st.caption(T["app_caption"])
if LOCAL_MODE:
    st.caption(T["local_mode"])


# ===== LoadedSpectrum =====
class LoadedSpectrum:
    """One spectrum of a stored file (a file can hold several)."""
    def __init__(self, name: str, source: str, spectrum):
        self.name = name
        self.source = source
        self.spectrum = spectrum


def mpl_to_plotly(text: str) -> str:
    r"""Label markup (<i>…</i>, _{…}, ^{…}, and $\it{x}$ mathtext) → Plotly HTML."""
    return to_html(text)


# ===== ラベル入力（書式ボタン付き） =====
def label_input(key: str, default: str = "") -> str:
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    if inp_key not in st.session_state:
        st.session_state[inp_key] = st.session_state[val_key]
    inp = st.text_input(T["label"], key=inp_key)
    st.session_state[val_key] = inp

    c1, c2, c3 = st.columns(3)
    if c1.button(T["add_italic"], key=f"_bi_{key}", **FULL_WIDTH["button"], help=T["italic_help"]):
        st.session_state[val_key] = inp + "<i>TEXT</i>"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button(T["add_subscript"], key=f"_bs_{key}", **FULL_WIDTH["button"], help=T["subscript_help"]):
        st.session_state[val_key] = inp + "_{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button(T["add_superscript"], key=f"_bp_{key}", **FULL_WIDTH["button"], help=T["superscript_help"]):
        st.session_state[val_key] = inp + "^{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    return st.session_state[val_key]


# ===== 色（グラフ描画appと同じ色の組み合わせ） =====
OTHER_COLOR = "—"
SWATCH_SETS = OrderedDict([(name, None) for name in PALETTES])


def swatch_set_label(key):
    return PALETTE_NAMES[_lang_code][key]


def named_colors():
    return load_colors(COLORS_FILE)


def _int_or(value, default):
    return value if type(value) is int else default


def spectrum_display_order():
    """Spectrum indices in display order (the 表示順 numbers; equal numbers keep the file order)."""
    state = st.session_state
    n = len(state.get("_spectrum_names") or [])
    return sorted(range(n), key=lambda i: (_int_or(state.get(f"ord_{i}"), i + 1), i))


def palette_range(n):
    """First and last display positions (1-based, inclusive) that take the palette; the rest are the base colour.

    palette_to = 0 means "to the last spectrum", so spectra added later are included.
    """
    state = st.session_state
    lo = min(max(_int_or(state.get("palette_from"), 1), 1), max(n, 1))
    to = _int_or(state.get("palette_to"), 0)
    hi = n if to <= 0 else min(max(to, lo), n)
    return lo, hi


def palette_base_color():
    value = st.session_state.get("palette_base_color")
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else "#000000"


def palette_items():
    """Colour keys in display order: [(key, visible)]."""
    state = st.session_state
    return [(f"sp_color_{i}", state.get(f"vis_{i}", True) is not False) for i in spectrum_display_order()]


def palette_plan():
    """The palette colour for every spectrum key, and the number of spectra it colours.

    Visible spectra within the chosen range take the palette in display order (shades spread over
    them, as the graph app spreads them over its series); hidden ones continue after them. Spectra
    outside the range (a substrate or a reference, for example) take the base colour, black by default.
    """
    state = st.session_state
    name = state.get("palette", DEFAULT_PALETTE)
    name = name if name in PALETTES else DEFAULT_PALETTE
    reverse, steps = bool(state.get("palette_reverse", False)), _int_or(state.get("palette_steps"), 0)
    items = palette_items()
    lo, hi = palette_range(len(items))
    shaded = items[lo - 1:hi]
    visible = [key for key, shown in shaded if shown]
    hidden = [key for key, shown in shaded if not shown]
    plan = {key: palette_base_color() for key, _ in items[:lo - 1] + items[hi:]}
    plan.update(zip(visible, palette_sequence(name, len(visible), reverse, steps)))
    plan.update(zip(hidden, palette_sequence(name, len(shaded), reverse, steps)[len(visible):]))
    return plan, len(visible)


def _apply_palette():
    """Colour the visible spectra: the palette within the range, the base colour outside."""
    plan, _ = palette_plan()
    for key, shown in palette_items():
        if shown:
            st.session_state[key] = plan[key]
    for key in list(st.session_state):
        if str(key).startswith(("_palview_", "_cname_")):
            del st.session_state[key]


def _set_palette_range(n):
    lo, hi = st.session_state.get("_palette_range", (1, n))
    st.session_state["palette_from"] = int(lo)
    st.session_state["palette_to"] = 0 if int(hi) >= n else int(hi)
    _apply_palette()


def top_to_bottom(levels):
    """Positions ordered as the curves appear from top to bottom, for the legend.

    ``levels`` are the curves' vertical positions (stacking offsets) in display order;
    equal levels (overlaid curves) keep the display order.
    """
    return sorted(range(len(levels)), key=lambda k: -levels[k])


def _set_color(key, value):
    st.session_state[key] = value


def _named_color_chosen(key, name_key):
    value = named_colors().get(st.session_state.get(name_key))
    if value:
        st.session_state[key] = value


def swatch_set_colors(set_key, n_items):
    return swatch_colors(set_key, n_items, bool(st.session_state.get("palette_reverse", False)),
                         _int_or(st.session_state.get("palette_steps"), 0))


# ===== カラーポップアップ =====
def color_picker_popover(key: str, default_hex: str, n_items: int = 1):
    """Colour of one spectrum: swatches of the graph app's palettes, colours by name, or any colour."""
    _widget_default(key, default_hex)
    current = str(st.session_state[key])

    st.markdown(
        f'<div style="background:{current};height:20px;border-radius:5px;'
        f'border:1px solid #ccc;margin-bottom:4px"></div>',
        unsafe_allow_html=True,
    )
    with st.popover(T["change_color"], **FULL_WIDTH["popover"]):
        view_key = f"_palview_{key}"
        if st.session_state.get(view_key) not in SWATCH_SETS:
            st.session_state[view_key] = st.session_state.get("palette", DEFAULT_PALETTE)
        st.selectbox(ui("色の組み合わせ", "Palette"), list(SWATCH_SETS), key=view_key,
                     format_func=swatch_set_label,
                     help=ui("グラフ描画appと同じ組み合わせです。濃淡は、色を付けるスペクトルの数（または「濃淡の段数」）で分けた色で、"
                             "一番薄い色でも白地とのコントラストが 3:1 以上です。",
                             "The graph app's palettes. Shades are split by the number of coloured spectra "
                             "(or the shade steps); even the lightest has at least 3:1 contrast against white."))
        colors = swatch_set_colors(st.session_state[view_key], n_items)
        cols = st.columns(len(colors))
        for j, hex_c in enumerate(colors):
            with cols[j]:
                selected = hex_c.lower() == current.lower()
                st.markdown(
                    f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                    f'border:{"3px solid #333" if selected else "1px solid #ccc"};'
                    f'margin-bottom:2px"></div>',
                    unsafe_allow_html=True,
                )
                st.button("✓" if selected else " ", key=f"{key}_sw_{j}", help=hex_c.upper(),
                          **FULL_WIDTH["button"], on_click=_set_color, args=(key, hex_c))
        names = named_colors()
        name_key = f"_cname_{key}"
        st.session_state[name_key] = next((n for n, h in names.items() if h.lower() == current.lower()), OTHER_COLOR)
        st.selectbox(ui("名前で選ぶ", "By name"), [OTHER_COLOR] + list(names), key=name_key,
                     on_change=_named_color_chosen, args=(key, name_key),
                     format_func=lambda n: n if n == OTHER_COLOR else f"{color_emoji(names[n])} {n}  {names[n]}",
                     help=ui("colors.csv の色（グラフ描画appと同じ書式「名前,#RRGGBB」）。グラフ描画appの colors.csv をこのアプリのフォルダに置くと、同じ名前の色を選べます。",
                             "Colours from colors.csv (the graph app's format, name,#RRGGBB). Put the graph app's colors.csv in this app's folder to share its names."))
        st.color_picker(ui("自由に選ぶ", "Any colour"), key=key)
    return st.session_state[key]


# ===== セッション保存/復元 =====
# Keep conditional widget values when their controls are temporarily hidden.
for _key in list(st.session_state):
    if _key != "lang_radio" and (_key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(_key)):
        st.session_state[_key] = st.session_state[_key]


def build_session_json() -> bytes:
    settings = {key: value for key, value in st.session_state.items()
                if key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key)}
    return encode_project(st.session_state.get("_spectra_bytes", {}), settings)


def restore_session(json_bytes: bytes):
    # This function is called only from callbacks, before widgets are created.
    project = decode_project(json_bytes)
    for key in list(st.session_state):
        # The screen language is the person's own choice: opening someone's project keeps it.
        if key != "lang_radio" and (key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key)
                                    or re.fullmatch(r"_(?:inp|ver)_lbl_.*|_(?:palview|cname)_.*|_auto_labels|_size_signature", key)):
            del st.session_state[key]
    st.session_state["_spectra_bytes"] = project["spectra_files"]
    st.session_state.pop("_upload_error", None)
    st.session_state.update({key: value for key, value in project["settings"].items() if key != "lang_radio"})
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_spectrum_names"] = None
    st.session_state["_project_loaded"] = True
    st.session_state.pop("_project_error", None)
    st.session_state.pop("_autosave_fingerprint", None)


def _open_uploaded_project():
    upload = st.session_state.get("_session_upload")
    if upload is not None:
        try:
            restore_session(upload.getvalue())
        except (ProjectError, ValueError) as exc:
            st.session_state["_project_error"] = str(exc)


def _open_local_autosave():
    try:
        restore_session((LOCAL_DATA_DIR / "autosave.json").read_bytes())
    except (OSError, ProjectError) as exc:
        st.session_state["_project_error"] = str(exc)


def _open_style_template(raw: bytes | None = None):
    """Apply a style template: the uploaded file, or ``raw`` bytes given directly."""
    if raw is None:
        upload = st.session_state.get("_style_upload")
        if upload is None:
            return
        raw = upload.getvalue()
    try:
        settings = decode_style_template(raw)
        clean_settings({**{key: st.session_state[key] for key in STYLE_KEYS if key in st.session_state}, **settings})
        st.session_state.update(settings)
        st.session_state.pop("_size_signature", None)
        st.session_state.pop("_auto_labels", None)
        st.session_state["_style_loaded"] = True
        st.session_state.pop("_style_error", None)
    except (ProjectError, ValueError) as exc:
        st.session_state["_style_error"] = str(exc)


def _apply_figure_preset():
    name = st.session_state["figure_preset"]
    preset = FIGURE_PRESETS_MM.get(name)
    if preset:
        st.session_state["fig_width"], st.session_state["fig_height"] = map(mm_to_inches, preset)
        if name == "figapp":
            st.session_state.update(FIGAPP_STYLE)
        else:
            compact = name == "single"
            size = 9.0 if compact else 11.0
            st.session_state.update(font_size=size, tick_font_size=size, legend_font_size=size,
                label_fontsize=8 if compact else 10, annotation_font_size=6 if compact else 7, marks_fontsize=6 if compact else 7,
                margin_left=.18 if compact else .13, margin_bottom=.22 if compact else .15)
    st.session_state.pop("_size_signature", None)


def _update_figure_dimension(axis):
    value = st.session_state[f"_figure_{axis}"]
    st.session_state[f"fig_{axis}"] = mm_to_inches(value) if st.session_state["size_unit"] == "mm" else value
    st.session_state["figure_preset"] = "custom"


def _update_margin(key):
    value = st.session_state[f"_percent_{key}"] / 100
    st.session_state[key] = 1 - value if key in ("margin_right", "margin_top") else value


def _new_project():
    for key in list(st.session_state):
        if ((key in GLOBAL_KEYS and key != "lang_radio") or PATTERN_KEY.fullmatch(key)
                or re.fullmatch(r"_(?:inp|ver)_lbl_.*|_(?:palview|cname)_.*|_auto_labels|_size_signature", key)):
            del st.session_state[key]
    st.session_state["_spectra_bytes"] = {}
    st.session_state.pop("_upload_error", None)
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_spectrum_names"] = None
    st.session_state.pop("_project_loaded", None)
    st.session_state.pop("_project_error", None)


def _uploaded_files_changed(widget_key):
    files = st.session_state.get(widget_key) or []
    names = [file.name for file in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        st.session_state["_upload_error"] = ", ".join(duplicates)
        return
    st.session_state.pop("_upload_error", None)
    # Comparing the spectrum identities happens later, before their widgets.
    st.session_state["_spectra_bytes"] = {file.name: file.getvalue() for file in files}


SPECTRUM_KEY_TEMPLATES = ["vis_{}", "ord_{}", "_val_lbl_{}", "_ver_lbl_{}", "sp_color_{}", "ls_{}", "extra_offset_{}", "yoff_{}", "scale_{}"]


def _remap_file_session_state(old_names: list, new_names: list, key_tmpls: list):
    """ファイルセットが変更された時、位置インデックスベースのキーを名前基準で再マップする。"""
    old_vals: dict[str, dict] = {}
    for old_i, name in enumerate(old_names):
        old_vals[name] = {tmpl: st.session_state.get(tmpl.format(old_i)) for tmpl in key_tmpls}
    for old_i in range(len(old_names)):
        for tmpl in key_tmpls:
            st.session_state.pop(tmpl.format(old_i), None)
    for new_i, name in enumerate(new_names):
        if name in old_vals:
            for tmpl, val in old_vals[name].items():
                if val is None:
                    continue
                new_k = tmpl.format(new_i)
                st.session_state[new_k] = val + 1 if tmpl.startswith("_ver_") else val


def _update_yrange():
    """The manual y range from its two inputs (kept as the saved pair only while the minimum is below the maximum)."""
    low, high = float(st.session_state["_yrange_lo"]), float(st.session_state["_yrange_hi"])
    if low < high:
        st.session_state["yrange"] = (low, high)
        st.session_state["_yrange_signature"] = (low, high)


def _add_band_preset():
    """Append the UV / Vis / NIR bands to the marks (once); runs before the text area is drawn."""
    existing = st.session_state.get("marks_text", "") or ""
    if BAND_PRESET not in existing:
        st.session_state["marks_text"] = (existing.rstrip("\n") + "\n" + BAND_PRESET).lstrip("\n")


def _axis_unit_ticks(unit):
    """Tick intervals and peak-label decimals that suit an axis unit (nm → eV …); the nm range stays."""
    major, minor = UNIT_TICKS[unit]
    st.session_state["major_tick"], st.session_state["minor_tick"] = major, minor
    st.session_state["peak_decimals"] = 0 if unit == "nm" else 2


def _x_unit_changed():
    _axis_unit_ticks(st.session_state["x_unit"])


def _y_mode_changed():
    """A Tauc plot puts photon energy on the x axis: its ticks follow; leaving it, the chosen unit's ticks return."""
    unit = "eV" if st.session_state["y_mode"] == "tauc" else st.session_state.get("x_unit", "nm")
    _axis_unit_ticks(unit if unit in UNIT_TICKS else "nm")


def _input_defaults():
    """(x unit, (mode, percent) or None) to use for files whose header names neither."""
    unit = st.session_state.get("input_x_unit", "nm")
    unit = unit if unit in X_UNITS else "nm"
    choice = st.session_state.get("input_y_mode", "auto")
    modes = {"T%": ("T", True), "T": ("T", False), "A": ("A", False), "R%": ("R", True), "R": ("R", False), "KM": ("KM", False)}
    return unit, modes.get(choice)


def data_mode(spectrum):
    """(mode, percent) of a spectrum's stored values: the header's, or the input default when the header said nothing."""
    meta = spectrum.metadata
    _, default_mode = _input_defaults()
    if meta.get("y_mode_guessed") and default_mode is not None:
        return default_mode
    return meta.get("y_mode", "unknown"), bool(meta.get("y_percent", False))


# ===== サイドバー =====

st.sidebar.caption(f"Last updated: {LAST_UPDATED}")
st.sidebar.header(T["session_header"])
st.sidebar.caption(T["project_help"])
session_upload = st.sidebar.file_uploader(
    T["load_session"], type=["json"], key="_session_upload",
    on_change=_open_uploaded_project,
)
if st.session_state.pop("_project_loaded", False):
    st.sidebar.success(T["project_loaded"])
if st.session_state.get("_project_error"):
    st.sidebar.error(T["project_error"] + st.session_state["_project_error"])
if LOCAL_DATA_DIR and (LOCAL_DATA_DIR / "autosave.json").is_file() and not st.session_state.get("_spectra_bytes"):
    st.sidebar.button(T["resume_local"], on_click=_open_local_autosave, **FULL_WIDTH["button"])
st.sidebar.button(T["new_project"], on_click=_new_project, **FULL_WIDTH["button"])
autosave_status = st.sidebar.empty()

_file_epoch = st.session_state.get("_file_epoch", 0)
st.sidebar.header(T["data_header"])
st.sidebar.caption(T["native_help"])
_upload_key = f"spectra_upload_{_file_epoch}"
uploaded_files = st.sidebar.file_uploader(
    T["data_upload"], type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
    key=_upload_key, on_change=_uploaded_files_changed, args=(_upload_key,),
)
if st.session_state.get("_upload_error"):
    st.sidebar.error(T["duplicate_names"] + st.session_state["_upload_error"])
with st.sidebar.expander(ui("ヘッダーに記載がないときの読み方", "Reading files whose header says nothing")):
    st.selectbox(ui("横軸の単位（記載がないとき）", "Wavelength unit (when not stated)"), list(X_UNITS), key="input_x_unit",
                 format_func=lambda u: X_UNIT_TEXT[u],
                 help=ui("「Wavelength (nm)」「eV」などの見出しがあればそれを優先します。見出しのない 2 列のテキストはこの単位で読みます。",
                         "A header such as 'Wavelength (nm)' or 'eV' takes precedence. Two-column text without a header is read in this unit."))
    st.selectbox(ui("値の種類（記載がないとき）", "Kind of value (when not stated)"), list(INPUT_Y_MODES), key="input_y_mode",
                 format_func=lambda m: {"auto": ui("自動（値の範囲から判断）", "Automatic (from the values)"), "T%": "%T", "T": ui("T（0〜1）", "T (0–1)"),
                                        "A": ui("吸光度 Abs", "Absorbance"), "R%": "%R", "R": ui("R（0〜1）", "R (0–1)"), "KM": "F(R)"}[m],
                 help=ui("「%T」「Abs.」「%R」などの見出しがあればそれを優先します。見出しのない値は、0〜100 なら %T、0〜1 なら T、それ以外は吸光度と読みます。"
                         "見出しのない反射率データはここで %R を選んでください。",
                         "A header such as '%T', 'Abs.' or '%R' takes precedence. Values without one are read as %T (0–100), T (0–1) or absorbance (otherwise). "
                         "Choose %R here for reflectance data without a header."))
_default_x_unit, _default_y_mode = _input_defaults()

active = []
for _source_name, _source_bytes in st.session_state.get("_spectra_bytes", {}).items():
    try:
        _spectra = read_spectra(_source_bytes, _source_name, default_x_unit=_default_x_unit, default_y_mode=_default_y_mode)
        for _spectrum in _spectra:
            _name = _source_name if len(_spectra) == 1 else f"{_source_name} [{_spectrum.name}]"
            active.append(LoadedSpectrum(_name, _source_name, _spectrum))
    except UvVisParseError as _error:
        st.error(T["read_error"] + f"{_source_name}: {_error}")

_new_names = [item.name for item in active]
_old_names = st.session_state.get("_spectrum_names")
if _old_names is not None and _old_names != _new_names:
    _remap_file_session_state(_old_names, _new_names, SPECTRUM_KEY_TEMPLATES)
st.session_state["_spectrum_names"] = _new_names
if active:
    st.sidebar.caption(T["scan_count"].format(files=len(st.session_state.get("_spectra_bytes", {})), scans=len(active)))
if active and not uploaded_files:
    st.sidebar.caption(T["restored"] + ", ".join(sorted({item.source for item in active})))
_seen_sources = set()
for _item in active:
    if _item.source in _seen_sources:
        continue
    _seen_sources.add(_item.source)
    _summary = condition_summary(_item.spectrum)
    if _summary:
        st.sidebar.caption(f"{_item.source}: {_summary}")
    if _item.spectrum.metadata.get("y_mode_guessed") and _default_y_mode is None:
        _mode, _percent = data_mode(_item.spectrum)
        st.sidebar.caption(ui(f"{_item.source}: 値の種類の記載がないため {'%' if _percent else ''}{_mode} として読みました。",
                              f"{_item.source}: no header names the values; read as {'%' if _percent else ''}{_mode}."))

# ----- graph settings
st.sidebar.header(T["graph_settings"])
_data_min = min(float(item.spectrum.x.min()) for item in active) if active else 200.0
_data_max = max(float(item.spectrum.x.max()) for item in active) if active else 2500.0
_slider_lo = max(X_LIMITS[0], float(np.floor(_data_min / 100) * 100)) if active else 100.0
_slider_hi = min(X_LIMITS[1], float(np.ceil(_data_max / 100) * 100)) if active else 3300.0
if _slider_hi <= _slider_lo:
    _slider_hi = _slider_lo + 100.0
_full_range = (float(max(_slider_lo, np.floor(_data_min / 10) * 10)), float(min(_slider_hi, np.ceil(_data_max / 10) * 10)))
if active and st.session_state.get("xrange") is None:
    st.session_state["xrange"] = _full_range
if active and st.sidebar.button(T["data_range"], **FULL_WIDTH["button"]):
    st.session_state["xrange"] = _full_range
_current = st.session_state.get("xrange") or (200.0, 2500.0)
_current = (float(min(max(_current[0], _slider_lo), _slider_hi - 10)), float(min(max(_current[1], _slider_lo + 10), _slider_hi)))
if _current[0] >= _current[1]:
    _current = (_slider_lo, _slider_hi)
st.session_state["xrange"] = _current
xrange = st.sidebar.slider(T["xrange"], _slider_lo, _slider_hi, step=10.0, key="xrange")
x_min, x_max = float(xrange[0]), float(xrange[1])

_widget_default("y_mode", "as_measured")
y_mode = st.sidebar.selectbox(ui("縦軸の量", "Quantity on the y axis"), list(Y_MODES), key="y_mode", on_change=_y_mode_changed,
    format_func=lambda m: {"as_measured": ui("測定値のまま", "As measured"), "T": ui("透過率 T", "Transmittance T"),
                           "A": ui("吸光度 A = −log₁₀T", "Absorbance A = −log₁₀T"), "R": ui("反射率 R", "Reflectance R"),
                           "KM": ui("Kubelka–Munk F(R) = (1−R)²/2R", "Kubelka–Munk F(R) = (1−R)²/2R"),
                           "tauc": ui("Tauc プロット (αhν)^{1/n} vs hν", "Tauc plot (αhν)^{1/n} vs hν")}[m],
    help=ui("透過率と吸光度は互いに換算できます（T ↔ A）。反射率と F(R) も同様です（R ↔ F(R)）。透過率のデータを反射率にはできません。"
            "Tauc プロットでは横軸が光子エネルギー (eV) になり、α は吸光度（透過データ）または F(R)（反射データ）に比例するとして描きます。",
            "Transmittance and absorbance convert into each other (T ↔ A), as do reflectance and F(R). Transmittance data cannot become reflectance. "
            "The Tauc plot puts photon energy (eV) on the x axis; α is taken proportional to the absorbance (transmission data) or to F(R) (reflection data)."))
if y_mode in ("T", "R"):
    y_percent = st.sidebar.checkbox(ui("% で表示（外すと 0〜1）", "Show in % (otherwise 0–1)"), value=_widget_default("y_percent", True, checkbox=True), key="y_percent")
else:
    y_percent = bool(st.session_state.get("y_percent", True))
if y_mode == "tauc":
    _widget_default("tauc_transition", "direct")
    tauc_transition = st.sidebar.selectbox(ui("遷移の種類（n）", "Transition (n)"), list(TAUC_TRANSITIONS), key="tauc_transition",
        format_func=lambda t: {"direct": ui("直接許容 n = 1/2 → (αhν)²", "Direct allowed, n = 1/2 → (αhν)²"),
                               "indirect": ui("間接許容 n = 2 → (αhν)^{1/2}", "Indirect allowed, n = 2 → (αhν)^{1/2}"),
                               "direct_forbidden": ui("直接禁制 n = 3/2 → (αhν)^{2/3}", "Direct forbidden, n = 3/2 → (αhν)^{2/3}"),
                               "indirect_forbidden": ui("間接禁制 n = 3 → (αhν)^{1/3}", "Indirect forbidden, n = 3 → (αhν)^{1/3}")}[t])
    tauc_fit_on = st.sidebar.checkbox(ui("直線部を当てはめて E_g を求める", "Fit the linear part and extrapolate E_g"),
                                      value=_widget_default("tauc_fit", True, checkbox=True), key="tauc_fit")
    _e_lo, _e_hi = range_in_unit(x_min, x_max, "eV")
    _e_lo, _e_hi = max(ENERGY_LIMITS[0], float(np.floor(_e_lo * 20) / 20)), min(ENERGY_LIMITS[1], float(np.ceil(_e_hi * 20) / 20))
    _tr = st.session_state.get("tauc_range")
    if not (isinstance(_tr, (list, tuple)) and len(_tr) == 2 and _e_lo <= _tr[0] < _tr[1] <= _e_hi):
        _span = _e_hi - _e_lo
        st.session_state["tauc_range"] = (round(_e_lo + 0.45 * _span, 2), round(_e_lo + 0.7 * _span, 2))
    tauc_range = st.sidebar.slider(ui("当てはめる範囲 (eV)", "Fit range (eV)"), _e_lo, _e_hi, step=0.01, key="tauc_range",
                                   help=ui("吸収端の直線部分を選びます。直線を Y = 0 まで延長した交点が E_g です。",
                                           "Choose the linear part of the absorption edge; the line extrapolated to Y = 0 gives E_g."))
    show_eg = st.sidebar.checkbox(ui("図に E_g の値を書く", "Write E_g on the figure"), value=_widget_default("show_eg", True, checkbox=True), key="show_eg")
    x_unit = "eV"
    st.session_state.setdefault("x_unit", "nm")
else:
    tauc_transition, tauc_fit_on, tauc_range, show_eg = "direct", False, (2.0, 3.0), False
    _widget_default("x_unit", "nm")
    x_unit = st.sidebar.selectbox(ui("横軸の単位", "x-axis unit"), list(X_UNITS), key="x_unit", format_func=lambda u: X_UNIT_TEXT[u],
                                  on_change=_x_unit_changed,
                                  help=ui("波長範囲は nm で指定し、表示だけを換えます（eV = 1239.84 / λ[nm]、cm⁻¹ = 10⁷ / λ[nm]）。",
                                          "The range is set in nm; only the display changes (eV = 1239.84 / λ[nm], cm⁻¹ = 10⁷ / λ[nm])."))
x_lo, x_hi = range_in_unit(x_min, x_max, x_unit)

_widget_default("normalize_mode", "none")
normalize_mode = st.sidebar.selectbox(ui("正規化", "Normalisation"), list(NORMALIZE_MODES), key="normalize_mode",
    format_func=lambda v: {"none": ui("しない", "None"), "max": ui("最大値 = 1", "Maximum = 1"),
                           "window": ui("指定範囲の最大値 = 1", "Maximum within a window = 1"),
                           "minmax": ui("最小 0・最大 1", "Minimum 0, maximum 1"), "at": ui("指定波長の値 = 1", "Value at a wavelength = 1"),
                           "area": ui("面積 = 1", "Unit area")}[v],
    help=ui("縦軸の換算のあとに掛けます。吸収スペクトルの形を比べるときに使います。T_lum などの指標は正規化前の値から計算します。",
            "Applied after the y-axis conversion, for comparing the shapes of absorption spectra. T_lum and the other indices are computed before it."))
if normalize_mode == "window":
    _nw = st.session_state.get("norm_window")
    if not (isinstance(_nw, (list, tuple)) and len(_nw) == 2 and x_min <= _nw[0] < _nw[1] <= x_max):
        st.session_state["norm_window"] = (x_min, x_max)
    norm_window = st.sidebar.slider(ui("正規化の基準範囲 (nm)", "Normalisation window (nm)"), x_min, x_max, step=1.0, key="norm_window")
else:
    norm_window = None
if normalize_mode == "at":
    _na = st.session_state.get("norm_at")
    if not isinstance(_na, (int, float)) or not x_min <= _na <= x_max:
        st.session_state["norm_at"] = float(min(max(550.0, x_min), x_max))
    norm_at = st.sidebar.number_input(ui("基準にする波長 (nm)", "Reference wavelength (nm)"), x_min, x_max, step=1.0, key="norm_at")
else:
    norm_at = None
stack = st.sidebar.checkbox(ui("積み重ねる（自動オフセット）", "Stack spectra (automatic offsets)"),
                            value=_widget_default("stack", False, checkbox=True), key="stack",
                            help=ui("表示順に下から積みます。外すと重ね描きで、各スペクトルの Y位置と倍率を手で決められます。",
                                    "Stacks in display order from the bottom. Otherwise the spectra are overlaid, with a Y position and scale factor each."))
global_offset = st.sidebar.slider(ui("オフセット（曲線の高さの倍率）", "Offset (multiple of the curve height)"), 0.0, 3.0,
                                  _widget_default("global_offset", 1.0), step=0.05, key="global_offset") if stack else 0.0
if "show_legend" not in st.session_state:
    st.session_state["show_legend"] = True
show_legend = st.sidebar.checkbox(T["show_legend_cb"], key="show_legend")
_widget_default("legend_loc", "upper right")
legend_loc = st.sidebar.selectbox(ui("凡例の位置", "Legend position"), list(LEGEND_LOCATIONS), key="legend_loc",
    format_func=lambda v: {"best": ui("自動", "Automatic"), "upper right": ui("右上", "Upper right"), "upper left": ui("左上", "Upper left"),
                           "lower right": ui("右下", "Lower right"), "lower left": ui("左下", "Lower left"),
                           "center right": ui("右中央", "Centre right"), "center left": ui("左中央", "Centre left"),
                           "upper center": ui("上中央", "Upper centre"), "lower center": ui("下中央", "Lower centre"),
                           "outside right": ui("図の外（右）", "Outside, right")}[v]) if show_legend else "upper right"
if show_legend and legend_loc == "outside right":
    st.sidebar.caption(ui("右余白（「線・フォント・余白」）を広げると凡例が入ります。", "Widen the right margin (Lines, fonts and margins) to make room for the legend."))

st.sidebar.subheader(T["tick_settings"])
show_yticks = st.sidebar.checkbox(ui("縦軸の目盛りを表示", "Show y-axis ticks"), value=_widget_default("show_yticks", True, checkbox=True), key="show_yticks")
_yc1, _yc2 = st.sidebar.columns(2)
yrange_auto = _yc1.checkbox(ui("縦軸 自動", "y auto"), value=_widget_default("yrange_auto", True, checkbox=True), key="yrange_auto")
y_major_tick = _yc2.number_input(ui("縦軸 主目盛り（0=自動）", "y major tick (0=auto)"), min_value=0.0, max_value=1e6,
                                 value=_widget_default("y_major_tick", 0.0, number=True), step=1.0, format="%g", key="y_major_tick")
if not yrange_auto:
    _yr = st.session_state.get("yrange")
    if not (isinstance(_yr, (list, tuple)) and len(_yr) == 2 and _yr[0] < _yr[1]):
        st.session_state["yrange"] = (0.0, 100.0)
    if st.session_state.get("_yrange_signature") != tuple(st.session_state["yrange"]):
        st.session_state["_yrange_lo"], st.session_state["_yrange_hi"] = (float(v) for v in st.session_state["yrange"])
        st.session_state["_yrange_signature"] = tuple(st.session_state["yrange"])
    _yr1, _yr2 = st.sidebar.columns(2)
    _yr1.number_input(ui("縦軸 下端", "y minimum"), step=1.0, format="%g", key="_yrange_lo", on_change=_update_yrange)
    _yr2.number_input(ui("縦軸 上端", "y maximum"), step=1.0, format="%g", key="_yrange_hi", on_change=_update_yrange)
    if not st.session_state["_yrange_lo"] < st.session_state["_yrange_hi"]:
        st.sidebar.warning(ui("縦軸の下端は上端より小さくしてください。", "The y minimum must be below the y maximum."))
    yrange = tuple(st.session_state["yrange"])
else:
    yrange = None
_major_default, _minor_default = UNIT_TICKS[x_unit]
major_tick = st.sidebar.number_input(f"{T['major_tick']} ({X_UNIT_TEXT[x_unit]})", min_value=1e-3, max_value=1e6,
                                     value=_widget_default("major_tick", _major_default, number=True), step=_major_default / 2, format="%g", key="major_tick")
show_minor = st.sidebar.checkbox(T["show_minor_cb"], value=_widget_default("show_minor", True, checkbox=True), key="show_minor")
minor_tick = st.sidebar.number_input(f"{T['minor_tick']} ({X_UNIT_TEXT[x_unit]})", min_value=1e-4, max_value=1e6,
                                     value=_widget_default("minor_tick", _minor_default, number=True), step=_minor_default / 2, format="%g", key="minor_tick") if show_minor else None

# ----- axis labels: the defaults follow the unit and the quantity until the text is edited
st.sidebar.subheader(ui("軸のラベル", "Axis labels"))


def _data_y_label():
    """The y title for 'as measured': the kind of the first visible spectrum's values."""
    for i in spectrum_display_order():
        if i < len(active) and st.session_state.get(f"vis_{i}", True) is not False:
            mode, percent = data_mode(active[i].spectrum)
            if mode == "T":
                return "Transmittance (%)" if percent else "Transmittance"
            if mode == "R":
                return "Reflectance (%)" if percent else "Reflectance"
            if mode == "A":
                return "Absorbance"
            if mode == "KM":
                return Y_MODE_LABELS["KM"]
            return active[i].spectrum.metadata.get("y_unit") or "Intensity (a.u.)"
    return "Intensity (a.u.)"


if y_mode == "tauc":
    _auto_y = TAUC_LABELS[tauc_transition]
elif y_mode in ("T", "R"):
    _auto_y = Y_MODE_LABELS[y_mode] if y_percent else Y_MODE_LABELS[y_mode].split(" (")[0]
elif y_mode == "as_measured":
    _auto_y = _data_y_label()
else:
    _auto_y = Y_MODE_LABELS[y_mode]
_auto_x = X_UNIT_LABELS[x_unit]
_previous_auto = st.session_state.get("_auto_labels") or {}
for _axis, _auto in (("x", _auto_x), ("y", _auto_y)):
    _label_key = f"{_axis}_label"
    _current_label = st.session_state.get(_label_key)
    if _current_label is None or _current_label == _previous_auto.get(_axis):
        st.session_state[_label_key] = _auto
st.session_state["_auto_labels"] = {"x": _auto_x, "y": _auto_y}
_axis_label_help = ui(
    "グラフ描画appと同じ書き方です。文字を選んで Ctrl+I で斜体、Ctrl+B で太字、Ctrl+,（カンマ）で下付き、"
    "Ctrl+.（ピリオド）で上付き（Mac は ⌘ でも可）。入力欄では `<i>λ</i>`、`_{lum}`、`^{-1}` のような記号で出て、"
    "Enter で図に反映されます。単位や量を変えると、書き換えていないラベルは自動で追従します。空欄にするとラベルを出しません。",
    "Same markup as the graph app. Select text and press Ctrl+I for italic, Ctrl+B bold, Ctrl+, (comma) subscript, "
    "Ctrl+. (period) superscript (⌘ on a Mac). The box shows codes such as `<i>λ</i>`, `_{lum}`, `^{-1}`; press Enter "
    "to update the figure. A label you have not edited follows the unit and the quantity. Leave empty for no label.")
x_title = st.sidebar.text_input(ui("横軸", "x axis"), key="x_label", help=_axis_label_help)
y_title = st.sidebar.text_input(ui("縦軸", "y axis"), key="y_label", help=_axis_label_help)
st.sidebar.caption(ui("例：`Wavelength, <i>λ</i> (nm)`、`(<i>αhν</i>)^{2} (a.u.)`、`VO_{2}`。試料のラベルも同じ書き方です。",
                      "e.g. `Wavelength, <i>λ</i> (nm)`, `(<i>αhν</i>)^{2} (a.u.)`, `VO_{2}`. Sample labels use the same markup."))

# ----- processing
with st.sidebar.expander(ui("前処理（平滑化）", "Processing (smoothing)")):
    smooth_on = st.checkbox(ui("Savitzky–Golay 平滑化", "Savitzky–Golay smoothing"), value=_widget_default("smooth_on", False, checkbox=True), key="smooth_on",
                            help=ui("検出器・光源の切り替え波長付近のノイズを均すときに使います。換算の前に、測定値のまま掛けます。",
                                    "Evens out noise, e.g. around the detector / lamp change wavelengths. Applied to the stored values before any conversion."))
    _sc1, _sc2 = st.columns(2)
    smooth_window = _sc1.number_input(ui("窓（点）", "Window (points)"), 3, 101, value=_widget_default("smooth_window", 11, number=True), step=2, key="smooth_window")
    smooth_order = _sc2.number_input(ui("多項式の次数", "Polynomial order"), 1, 5, value=_widget_default("smooth_order", 3, number=True), key="smooth_order")

# ----- peaks
with st.sidebar.expander(ui("ピーク位置の表示", "Peak positions")):
    show_peaks = st.checkbox(ui("極大・極小の位置を表示", "Mark maxima / minima"), value=_widget_default("show_peaks", False, checkbox=True), key="show_peaks")
    _widget_default("peak_kind", "max")
    peak_kind = st.radio(ui("種類", "Kind"), ["max", "min"], horizontal=True, key="peak_kind",
                         format_func=lambda k: ui("極大（吸収ピークなど）", "Maxima (absorption peaks …)") if k == "max" else ui("極小（透過率の谷など）", "Minima (transmission dips …)"))
    peak_prom = st.slider(ui("感度（高さに対する突出の割合）", "Sensitivity (prominence relative to the height)"), 0.005, 0.9,
                          _widget_default("peak_prom", 0.05, number=False), step=0.005, key="peak_prom")
    peak_distance = st.number_input(ui("最小間隔 (nm)", "Minimum separation (nm)"), 0.0, 10000.0,
                                    value=_widget_default("peak_distance", 10.0, number=True), step=1.0, format="%g", key="peak_distance",
                                    help=ui("近いピークの一方だけを残す間隔。波長で数え、eV や cm⁻¹ の表示でも同じです。",
                                            "Peaks closer than this keep only the more prominent one; counted in wavelength whatever the axis unit."))
    _pk1, _pk2 = st.columns(2)
    peak_decimals = _pk1.number_input(ui("小数桁", "Decimals"), 0, 3, value=_widget_default("peak_decimals", 0 if x_unit == "nm" else 2, number=True), key="peak_decimals")
    peak_label_rotate = _pk2.checkbox(ui("縦書き", "Rotate labels"), value=_widget_default("peak_label_rotate", False, checkbox=True), key="peak_label_rotate")

# ----- marks and bands
with st.sidebar.expander(ui("注記の破線・帯（UV / Vis / NIR など）", "Guide lines and bands (UV / Vis / NIR …)")):
    _widget_default("marks_text", "")
    st.text_area(ui("1 行に 1 つ（横軸の単位で）", "One per line (in the x-axis unit)"), key="marks_text", height=120,
                 help=ui("破線：`550` や `550 <i>T</i>_{lum}`（値とラベル）。帯：`780-2500 NIR`（範囲とラベル）。`#` 以降はメモ。"
                         "値は図の横軸と同じ単位です（nm 表示なら nm、Tauc プロットなら eV）。",
                         "Guide line: `550` or `550 <i>T</i>_{lum}` (value and label). Band: `780-2500 NIR` (range and label). `#` starts a note. "
                         "Values are in the unit shown on the x axis (nm, or eV on a Tauc plot)."))
    st.button(ui("UV / Vis / NIR の帯を追加", "Add UV / Vis / NIR bands"), **FULL_WIDTH["button"], key="_band_preset", on_click=_add_band_preset)
    marks, bands = parse_marks(st.session_state.get("marks_text", ""))
    _mk1, _mk2 = st.columns(2)
    _widget_default("marks_style", "dashed")
    marks_style = _mk1.selectbox(ui("線種", "Line style"), ["dashed", "dotted", "solid", "dashdot"], key="marks_style",
                                 format_func=lambda v: {"dashed": ui("破線", "Dashed"), "dotted": ui("点線", "Dotted"), "solid": ui("実線", "Solid"), "dashdot": ui("一点鎖線", "Dash-dot")}[v])
    _widget_default("marks_color", "#555555")
    marks_color = _mk2.color_picker(ui("線の色", "Line colour"), key="marks_color")
    _mk3, _mk4 = st.columns(2)
    _widget_default("marks_label_mode", "top")
    marks_label_mode = _mk3.selectbox(ui("ラベル", "Labels"), ["top", "rotated", "none"], key="marks_label_mode",
                                      format_func=lambda v: {"top": ui("上に横書き", "Above, upright"), "rotated": ui("上に縦書き", "Above, rotated"), "none": ui("なし", "None")}[v])
    marks_fontsize = _mk4.number_input(ui("文字サイズ (pt)", "Font size (pt)"), 4, 20, value=_widget_default("marks_fontsize", 7, number=True), key="marks_fontsize")
    _mk5, _mk6 = st.columns(2)
    marks_show_value = _mk5.checkbox(ui("値も書く", "Show the value"), value=_widget_default("marks_show_value", True, checkbox=True), key="marks_show_value")
    marks_decimals = _mk6.number_input(ui("値の小数桁", "Decimals"), 0, 3, value=_widget_default("marks_decimals", 0, number=True), key="marks_decimals")
    _mk7, _mk8 = st.columns(2)
    _widget_default("bands_color", "#BBBBBB")
    bands_color = _mk7.color_picker(ui("帯の色", "Band colour"), key="bands_color")
    bands_alpha = _mk8.slider(ui("帯の濃さ", "Band opacity"), 0.0, 1.0, _widget_default("bands_alpha", 0.25), step=0.05, key="bands_alpha")
    bands_label = st.checkbox(ui("帯のラベルを表示", "Show band labels"), value=_widget_default("bands_label", True, checkbox=True), key="bands_label")

# ----- zoom inset
with st.sidebar.expander(ui("拡大図", "Zoom inset")):
    show_zoom = st.checkbox(ui("拡大図を表示", "Show zoom inset"), value=_widget_default("show_zoom", False, checkbox=True), key="show_zoom")
    _zr = st.session_state.get("zoom_range")
    if not (isinstance(_zr, (list, tuple)) and len(_zr) == 2 and x_min <= _zr[0] < _zr[1] <= x_max):
        st.session_state["zoom_range"] = (float(x_min + 0.3 * (x_max - x_min)), float(x_min + 0.5 * (x_max - x_min)))
    zoom_range = st.slider(ui("拡大する波長範囲 (nm)", "Inset wavelength range (nm)"), x_min, x_max, step=1.0, key="zoom_range")
    _widget_default("zoom_location", "upper left")
    zoom_location = st.selectbox(ui("拡大図の位置", "Inset position"), ["upper left", "upper right", "lower left", "lower right"],
        format_func=lambda value: {"upper left": ui("左上", "Upper left"), "upper right": ui("右上", "Upper right"), "lower left": ui("左下", "Lower left"), "lower right": ui("右下", "Lower right")}[value], key="zoom_location")
    annotation_font_size = st.slider(ui("注記の文字サイズ", "Annotation font size"), 4, 20, _widget_default("annotation_font_size", 7), key="annotation_font_size")

# ----- optical indices
with st.sidebar.expander(ui("光学指標（T_lum・T_sol・T_nir）", "Optical indices (T_lum, T_sol, T_nir)")):
    show_indices = st.checkbox(ui("図の下に表を出す", "Show the table under the figure"), value=_widget_default("show_indices", False, checkbox=True), key="show_indices")
    indices_decimals = st.number_input(ui("小数桁", "Decimals"), 0, 3, value=_widget_default("indices_decimals", 1, number=True), key="indices_decimals")
    st.caption(ui("T_lum：CIE 1931 の視感度 ȳ(λ) で重み付けした 380〜780 nm の平均。T_sol・T_vis・T_nir・T_uv：AM1.5G（ASTM G173-03）で重み付けした 300〜2500・380〜780・780〜2500・300〜380 nm の平均。"
                  "反射率のデータでは R の平均（顔料の NIR 日射反射率）になります。正規化やオフセットの前の値から計算します。",
                  "T_lum: 380–780 nm average weighted by the CIE 1931 photopic ȳ(λ). T_sol, T_vis, T_nir, T_uv: averages weighted by AM1.5G (ASTM G173-03) over 300–2500, 380–780, 780–2500 and 300–380 nm. "
                  "Reflectance data give the same averages of R (the NIR solar reflectance of pigments). Computed before normalisation and offsets."))

# ----- sample labels
_LR_OPTIONS = ["左", "右"]
_lr_fmt = lambda x: T["left"] if x == "左" else T["right"]
st.sidebar.subheader(T["sample_labels"])
show_side_labels = st.sidebar.checkbox(T["show_side_labels_cb"], value=_widget_default("show_side_labels", False, checkbox=True), key="show_side_labels")
if show_side_labels:
    _widget_default("label_side", "右")
    label_side = st.sidebar.radio(T["label_pos"], _LR_OPTIONS, format_func=_lr_fmt, horizontal=True, key="label_side")
    label_fontsize = st.sidebar.slider(T["label_fontsize"], 4, 28, _widget_default("label_fontsize", 9), key="label_fontsize")
    label_offset_x = st.sidebar.slider(T["label_offset_x"], -50.0, 100.0, _widget_default("label_offset_x", 2.0), step=0.5,
                                       help=T["label_offset_x_help"], key="label_offset_x")
    label_offset_y = st.sidebar.slider(T["label_offset_y"], -1.0, 2.0, _widget_default("label_offset_y", 0.5), step=0.05,
                                       help=T["label_offset_y_help"], key="label_offset_y")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = st.session_state.get("label_side", "右"), 9, 2.0, 0.5

# ----- figure size and export
st.sidebar.header(T["fig_size"])
if "figure_preset" not in st.session_state:
    # A new project starts in the graph app's look (88 × 70 mm); projects and styles bring their own values.
    st.session_state["figure_preset"] = "figapp"
    for _key, _value in FIGAPP_STYLE.items():
        st.session_state.setdefault(_key, _value)
for _key, _value in STYLE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
st.session_state.setdefault("fig_width", mm_to_inches(88.0))
st.session_state.setdefault("fig_height", mm_to_inches(70.0))
st.sidebar.selectbox(ui("論文図のひな形", "Publication figure preset"), list(FIGURE_PRESETS_MM),
    format_func=lambda key: {"custom": ui("自由指定", "Custom"),
        "single": ui("1段幅の例 · 85 × 70 mm", "Single column example · 85 × 70 mm"),
        "double": ui("2段幅の例 · 180 × 120 mm", "Double column example · 180 × 120 mm"),
        "figapp": ui("グラフ描画appと同じ体裁 · 88 × 70 mm", "Graph app look · 88 × 70 mm")}[key],
    key="figure_preset", on_change=_apply_figure_preset)
st.sidebar.caption(ui("ひな形は寸法・文字サイズ・余白を設定します。「グラフ描画appと同じ体裁」は Arial 7 pt・目盛り 6.5 pt・細い線で、グラフ描画app・XRD Maker・Raman Maker の図と並べられます。",
                      "Presets set dimensions, font sizes and margins. 'Graph app look' uses Arial 7 pt, 6.5 pt ticks and thin lines to sit next to the graph app's, XRD Maker's and Raman Maker's figures."))
size_unit = st.sidebar.radio(ui("寸法の単位", "Size unit"), ["mm", "inch"], horizontal=True, key="size_unit")
_signature = (size_unit, st.session_state["fig_width"], st.session_state["fig_height"])
if st.session_state.get("_size_signature") != _signature:
    for _axis in ("width", "height"):
        _value = st.session_state[f"fig_{_axis}"]
        st.session_state[f"_figure_{_axis}"] = inches_to_mm(_value) if size_unit == "mm" else _value
    st.session_state["_size_signature"] = _signature
for _axis, _label in (("width", ui("図の幅", "Figure width")), ("height", ui("図の高さ", "Figure height"))):
    st.sidebar.number_input(f"{_label} ({size_unit})", min_value=25.0 if size_unit == "mm" else 25 / 25.4,
        max_value=508.0 if size_unit == "mm" else 20.0, step=1.0 if size_unit == "mm" else 0.1,
        format="%.2f", key=f"_figure_{_axis}", on_change=_update_figure_dimension, args=(_axis,))
fig_width, fig_height = st.session_state["fig_width"], st.session_state["fig_height"]
dpi_export = st.sidebar.selectbox(ui("出力 DPI", "Output DPI"), [300, 600], index=0, key="dpi_export")
st.session_state["font_size"] = float(st.session_state.get("font_size") or 7.0)
for _key in ("tick_font_size", "legend_font_size"):
    st.session_state[_key] = float(st.session_state.get(_key) or st.session_state["font_size"])
st.sidebar.caption(ui("文字の大きさ (pt)", "Text sizes (pt)"))
_size_columns = st.sidebar.columns(3)
for _column, _key, _label in ((_size_columns[0], "font_size", ui("軸ラベル", "Axis labels")),
                              (_size_columns[1], "tick_font_size", ui("目盛り", "Ticks")),
                              (_size_columns[2], "legend_font_size", ui("凡例", "Legend"))):
    _column.number_input(_label, min_value=4.0, max_value=28.0, step=0.5, format="%.1f", key=_key)
font_size = st.session_state["font_size"]
tick_font_size = st.session_state["tick_font_size"]
legend_font_size = st.session_state["legend_font_size"]
with st.sidebar.expander(ui("線・フォント・余白", "Lines, fonts and margins")):
    st.selectbox(ui("フォント", "Font family"), list(FONT_FAMILIES), key="font_family",
                 help=ui("グラフ描画appと同じ選択肢です。斜体・下付き・上付きも同じ書体で描きます。",
                         "The graph app's choices. Italics, subscripts and superscripts use the same font."))
    _actual_font = resolve_font(st.session_state["font_family"])
    if _actual_font != st.session_state["font_family"]:
        if _actual_font in METRIC_COMPATIBLE.get(st.session_state["font_family"], ()):
            st.caption(ui("この環境には {requested} がないため、字幅が同じ {actual} で描きます。"
                          "SVG には {requested} と書き込むので、手元で開けば {requested} になります。",
                          "{requested} is not installed here, so {actual} (same character widths) is used. "
                          "SVG files name {requested}, so they open in it on your computer.")
                       .format(requested=st.session_state["font_family"], actual=_actual_font))
        else:
            st.caption(ui("この環境では {actual} を使用します。", "This environment uses {actual}.").format(actual=_actual_font))
    for _key, _label in (("line_width", ui("スペクトルの線幅 (pt)", "Spectrum line width (pt)")),
                        ("reference_line_width", ui("注記・当てはめ線の線幅 (pt)", "Guide / fit line width (pt)")),
                        ("axes_line_width", ui("軸・目盛りの線幅 (pt)", "Axes and tick width (pt)"))):
        st.number_input(_label, .1, 5., step=.1, key=_key)
    st.number_input(ui("目盛りの長さ (pt)", "Tick length (pt)"), .5, 10., step=.5, key="tick_length",
                    help=ui("主目盛りの長さ。副目盛りはその半分です。", "Major tick length; minor ticks are half as long."))
    st.selectbox(ui("目盛りの向き", "Tick direction"), ["auto", "in", "out"], key="tick_direction",
                 format_func=lambda value: {"auto": ui("内向き（初期値）", "Inward (default)"),
                                            "in": ui("内向き", "Inward"), "out": ui("外向き", "Outward")}[value])
    st.checkbox(ui("上と右の枠線を描く", "Draw the top and right frame"), key="axes_box")
    for _key, _label in (("margin_left", ui("左余白 (%)", "Left margin (%)")),
                        ("margin_right", ui("右余白 (%)", "Right margin (%)")),
                        ("margin_bottom", ui("下余白 (%)", "Bottom margin (%)")),
                        ("margin_top", ui("上余白 (%)", "Top margin (%)"))):
        _fraction = st.session_state[_key]
        st.session_state[f"_percent_{_key}"] = round(100 * (1 - _fraction if _key in ("margin_right", "margin_top") else _fraction), 8)
        st.number_input(_label, 0., 100., step=1., key=f"_percent_{_key}", on_change=_update_margin, args=(_key,))
    st.checkbox(ui("背景を透明にする", "Transparent background"), key="transparent_background")
    st.checkbox(ui("SVGの文字を編集可能にする", "Keep SVG text editable"), key="svg_text_editable")
    st.caption(ui("編集可能なSVGの表示には、開くPCにも使用フォントが必要です。", "Editable SVG text uses fonts installed on the computer opening it."))
plot_style = {key: st.session_state[key] for key in STYLE_KEYS}
try:
    plot_style = validate_style_settings(plot_style, fill_defaults=True)
except ValueError as _error:
    st.sidebar.error(ui("余白の合計が100%未満になるよう調整してください。", "Adjust opposing margins so their sum is below 100%."))
    st.stop()
_label_height = top_labels_height(marks, bands, x_range=(x_lo, x_hi), label_mode=marks_label_mode, show_value=marks_show_value,
                                  decimals=marks_decimals, show_band_labels=bands_label, plain=plain,
                                  axes_width_points=fig_width * 72 * (plot_style["margin_right"] - plot_style["margin_left"]),
                                  font_size=marks_fontsize)
if _label_height > 0:
    _needed_top = round(1 - (_label_height + 3) / (fig_height * 72), 3)
    if plot_style["margin_top"] > _needed_top + 0.002:
        st.sidebar.caption(ui(f"注記のラベルは図の上に出ます。上余白を {100 * (1 - _needed_top):.0f}% 以上にすると切れません。",
                              f"The guide labels sit above the axes; a top margin of {100 * (1 - _needed_top):.0f}% or more keeps them inside."))

with st.sidebar.expander(ui("作図スタイルの保存・読み込み", "Save and load figure styles")):
    st.caption(ui("寸法・線・フォント・ラベル位置を再利用できます。測定データや試料名は含みません。XRD Maker・Raman Maker のスタイルも開けます（共通の項目だけ取り込みます）。",
                  "Reuse dimensions, lines, fonts and label positions. Styles do not include measurements or sample names. XRD Maker and Raman Maker styles open too (shared settings only)."))
    st.file_uploader(ui("スタイルを開く (.json)", "Open figure style (.json)"), type=["json"], key="_style_upload", on_change=_open_style_template)
    if st.session_state.pop("_style_loaded", False):
        st.success(ui("作図スタイルを読み込みました。", "Figure style loaded."))
    if st.session_state.get("_style_error"):
        st.error(st.session_state["_style_error"])
    st.download_button(ui("スタイルを保存 (.json)", "Save figure style (.json)"),
        encode_style_template(dict(st.session_state)), file_name="uvvisnir_figure_style.json", mime="application/json", **FULL_WIDTH["download_button"])

st.sidebar.divider()


# ===== メインエリア =====

show_panel = st.toggle(T["show_panel"], value=_widget_default("show_panel", True, checkbox=True), key="show_panel")

if active:
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    orders, visibles, labels, colors_sel, line_styles, abs_offsets, scales = [], [], [], [], [], [], []

    for _key, _value in (("palette", DEFAULT_PALETTE), ("palette_reverse", False), ("palette_steps", 0),
                         ("palette_from", 1), ("palette_to", 0), ("palette_base_color", "#000000")):
        st.session_state.setdefault(_key, _value)
    if st.session_state["palette"] not in PALETTES:
        st.session_state["palette"] = DEFAULT_PALETTE
    palette_defaults, palette_count = palette_plan()

    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):

                st.markdown(ui("#### 色", "#### Colours"))
                st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key="palette",
                    format_func=lambda key: PALETTE_NAMES[_lang_code][key], on_change=_apply_palette,
                    help=ui("グラフ描画appと同じ色の組み合わせです。選ぶと、表示中のスペクトルに表示順で色が付きます。"
                            "濃淡と viridis は、一番薄い色でも白地とのコントラストが 3:1 以上（細い線の目安）になる濃さから始まります。",
                            "The graph app's palettes. Choosing one colours the visible spectra in display order. "
                            "Shades and viridis start at a colour with at least 3:1 contrast against white (the guideline for thin lines)."))
                if len(active) >= 2:
                    _n = len(active)
                    _lo, _hi = palette_range(_n)
                    st.session_state["_palette_range"] = (_lo, _hi)
                    st.slider(ui("色を付ける範囲（表示順）", "Coloured spectra (display order)"), min_value=1, max_value=_n,
                        step=1, key="_palette_range", on_change=_set_palette_range, args=(_n,),
                        help=ui("この範囲のスペクトルだけに色の組み合わせ（濃淡）を付けます。範囲外（基板や参照試料など）は「範囲外の色」（初期値は黒）になります。",
                                "Only these spectra take the palette (shades). Spectra outside the range, such as a substrate or a reference, take the outside colour (black by default)."))
                    _names = [plain(st.session_state.get(f"_val_lbl_{i}") or os.path.splitext(active[i].name)[0])
                              for i in spectrum_display_order()]
                    _outside = _names[:_lo - 1] + _names[_hi:]
                    _shaded = _names[_lo - 1:_hi]
                    st.caption(ui("色の組み合わせ：", "Palette: ") + (f"{_shaded[0]} 〜 {_shaded[-1]}" if len(_shaded) > 1 else _shaded[0])
                               + (ui("　／　範囲外：", "  /  outside: ") + "、".join(_outside) if _outside else ""))
                    if _outside:
                        st.color_picker(ui("範囲外の色", "Colour outside the range"), key="palette_base_color",
                                        on_change=_apply_palette,
                                        help=ui("範囲外のスペクトルの色です。", "Colour of the spectra outside the range."))
                st.checkbox(ui("濃淡（色の順）を逆に", "Reverse shades"), key="palette_reverse", on_change=_apply_palette,
                    help=ui("「…の濃淡」では 1 つ目のスペクトルが濃い色になります。ほかの組み合わせでは色の順が逆になります。",
                            "With shades, the first spectrum gets the dark one; other palettes reverse the order."))
                _pc1, _pc2 = st.columns([0.45, 0.55])
                _pc1.number_input(ui("濃淡の段数", "Shade steps"), min_value=0, max_value=MAX_SHADE_STEPS, step=1,
                    key="palette_steps", on_change=_apply_palette,
                    help=ui("0 はスペクトルの数で分けます。ほかの図と系列の数が違うときは、その図の系列の数を入れると同じ濃さになります（濃淡と viridis）。",
                            "0 splits the shades by the number of spectra. Enter the number of series of another figure to get its shades (shades and viridis)."))
                _pc2.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                _pc2.button(ui("色を当てはめ直す", "Re-apply"), on_click=_apply_palette, **FULL_WIDTH["button"],
                    help=ui("スペクトルごとに変えた色を戻すときや、表示順・表示を変えたあとに押します。",
                            "Restores the palette after changing single colours, the display order or visibility."))

                st.markdown(T["spectra_md"])
                for i, item in enumerate(active):
                    default_name = os.path.splitext(item.name)[0] if item.name == item.source else item.spectrum.name
                    default_hex = palette_defaults.get(f"sp_color_{i}", REFERENCE_COLORS[0])

                    with st.expander(f"**{i + 1}. {default_name}**", expanded=True):
                        order = st.number_input(T["display_order"], value=_widget_default(f"ord_{i}", i + 1, number=True), min_value=1, max_value=10000, key=f"ord_{i}")
                        visible = st.checkbox(T["show_cb"], value=_widget_default(f"vis_{i}", True, checkbox=True), key=f"vis_{i}")
                        label = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"sp_color_{i}", default_hex, palette_count)
                        _widget_default(f"ls_{i}", "solid")
                        line_style = st.selectbox(T["line_style"], list(LINE_STYLES), key=f"ls_{i}",
                                                  format_func=lambda v: {"solid": ui("実線", "Solid"), "dashed": ui("破線", "Dashed"), "dotted": ui("点線", "Dotted"), "dashdot": ui("一点鎖線", "Dash-dot")}[v])
                        if stack:
                            eoff_key = f"extra_offset_{i}"
                            if eoff_key not in st.session_state:
                                st.session_state[eoff_key] = 0.0
                            st.slider(T["offset_adj"], min_value=-5.0, max_value=15.0, step=0.05, key=eoff_key)
                            abs_offsets.append(0.0)
                            scales.append(1.0)
                        else:
                            _o1, _o2 = st.columns(2)
                            yoff = _o1.number_input(T["y_position"], value=_widget_default(f"yoff_{i}", 0.0, number=True), step=1.0, format="%g", key=f"yoff_{i}")
                            scale = _o2.number_input(T["scale"], min_value=0.001, max_value=1e6, value=_widget_default(f"scale_{i}", 1.0, number=True), step=0.1, format="%g", key=f"scale_{i}")
                            abs_offsets.append(float(yoff))
                            scales.append(float(scale))

                    orders.append(order)
                    visibles.append(visible)
                    labels.append(label)
                    colors_sel.append(chosen_color)
                    line_styles.append(line_style)
    else:
        for i, item in enumerate(active):
            key = f"sp_color_{i}"
            default_name = os.path.splitext(item.name)[0] if item.name == item.source else item.spectrum.name
            orders.append(st.session_state.get(f"ord_{i}", i + 1))
            visibles.append(st.session_state.get(f"vis_{i}", True))
            labels.append(st.session_state.get(f"_val_lbl_{i}", default_name))
            colors_sel.append(st.session_state.get(key, palette_defaults.get(key, REFERENCE_COLORS[0])))
            line_styles.append(st.session_state.get(f"ls_{i}", "solid"))
            if stack:
                st.session_state.setdefault(f"extra_offset_{i}", 0.0)
                abs_offsets.append(0.0)
                scales.append(1.0)
            else:
                abs_offsets.append(float(st.session_state.get(f"yoff_{i}", 0.0)))
                scales.append(float(st.session_state.get(f"scale_{i}", 1.0)))
    sort_idx = sorted(range(len(active)), key=lambda i: orders[i])

    # ===== 曲線の準備（切り出し → 平滑化 → 換算 → 正規化 → 配置） =====
    def convertible(mode, target):
        """Whether stored values of kind ``mode`` can be shown as ``target``."""
        if target == "as_measured":
            return True
        if target == "tauc":
            return mode in ("T", "A", "R", "KM")
        if target in ("T", "A"):
            return mode in ("T", "A")
        return mode in ("R", "KM")

    def prepare_curves():
        """[(index, x_display, y_plot, baseline, span, fit)] in display order, plus the labels of spectra that cannot be converted."""
        curves, skipped = [], []
        cumulative = 0.0
        for i in sort_idx:
            if not visibles[i]:
                continue
            spectrum = active[i].spectrum
            mode, percent = data_mode(spectrum)
            if not convertible(mode, y_mode):
                skipped.append((plain(labels[i]), mode))
                continue
            x_nm, y = spectrum.x, spectrum.y.astype(float)
            if smooth_on:
                y = smooth(y, int(smooth_window), int(smooth_order))
            mask = (x_nm >= x_min) & (x_nm <= x_max)
            x_nm, y = x_nm[mask], y[mask]
            if len(x_nm) < 2:
                continue
            if y_mode == "as_measured":
                x_disp, y_conv = x_from_nm(x_nm, x_unit), y
            elif y_mode == "tauc":
                x_disp, y_conv = tauc_transform(x_nm, y, mode, percent, tauc_transition)
                x_nm = EV_NM / x_disp
            else:
                x_disp, y_conv = x_from_nm(x_nm, x_unit), convert_y(y, mode, percent, y_mode, y_percent)
            order = np.argsort(x_disp, kind="stable")
            x_disp, y_conv, x_nm = x_disp[order], y_conv[order], x_nm[order]
            finite = np.isfinite(y_conv)
            x_disp, y_conv, x_nm = x_disp[finite], y_conv[finite], x_nm[finite]
            if len(x_disp) < 2:
                continue
            y_norm = normalize(x_nm, y_conv, normalize_mode, norm_window, norm_at)
            fit = None
            if y_mode == "tauc" and tauc_fit_on:
                fit = tauc_fit(x_disp, y_norm, float(tauc_range[0]), float(tauc_range[1]))
            y_low, y_high = float(np.min(y_norm)), float(np.max(y_norm))
            span = max(y_high - y_low, 1e-12)
            if stack:
                baseline = cumulative + float(st.session_state.get(f"extra_offset_{i}", 0.0)) * span - y_low
                y_plot = y_norm + baseline
                cumulative += global_offset * span
            else:
                baseline = abs_offsets[i]
                y_plot = y_norm * scales[i] + baseline
                span *= scales[i]
                y_low *= scales[i]
            curves.append({"index": i, "x": x_disp, "x_nm": x_nm, "y": y_plot, "baseline": baseline, "span": span, "y_low": y_low,
                           "fit": fit, "scale": 1.0 if stack else scales[i]})
        return curves, skipped

    curves, skipped = prepare_curves()
    unit_text = X_UNIT_TEXT[x_unit]

    def legend_order(curves):
        """Stacked spectra from top to bottom; overlaid spectra (equal baselines) keep the display order."""
        return top_to_bottom([c["baseline"] for c in curves])

    def label_anchor(curve):
        """(x, y) of a sample label in display coordinates."""
        frac = label_offset_x / 100.0
        x_pos = x_lo + frac * (x_hi - x_lo) if label_side == "左" else x_hi - frac * (x_hi - x_lo)
        y_pos = curve["baseline"] + curve["y_low"] + label_offset_y * curve["span"]
        return x_pos, y_pos

    def fit_line(curve):
        """(x0, x1, y0, y1) of the extrapolated Tauc line, in plot coordinates, from E_g on the baseline to a little past the fit range."""
        fit = curve["fit"]
        if not fit:
            return None
        scale = curve["scale"]
        x0 = fit["eg"]
        x1 = min(x_hi, fit["e_high"] + 0.15 * (fit["e_high"] - fit["e_low"]))
        if x1 <= x0:
            return None
        y0 = curve["baseline"]
        y1 = curve["baseline"] + scale * (fit["slope"] * x1 + fit["intercept"])
        return x0, x1, y0, y1

    def eg_label_anchor(segment):
        """Where the E_g text goes: right of the line's upper end (under the rising curve), or left of the intercept near the right edge."""
        x0, x1, y0, y1 = segment
        if x1 < x_lo + 0.8 * (x_hi - x_lo):
            return x1, y1, "left", "center", (4, 0)
        return x0, y0, "right", "bottom", (-3, 3)

    def mpl_legend_options():
        if legend_loc == "outside right":
            return dict(loc="center left", bbox_to_anchor=(1.01, 0.5))
        return dict(loc=legend_loc)

    # ===== 図の生成 =====
    @plt.rc_context()
    def build_figure(show_legend=show_legend):
        from matplotlib.ticker import MultipleLocator

        plt.rcParams.update({**font_rcparams(plot_style), "font.size": font_size, "axes.labelsize": font_size,
                             "xtick.labelsize": tick_font_size, "ytick.labelsize": tick_font_size,
                             "legend.fontsize": legend_font_size, "legend.title_fontsize": legend_font_size})
        major_len, minor_len = plot_style["tick_length"], plot_style["tick_length"] / 2
        direction = "in" if plot_style["tick_direction"] == "auto" else plot_style["tick_direction"]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.tick_params(which="major", axis="both", length=major_len, direction=direction)
        if show_minor and minor_tick:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax.tick_params(which="minor", axis="both", length=minor_len, direction=direction)
        if not plot_style["axes_box"]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        prepared, legend_lines = [], []
        for curve in curves:
            i = curve["index"]
            line, = ax.plot(curve["x"], curve["y"], color=colors_sel[i], linewidth=plot_style["line_width"],
                            linestyle=MPL_LINESTYLE[line_styles[i]], label=rich(labels[i]))
            legend_lines.append(line)
            prepared.append(PreparedCurve(rich(labels[i]), curve["x"], curve["y"], colors_sel[i], plot_style["line_width"]))
            if show_peaks:
                pks = detect_peaks(curve["x_nm"], curve["y"], peak_prom, float(peak_distance), peak_kind)
                if len(pks):
                    ax.plot(curve["x"][pks], curve["y"][pks], "v" if peak_kind == "max" else "^", color=colors_sel[i], markersize=4, linestyle="none")
                    for pk in pks:
                        ax.annotate(f"{curve['x'][pk]:.{int(peak_decimals)}f}", xy=(curve["x"][pk], curve["y"][pk]),
                                    xytext=(0, 6 if peak_kind == "max" else -6), textcoords="offset points",
                                    ha="center", va="bottom" if peak_kind == "max" else "top", fontsize=annotation_font_size,
                                    color=colors_sel[i], rotation=90 if peak_label_rotate else 0)
            segment = fit_line(curve)
            if segment:
                x0, x1, y0, y1 = segment
                ax.plot([x0, x1], [y0, y1], color=colors_sel[i], linewidth=plot_style["reference_line_width"], linestyle="--", zorder=4)
                if show_eg:
                    ax_x, ax_y, ha, va, shift = eg_label_anchor(segment)
                    ax.annotate(f"$E_\\mathrm{{g}}$ = {curve['fit']['eg']:.2f} eV", xy=(ax_x, ax_y), xytext=shift, textcoords="offset points",
                                ha=ha, va=va, fontsize=annotation_font_size, color=colors_sel[i],
                                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        if show_side_labels:
            for curve in curves:
                i = curve["index"]
                x_pos, y_pos = label_anchor(curve)
                ax.text(x_pos, y_pos, rich(labels[i]), color=colors_sel[i], fontsize=label_fontsize,
                        ha="left" if label_side == "左" else "right", va="center",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

        ax.set_xlim(x_lo, x_hi)
        if yrange is not None:
            ax.set_ylim(yrange)
        elif curves:
            ys = np.concatenate([c["y"] for c in curves])
            low, high = float(np.min(ys)), float(np.max(ys))
            span = high - low if high > low else max(abs(high), 1.0)
            # Room for the peak labels above the maxima (or below the minima).
            pad_top = 0.12 if show_peaks and peak_kind == "max" else 0.04
            pad_bottom = 0.12 if show_peaks and peak_kind == "min" else 0.04
            ax.set_ylim(low - pad_bottom * span, high + pad_top * span)
        if y_major_tick and y_major_tick > 0:
            ax.yaxis.set_major_locator(MultipleLocator(y_major_tick))
        if not show_yticks:
            ax.set_yticks([])
        ax.set_xlabel(rich(x_title))
        ax.set_ylabel(rich(y_title))
        add_mpl_bands(ax, bands, color=bands_color, alpha=bands_alpha, show_label=bands_label, font_size=marks_fontsize, rich=rich, x_range=(x_lo, x_hi))
        add_mpl_marks(ax, marks, color=marks_color, linestyle=MPL_LINESTYLE[marks_style], linewidth=plot_style["reference_line_width"],
                      label_mode=marks_label_mode, show_value=marks_show_value, decimals=marks_decimals, font_size=marks_fontsize,
                      rich=rich, x_range=(x_lo, x_hi))
        if show_legend and legend_lines:
            ordered = [legend_lines[k] for k in legend_order(curves)]
            ordered = [h for h in ordered if h.get_label() and not h.get_label().startswith("_")]
            if ordered:
                ax.legend(handles=ordered, frameon=False, **mpl_legend_options())
        apply_mpl_style(fig, plot_style)
        if show_zoom and y_mode != "tauc" and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_mpl_zoom_inset(ax, prepared, range_in_unit(zoom_range[0], zoom_range[1], x_unit), location=zoom_location,
                               font_size=annotation_font_size, unit_text=unit_text)
        return fig

    # ===== インタラクティブ Plotly プレビュー =====
    def build_plotly_figure(show_legend=show_legend):
        pfig = go.Figure()
        prepared, legend_traces = [], []
        family = css_font_stack(plot_style["font_family"])
        for curve in curves:
            i = curve["index"]
            pfig.add_trace(go.Scatter(x=curve["x"], y=curve["y"], name=mpl_to_plotly(labels[i]),
                                      line=dict(color=colors_sel[i], width=plot_style["line_width"] * 4 / 3, dash=PLOTLY_DASH[line_styles[i]]),
                                      mode="lines", showlegend=show_legend,
                                      hovertemplate=plain(labels[i]) + "<br>x=%{x:.4g} " + unit_text + "<br>y=%{y:.4g}<extra></extra>"))
            legend_traces.append(len(pfig.data) - 1)
            prepared.append(PreparedCurve(mpl_to_plotly(labels[i]), curve["x"], curve["y"], colors_sel[i], plot_style["line_width"]))
            if show_peaks:
                pks = detect_peaks(curve["x_nm"], curve["y"], peak_prom, float(peak_distance), peak_kind)
                if len(pks):
                    pfig.add_trace(go.Scatter(x=curve["x"][pks], y=curve["y"][pks], mode="markers+text",
                                              marker=dict(color=colors_sel[i], symbol="triangle-down" if peak_kind == "max" else "triangle-up", size=7),
                                              text=[f"{curve['x'][pk]:.{int(peak_decimals)}f}" for pk in pks],
                                              textposition="top center" if peak_kind == "max" else "bottom center",
                                              textfont=dict(size=annotation_font_size, color=colors_sel[i], family=family), showlegend=False, hoverinfo="skip"))
            segment = fit_line(curve)
            if segment:
                x0, x1, y0, y1 = segment
                pfig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=colors_sel[i], width=plot_style["reference_line_width"] * 4 / 3, dash="dash"),
                                          showlegend=False, hovertemplate=f"E<sub>g</sub> = {curve['fit']['eg']:.3f} eV<extra></extra>"))
                if show_eg:
                    ax_x, ax_y, ha, va, shift = eg_label_anchor(segment)
                    pfig.add_annotation(x=ax_x, y=ax_y, text=f"<i>E</i><sub>g</sub> = {curve['fit']['eg']:.2f} eV", showarrow=False,
                                        xanchor=ha, yanchor={"center": "middle", "bottom": "bottom"}[va], xshift=shift[0], yshift=shift[1],
                                        bgcolor="rgba(255,255,255,0.7)", font=dict(size=annotation_font_size, color=colors_sel[i], family=family))
            if show_side_labels:
                x_pos, y_pos = label_anchor(curve)
                pfig.add_trace(go.Scatter(x=[x_pos], y=[y_pos], mode="text", text=[mpl_to_plotly(labels[i])],
                                          textposition="middle right" if label_side == "左" else "middle left",
                                          textfont=dict(color=colors_sel[i], size=label_fontsize, family=family), showlegend=False, hoverinfo="skip"))
        for rank, k in enumerate(legend_order(curves), start=1):
            pfig.data[legend_traces[k]].legendrank = rank

        add_plotly_bands(pfig, bands, color=bands_color, alpha=bands_alpha, show_label=bands_label, font_size=marks_fontsize,
                         to_html=mpl_to_plotly, x_range=(x_lo, x_hi), font_family=family)
        add_plotly_marks(pfig, marks, color=marks_color, dash=PLOTLY_DASH[marks_style], width=plot_style["reference_line_width"],
                         label_mode=marks_label_mode, show_value=marks_show_value, decimals=marks_decimals, font_size=marks_fontsize,
                         to_html=mpl_to_plotly, x_range=(x_lo, x_hi), font_family=family)
        tick_px = plot_style["tick_length"] * 4 / 3
        ticks = {"in": "inside", "out": "outside"}.get(plot_style["tick_direction"], "inside")
        minor_cfg = dict(ticks=ticks, dtick=minor_tick, ticklen=tick_px / 2, showgrid=False) if (show_minor and minor_tick) else {}
        pfig.update_xaxes(range=[x_lo, x_hi], dtick=major_tick, ticks=ticks, ticklen=tick_px, showline=True, linecolor="black",
                          mirror=plot_style["axes_box"], showgrid=False, zeroline=False, tickfont=dict(color="black", size=tick_font_size),
                          title_font=dict(color="black", size=font_size), minor=minor_cfg, title_text=mpl_to_plotly(x_title),
                          linewidth=plot_style["axes_line_width"] * 4 / 3)
        yaxis = dict(showticklabels=show_yticks, ticks=ticks if show_yticks else "", ticklen=tick_px, showline=True, linecolor="black",
                     mirror=plot_style["axes_box"], showgrid=False, zeroline=False, tickfont=dict(color="black", size=tick_font_size),
                     title_font=dict(color="black", size=font_size), title_text=mpl_to_plotly(y_title), linewidth=plot_style["axes_line_width"] * 4 / 3)
        if yrange is not None:
            yaxis.update(range=list(yrange), autorange=False)
        if y_major_tick and y_major_tick > 0:
            yaxis.update(dtick=y_major_tick)
        pfig.update_yaxes(**yaxis)
        legend_positions = {"upper right": (1.0, 1.0, "right", "top"), "upper left": (0.0, 1.0, "left", "top"),
                            "lower right": (1.0, 0.0, "right", "bottom"), "lower left": (0.0, 0.0, "left", "bottom"),
                            "center right": (1.0, 0.5, "right", "middle"), "center left": (0.0, 0.5, "left", "middle"),
                            "upper center": (0.5, 1.0, "center", "top"), "lower center": (0.5, 0.0, "center", "bottom"),
                            "outside right": (1.02, 1.0, "left", "top"), "best": (1.0, 1.0, "right", "top")}
        lx, ly, lxa, lya = legend_positions[legend_loc]
        pfig.update_layout(
            dragmode="zoom",
            font=dict(family=family, size=font_size, color="black"),
            height=max(320, min(1800, int(fig_height * 96))),
            plot_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            paper_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            showlegend=show_legend,
            legend=dict(x=lx, y=ly, xanchor=lxa, yanchor=lya, font=dict(color="black", size=legend_font_size)),
            margin=dict(l=int(fig_width * 96 * plot_style["margin_left"]),
                        r=int(fig_width * 96 * (1 - plot_style["margin_right"])),
                        t=int(fig_height * 96 * (1 - plot_style["margin_top"])),
                        b=int(fig_height * 96 * plot_style["margin_bottom"])),
        )
        if show_zoom and y_mode != "tauc" and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_plotly_zoom_inset(pfig, prepared, range_in_unit(zoom_range[0], zoom_range[1], x_unit), location=zoom_location,
                                  font_size=annotation_font_size, unit_text=unit_text, hover_unit=unit_text)
        return pfig

    # ===== 光学指標 =====
    def indices_table():
        rows = []
        for i in sort_idx:
            if not visibles[i]:
                continue
            spectrum = active[i].spectrum
            mode, percent = data_mode(spectrum)
            if mode not in ("T", "A", "R", "KM"):
                continue
            result = optical_indices(spectrum.x, spectrum.y, mode, percent)
            quantity = result["quantity"]
            row = {ui("スペクトル", "Spectrum"): plain(labels[i]), ui("量", "Quantity"): quantity}
            for name in WEIGHTED_BANDS:
                value, coverage = result["values"][name], result["coverage"][name]
                text = "—" if not np.isfinite(value) else f"{100 * value:.{int(indices_decimals)}f}" + ("" if coverage >= 0.98 else " *")
                row[f"{quantity}_{name} (%)"] = text
            rows.append(row)
        return rows

    def markdown_table(rows):
        """A small table as Markdown (st.dataframe needs pyarrow, which not every PC has)."""
        if not rows:
            return ""
        headers = list(rows[0])
        escape = lambda v: str(v).replace("|", "\\|")
        lines = ["| " + " | ".join(escape(h) for h in headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
        lines += ["| " + " | ".join(escape(row.get(h, "")) for h in headers) + " |" for row in rows]
        return "\n".join(lines)

    # ===== 描画 & ダウンロード =====
    with col_graph:
        if skipped:
            st.warning(ui("縦軸の量に換算できないスペクトルは描いていません（透過率のデータは反射率に、反射率のデータは透過率にはできません）: ",
                          "Spectra that cannot be converted to the chosen quantity are not drawn (transmittance data cannot become reflectance, nor the reverse): ")
                       + ", ".join(f"{name} ({mode})" for name, mode in skipped))
        interactive_tab, export_tab = st.tabs([T["interactive_preview"], T["export_preview"]])
        pfig = build_plotly_figure()
        with interactive_tab:
            st.caption(T["drag_zoom"])
            st.plotly_chart(pfig, **FULL_WIDTH["plotly_chart"], config={
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["drawrect"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })

        _sl = bool(show_legend)
        fig = build_figure(show_legend=_sl)
        png_bytes = export_figure(fig, "png", 150, style=plot_style)
        pdf_bytes = export_figure(fig, "pdf", dpi_export, style=plot_style)
        svg_bytes = export_figure(fig, "svg", dpi_export, style=plot_style)
        # Bound the in-memory raster canvas for the shared web and PC editions.
        _raster_ok = fig_width * fig_height * dpi_export ** 2 <= 36_000_000
        png_export_bytes = export_figure(fig, "png", dpi_export, style=plot_style) if _raster_ok else None
        tiff_bytes = export_figure(fig, "tiff", dpi_export, style=plot_style) if _raster_ok else None
        plt.close(fig)

        with export_tab:
            st.image(png_bytes, **FULL_WIDTH_IMAGE)
            st.caption(T["export_preview_caption"].format(width=inches_to_mm(fig_width), height=inches_to_mm(fig_height), dpi=dpi_export))
            st.caption(ui("軸や文字が端で切れる場合は、余白または文字サイズを調整してください。", "If text or axes are clipped, adjust the margins or font sizes."))

        if y_mode == "tauc" and tauc_fit_on:
            _fit_rows = [{ui("スペクトル", "Spectrum"): plain(labels[c["index"]]),
                          "E_g (eV)": f"{c['fit']['eg']:.3f}" if c["fit"] else "—",
                          "R²": f"{c['fit']['r2']:.4f}" if c["fit"] else "—",
                          ui("点数", "Points"): c["fit"]["n"] if c["fit"] else 0} for c in curves]
            if _fit_rows:
                st.caption(ui(f"Tauc 直線の当てはめ（{tauc_range[0]:.2f}〜{tauc_range[1]:.2f} eV）。「—」は範囲内の点が 3 点未満か傾きが正でないときです。",
                              f"Tauc line fit ({tauc_range[0]:.2f}–{tauc_range[1]:.2f} eV). '—': fewer than three points in the range or a non-positive slope."))
                st.markdown(markdown_table(_fit_rows))
        if show_indices:
            _rows = indices_table()
            if _rows:
                _frame = pd.DataFrame(_rows)
                st.markdown(markdown_table(_rows))
                st.caption(ui("T_lum = ∫ȳ(λ)T(λ)dλ / ∫ȳ(λ)dλ（380〜780 nm）、T_sol = ∫φ(λ)T(λ)dλ / ∫φ(λ)dλ（AM1.5G、300〜2500 nm）。* はデータがその帯域の一部しか覆っていない値（覆う部分だけの平均）です。",
                              "T_lum = ∫ȳ(λ)T(λ)dλ / ∫ȳ(λ)dλ (380–780 nm), T_sol = ∫φ(λ)T(λ)dλ / ∫φ(λ)dλ (AM1.5G, 300–2500 nm). * marks a band the data covers only in part (average over the covered part)."))
                st.download_button(ui("📥 指標を CSV で保存", "📥 Save the indices as CSV"), _frame.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="optical_indices.csv", mime="text/csv", **FULL_WIDTH["download_button"])
            else:
                st.info(ui("指標を計算できるのは、透過率・吸光度・反射率・F(R) と分かっているスペクトルです。", "Indices need spectra known to be transmittance, absorbance, reflectance or F(R)."))

        st.sidebar.download_button(ui("📥 PDF として保存", "📥 Save as PDF"), data=pdf_bytes, file_name="uvvisnir_result.pdf", mime="application/pdf", **FULL_WIDTH["download_button"])
        st.sidebar.download_button(ui("📥 SVG として保存", "📥 Save as SVG"), data=svg_bytes, file_name="uvvisnir_result.svg", mime="image/svg+xml", **FULL_WIDTH["download_button"])
        if not _raster_ok:
            st.sidebar.warning(ui("PNG / TIFFは3,600万画素までです。図の寸法またはDPIを下げてください。PDF / SVGは保存できます。", "For PNG / TIFF, reduce the dimensions or DPI to stay within 36 megapixels. PDF / SVG are available."))
        if _raster_ok:
            st.sidebar.download_button(T["save_tiff"].format(dpi=dpi_export), data=tiff_bytes, file_name="uvvisnir_result.tiff", mime="image/tiff",
                                       key=f"dl_tiff_{_sl}_{dpi_export}", **FULL_WIDTH["download_button"])
            st.sidebar.download_button(T["save_png"].format(dpi=dpi_export), data=png_export_bytes, file_name="uvvisnir_result.png", mime="image/png",
                                       key=f"dl_png_{_sl}", **FULL_WIDTH["download_button"])
        st.sidebar.divider()

else:
    st.info(T["upload_prompt"])

if st.session_state.get("_spectra_bytes"):
    try:
        _project_bytes = build_session_json()
        st.sidebar.download_button(T["save_session"], data=_project_bytes, file_name="uvvisnir_project.json",
                                   mime="application/json", **FULL_WIDTH["download_button"])
        if LOCAL_DATA_DIR:
            try:
                _fingerprint = project_fingerprint(_project_bytes)
                if _fingerprint != st.session_state.get("_autosave_fingerprint"):
                    atomic_save(LOCAL_DATA_DIR / "autosave.json", _project_bytes)
                    st.session_state["_autosave_fingerprint"] = _fingerprint
                    st.session_state["_autosave_time"] = datetime.now().astimezone().strftime("%H:%M:%S")
                autosave_status.caption(T["autosaved"].format(time=st.session_state.get("_autosave_time", "")))
            except (OSError, ProjectError) as _error:
                autosave_status.warning(T["autosave_error"] + str(_error))
    except ProjectError as _error:
        autosave_status.warning(T["project_save_error"] + str(_error))

st.sidebar.divider()
try:
    from build_bundle import create_bundle_bytes
    st.sidebar.download_button(T["download_desktop"], data=create_bundle_bytes(), file_name="UV-Vis-NIR-Maker-PC.zip",
                               mime="application/zip", **FULL_WIDTH["download_button"])
    st.sidebar.caption(T["desktop_help"])
except (ImportError, FileNotFoundError):
    pass

# ===== 書式のショートカット（グラフ描画appと同じ） =====
# 入力欄の文字を選んで Ctrl+I（斜体）・Ctrl+B（太字）・Ctrl+,（下付き）・Ctrl+.（上付き）。Word の Ctrl+= / Ctrl+Shift+= も。
_SHORTCUTS_HTML = """<script>
(function () {
  const P = window.parent;
  if (!P || !P.document) return;
  const handler = function (e) {
    const el = e.target;
    if (!el || !(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || !el.type)))) return;
    if (el.getAttribute('role') === 'combobox' || (el.closest && el.closest('[data-baseweb="select"]'))) return;
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = e.key;
    let open = null, close = null;
    if (k === 'i' || k === 'I') { open = '<i>'; close = '</i>'; }
    else if (k === 'b' || k === 'B') { open = '<b>'; close = '</b>'; }
    else if (k === ',' || k === '=') { open = '_{'; close = '}'; }
    else if (k === '.' || k === '+') { open = '^{'; close = '}'; }
    if (!open) return;
    e.preventDefault();
    e.stopPropagation();
    const v = el.value || '';
    const a = (el.selectionStart == null) ? v.length : el.selectionStart;
    const b = (el.selectionEnd == null) ? a : el.selectionEnd;
    const nv = v.slice(0, a) + open + v.slice(a, b) + close + v.slice(b);
    const proto = el.tagName === 'TEXTAREA' ? P.HTMLTextAreaElement.prototype : P.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, nv);
    el.dispatchEvent(new P.Event('input', { bubbles: true }));
    el.setSelectionRange(a + open.length, b + open.length);
  };
  if (P.__uvvisMakerKeys) P.document.removeEventListener('keydown', P.__uvvisMakerKeys, true);
  P.__uvvisMakerKeys = handler;
  P.document.addEventListener('keydown', handler, true);
})();
</script>"""
if hasattr(st, "iframe"):     # newer Streamlit, where components.html is being removed
    st.iframe(_SHORTCUTS_HTML, height="content")
else:                         # Streamlit 1.32 and other versions before st.iframe
    import streamlit.components.v1 as components
    components.html(_SHORTCUTS_HTML, height=0)
