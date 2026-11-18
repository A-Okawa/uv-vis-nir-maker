"""Portable figure appearance, separate from measurements and sample settings."""
from __future__ import annotations

import json
from figure_style import STYLE_KEYS
from project_io import clean_settings, ProjectError

TEMPLATE_KEYS = frozenset(STYLE_KEYS) | {
    "fig_width", "fig_height", "dpi_export", "font_size", "tick_font_size", "legend_font_size",
    "show_legend", "legend_loc", "legend_ncol", "legend_fit", "show_yticks", "y_major_tick", "major_tick", "show_minor", "minor_tick",
    "show_side_labels", "label_side", "label_fontsize", "label_offset_x", "label_offset_y",
    "annotation_font_size", "x_label", "y_label",
    "marks_style", "marks_color", "marks_label_mode", "marks_show_value", "marks_decimals", "marks_fontsize",
    "bands_color", "bands_alpha", "bands_label", "peak_decimals", "peak_label_rotate", "indices_decimals",
}
STYLE_FORMAT = "uvvisnir-maker-style"
SISTER_FORMATS = ("xrd-maker-style", "raman-maker-style")
MAX_TEMPLATE_BYTES = 64 * 1024


def encode_style_template(settings: dict, name: str = "Figure style") -> bytes:
    if not isinstance(name, str) or len(name) > 200:
        raise ProjectError("Invalid style name.")
    cleaned = clean_settings({k: v for k, v in settings.items() if k in TEMPLATE_KEYS})
    return json.dumps({"format": STYLE_FORMAT, "version": 1, "name": name, "settings": cleaned},
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode_style_template(raw: bytes) -> dict:
    if len(raw) > MAX_TEMPLATE_BYTES:
        raise ProjectError("Style template exceeds the 64 KB limit.")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a valid figure style JSON file.") from exc
    if (not isinstance(document, dict) or document.get("format") not in (STYLE_FORMAT,) + SISTER_FORMATS
            or type(document.get("version")) is not int or document["version"] != 1
            or not isinstance(document.get("settings"), dict)):
        raise ProjectError("Unsupported figure style template.")
    # XRD Maker and Raman Maker styles carry keys of their own (2θ ticks, Raman shift labels…); only the
    # shared appearance keys are taken, and their axis titles and tick intervals (other units) are left out.
    settings = {k: v for k, v in document["settings"].items() if k in TEMPLATE_KEYS}
    if document.get("format") in SISTER_FORMATS:
        for key in ("x_label", "y_label", "major_tick", "minor_tick", "y_major_tick", "label_offset_x"):
            settings.pop(key, None)
    elif any(k not in TEMPLATE_KEYS for k in document["settings"]):
        raise ProjectError("Style template contains settings unrelated to figure appearance.")
    return clean_settings(settings)
