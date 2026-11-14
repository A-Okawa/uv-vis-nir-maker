"""Portable UV-Vis-NIR Maker projects and atomic, local-only autosave files.

The same layout as XRD Maker's and Raman Maker's projects, with this app's keys."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from figure_style import STYLE_KEYS, validate_style_settings
from palettes import PALETTES

PROJECT_VERSION = 1
PROJECT_FORMAT = "uvvisnir-maker-project"
MAX_PROJECT_BYTES = 190 * 1024 * 1024
MAX_DATA_BYTES = 140 * 1024 * 1024
X_LIMITS = (100.0, 25000.0)              # nm
ENERGY_LIMITS = (0.1, 12.0)              # eV
X_UNITS = ("nm", "um", "eV", "cm-1")
Y_MODES = ("as_measured", "T", "A", "R", "KM", "tauc")
INPUT_Y_MODES = ("auto", "T%", "T", "A", "R%", "R", "KM")
NORMALIZE_MODES = ("none", "max", "window", "minmax", "at", "area")
TAUC_TRANSITIONS = ("direct", "indirect", "direct_forbidden", "indirect_forbidden")
LEGEND_LOCATIONS = ("best", "upper right", "upper left", "lower right", "lower left", "center right", "center left",
                    "upper center", "lower center", "outside right")
LINE_STYLES = ("solid", "dashed", "dotted", "dashdot")
GLOBAL_KEYS = (
    # axes and arrangement
    "xrange", "x_unit", "y_mode", "y_percent", "tauc_transition", "tauc_fit", "tauc_range", "show_eg",
    "normalize_mode", "norm_window", "norm_at", "stack", "global_offset",
    "show_legend", "legend_loc", "yrange_auto", "yrange", "y_major_tick", "show_yticks",
    "major_tick", "show_minor", "minor_tick", "x_label", "y_label",
    # input defaults and processing
    "input_x_unit", "input_y_mode", "smooth_on", "smooth_window", "smooth_order",
    # peaks, marks, bands, inset, indices
    "show_peaks", "peak_kind", "peak_prom", "peak_distance", "peak_decimals", "peak_label_rotate",
    "marks_text", "marks_style", "marks_color", "marks_label_mode", "marks_show_value", "marks_decimals", "marks_fontsize",
    "bands_color", "bands_alpha", "bands_label",
    "show_zoom", "zoom_range", "zoom_location", "annotation_font_size", "show_indices", "indices_decimals",
    # sample labels
    "show_side_labels", "label_side", "label_fontsize", "label_offset_x", "label_offset_y",
    # figure
    "fig_width", "fig_height", "dpi_export", "font_size", "tick_font_size", "legend_font_size",
    "show_panel", "lang_radio",
    # palette
    "palette", "palette_reverse", "palette_steps", "palette_from", "palette_to", "palette_base_color",
) + tuple(STYLE_KEYS)
PATTERN_KEY = re.compile(r"(?:vis|ord|_val_lbl|sp_color|extra_offset|yoff|scale|ls)_\d+$")
BOOL_KEYS = {"y_percent", "tauc_fit", "show_eg", "stack", "show_legend", "yrange_auto", "show_yticks", "show_minor",
             "smooth_on", "show_peaks", "peak_label_rotate", "marks_show_value", "bands_label", "show_zoom",
             "show_indices", "show_side_labels", "show_panel", "palette_reverse"}
BOUNDS = {
    "global_offset": (0., 3.), "y_major_tick": (0., 1e6), "major_tick": (1e-3, 1e6), "minor_tick": (1e-4, 1e6),
    "norm_at": X_LIMITS,
    "smooth_window": (3, 101), "smooth_order": (1, 5),
    "peak_prom": (.005, .9), "peak_distance": (0., 10000.), "peak_decimals": (0, 3),
    "marks_decimals": (0, 3), "marks_fontsize": (4, 20), "bands_alpha": (0., 1.), "annotation_font_size": (4, 20),
    "indices_decimals": (0, 3),
    "label_fontsize": (4, 28), "label_offset_x": (-10000., 10000.), "label_offset_y": (-1., 2.),
    "fig_width": (25 / 25.4, 20.), "fig_height": (25 / 25.4, 20.),
    "font_size": (4, 28), "tick_font_size": (4, 28), "legend_font_size": (4, 28),
    "palette_steps": (0, 12), "palette_from": (1, 10000), "palette_to": (0, 10000),
}
INT_KEYS = {"smooth_window", "smooth_order", "peak_decimals", "marks_decimals", "marks_fontsize", "annotation_font_size",
            "indices_decimals", "label_fontsize", "palette_steps", "palette_from", "palette_to"}
MAX_AXIS_LABEL = 500
MAX_MARKS_TEXT = 20000


class ProjectError(ValueError):
    """A project is damaged, unsupported, or contains invalid display settings."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pair(value, limits, allow_equal=False):
    ok = (isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) for v in value)
          and limits[0] <= value[0] and value[1] <= limits[1] and (value[0] <= value[1] if allow_equal else value[0] < value[1]))
    return ok


def clean_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ProjectError("Project settings must be an object.")
    try:
        style_values = validate_style_settings(settings)
    except (ValueError, TypeError) as exc:
        raise ProjectError(str(exc)) from exc
    cleaned = {}
    for key, value in settings.items():
        if key not in GLOBAL_KEYS and not PATTERN_KEY.fullmatch(key):
            continue
        valid = True
        if key in STYLE_KEYS:
            value = style_values[key]
        elif key in BOOL_KEYS or re.fullmatch(r"vis_\d+", key):
            valid = isinstance(value, bool)
        elif key in BOUNDS:
            lower, upper = BOUNDS[key]
            valid = _number(value) and lower <= value <= upper
            if key in INT_KEYS:
                valid = valid and isinstance(value, int)
        elif key in ("xrange", "norm_window", "zoom_range"):
            valid = _pair(value, X_LIMITS)
            if valid:
                value = tuple(float(v) for v in value)
        elif key == "tauc_range":
            valid = _pair(value, ENERGY_LIMITS)
            if valid:
                value = tuple(float(v) for v in value)
        elif key == "yrange":
            valid = _pair(value, (-1e9, 1e9))
            if valid:
                value = tuple(float(v) for v in value)
        elif key == "x_unit" or key == "input_x_unit":
            valid = value in X_UNITS
        elif key == "y_mode":
            valid = value in Y_MODES
        elif key == "input_y_mode":
            valid = value in INPUT_Y_MODES
        elif key == "tauc_transition":
            valid = value in TAUC_TRANSITIONS
        elif key == "normalize_mode":
            valid = value in NORMALIZE_MODES
        elif key == "legend_loc":
            valid = value in LEGEND_LOCATIONS
        elif key == "peak_kind":
            valid = value in ("max", "min")
        elif key == "label_side":
            valid = value in ("左", "右")
        elif key == "lang_radio":
            valid = value in ("日本語", "English")
        elif key == "zoom_location":
            valid = value in ("upper left", "upper right", "lower left", "lower right")
        elif key in ("x_label", "y_label"):
            valid = isinstance(value, str) and len(value) <= MAX_AXIS_LABEL
        elif key == "marks_text":
            valid = isinstance(value, str) and len(value) <= MAX_MARKS_TEXT
        elif key == "marks_style":
            valid = value in ("dashed", "dotted", "solid", "dashdot")
        elif key == "marks_label_mode":
            valid = value in ("top", "rotated", "none")
        elif key in ("marks_color", "bands_color", "palette_base_color"):
            valid = isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None
        elif key == "palette":
            valid = value in PALETTES
        elif key == "dpi_export":
            valid = type(value) is int and value in (300, 600)
        elif re.fullmatch(r"ord_\d+", key):
            valid = type(value) is int and 1 <= value <= 10000
        elif key.startswith("_val_"):
            valid = isinstance(value, str) and len(value) <= 10000
        elif key.startswith("sp_color_"):
            valid = isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None
        elif key.startswith("ls_"):
            valid = value in LINE_STYLES
        elif key.startswith("extra_offset_"):
            valid = _number(value) and -5 <= value <= 15
        elif key.startswith("yoff_"):
            valid = _number(value)
        elif key.startswith("scale_"):
            valid = _number(value) and 0 < value <= 1e6
        if not valid:
            raise ProjectError(f"Invalid display setting: {key}")
        cleaned[key] = value
    return cleaned


def encode_project(spectra_files: dict, settings: dict) -> bytes:
    result = {
        "format": PROJECT_FORMAT, "version": PROJECT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": clean_settings(settings), "spectra_files": {}, "checksums": {"spectra_files": {}},
    }
    total = 0
    for name, raw in spectra_files.items():
        if not isinstance(name, str) or not name or len(name) > 1024 or "\0" in name:
            raise ProjectError("Invalid embedded filename.")
        if not isinstance(raw, bytes):
            raise ProjectError(f"Invalid file contents: {name}")
        total += len(raw)
        if total > MAX_DATA_BYTES:
            raise ProjectError("Project data exceeds the 140 MB limit.")
        result["spectra_files"][name] = base64.b64encode(raw).decode("ascii")
        result["checksums"]["spectra_files"][name] = hashlib.sha256(raw).hexdigest()
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_PROJECT_BYTES:
        raise ProjectError("Project file exceeds the 190 MB limit.")
    return payload


def decode_project(raw: bytes) -> dict:
    """Validate the complete project before the caller updates any session state."""
    if len(raw) > MAX_PROJECT_BYTES:
        raise ProjectError("Project file exceeds the 190 MB limit.")
    try:
        project = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ProjectError("This is not a valid UV-Vis-NIR Maker JSON project.") from exc
    if not isinstance(project, dict) or project.get("format") != PROJECT_FORMAT:
        raise ProjectError("Unrecognized project format (expected a UV-Vis-NIR Maker project).")
    if type(project.get("version")) is not int or project["version"] != PROJECT_VERSION:
        raise ProjectError("Unsupported UV-Vis-NIR Maker project version.")
    result = {"settings": clean_settings(project.get("settings", {})),
              "saved_at": project.get("saved_at", ""), "version": project["version"], "spectra_files": {}}
    files = project.get("spectra_files", {})
    if not isinstance(files, dict):
        raise ProjectError("Invalid file list.")
    total = 0
    for name, encoded in files.items():
        if not isinstance(name, str) or not name or len(name) > 1024 or "\0" in name or not isinstance(encoded, str):
            raise ProjectError("Invalid embedded file.")
        try:
            contents = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ProjectError(f"Corrupted embedded file: {name}") from exc
        total += len(contents)
        if total > MAX_DATA_BYTES:
            raise ProjectError("Project data exceeds the 140 MB limit.")
        try:
            expected = project["checksums"]["spectra_files"][name]
        except (KeyError, TypeError) as exc:
            raise ProjectError(f"Missing file checksum: {name}") from exc
        if expected != hashlib.sha256(contents).hexdigest():
            raise ProjectError(f"File checksum mismatch: {name}")
        result["spectra_files"][name] = contents
    return result


def project_fingerprint(raw: bytes) -> str:
    project = json.loads(raw)
    project.pop("saved_at", None)
    project["settings"] = dict(sorted(project.get("settings", {}).items()))
    return hashlib.sha256(json.dumps(project, ensure_ascii=False).encode()).hexdigest()


def atomic_save(path: str | Path, contents: bytes) -> None:
    """Replace a local checkpoint only after a complete, flushed file is written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uvvis-checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
