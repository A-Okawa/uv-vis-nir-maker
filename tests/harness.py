"""Shared helpers for the UV-Vis-NIR Maker tests: the real UI under Streamlit's AppTest with the sample files.

The app source is run as it is (authentication bypassed through session state), plus a short capture
block that stores what was drawn in ``st.session_state["_regression_report"]``. Streamlit 1.32's AppTest
serialises formatted selectbox values incorrectly, so ``format_func`` keywords are removed from the test
source (the real options and callbacks stay).
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(APP_DIR))

TREE = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"), filename=str(APP_DIR / "app.py"))

CAPTURE = '''
if active:
    import json as _json
    st.session_state["_regression_report"] = {
        "names": [item.name for item in active],
        "orders": list(orders), "visibles": list(visibles), "labels": list(labels), "colors": list(colors_sel),
        "line_styles": list(line_styles),
        "curves": [(plain(labels[c["index"]]), colors_sel[c["index"]], float(c["baseline"]), float(np.min(c["y"])), float(np.max(c["y"])),
                    float(c["x"][0]), float(c["x"][-1]), len(c["x"]), (c["fit"] or {}).get("eg")) for c in curves],
        "skipped": list(skipped),
        "mpl_lines": [(l.get_label(), l.get_color(), l.get_linestyle(), np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist())
                      for l in fig.axes[0].get_lines() if l.get_label() and not l.get_label().startswith("_")],
        "mpl_legend": [t.get_text() for t in fig.axes[0].get_legend().get_texts()] if fig.axes[0].get_legend() else [],
        "mpl_xlim": list(fig.axes[0].get_xlim()), "mpl_ylim": list(fig.axes[0].get_ylim()),
        "mpl_xlabel": fig.axes[0].get_xlabel(), "mpl_ylabel": fig.axes[0].get_ylabel(),
        "mpl_yticks": [float(v) for v in fig.axes[0].get_yticks()],
        "mpl_texts": [t.get_text() for t in fig.axes[0].texts],
        "mpl_patches": len(fig.axes[0].patches), "mpl_inset": len(fig.axes[0].child_axes),
        "preview_lines": [(t.name, t.line.color, t.line.dash, list(t.x), list(t.y)) for t in pfig.data if t.mode == "lines" and t.xaxis in (None, "x")],
        "preview_shapes": len(pfig.layout.shapes), "preview_annotations": [a.text for a in pfig.layout.annotations],
        "preview_xrange": list(pfig.layout.xaxis.range), "preview_xlabel": pfig.layout.xaxis.title.text,
        "preview_ylabel": pfig.layout.yaxis.title.text,
        "session": _json.loads(build_session_json()),
        "png_header": png_bytes[:8], "png": png_export_bytes or png_bytes, "tiff_header": (tiff_bytes or b"")[:4], "svg": svg_bytes.decode("utf-8", "replace")[:200000],
        "indices": indices_table() if show_indices else None,
    }
'''


# Test hooks, run where the app's own callbacks would run (before any widget of the sidebar is created):
# a project to restore, a style template to open, a request to start a new project.
HOOKS = ast.parse('''
if st.session_state.get("_test_restore") is not None:
    restore_session(st.session_state.pop("_test_restore"))
if st.session_state.get("_test_style") is not None:
    _open_style_template(st.session_state.pop("_test_style"))
if st.session_state.pop("_test_new_project", False):
    _new_project()
''').body


def ui_test_source() -> str:
    tree = copy.deepcopy(TREE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("selectbox", "radio"):
            node.keywords = [kw for kw in node.keywords if kw.arg != "format_func"]
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("st.sidebar.caption(f'Last updated"):
            tree.body[index:index] = copy.deepcopy(HOOKS)
            break
    else:
        raise AssertionError("the sidebar's first line was not found")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n" + CAPTURE


def data_files(*names) -> dict:
    return {name: (DATA_DIR / name).read_bytes() for name in names}


def run_app(files: dict, extra: dict | None = None, language: str = "日本語", timeout: float = 120):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(ui_test_source(), default_timeout=timeout)
    state = {"authenticated": True, "lang_radio": language, "_spectra_bytes": dict(files), "show_panel": True}
    state.update(extra or {})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def rerun(at, extra: dict | None = None):
    for key, value in (extra or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def report(at) -> dict:
    assert not at.exception, [e.message for e in at.exception]
    return at.session_state["_regression_report"]
