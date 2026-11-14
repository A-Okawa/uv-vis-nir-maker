"""Conversions and small analyses for UV-Vis-NIR spectra: pure numpy/scipy, used by the app and the tests.

- wavelength units (nm, µm, eV, cm⁻¹) and the ordinate kinds T (transmittance), A (absorbance),
  R (reflectance) and F(R) (Kubelka–Munk);
- Savitzky–Golay smoothing, normalisation, peak (maximum / minimum) detection;
- Tauc plots, (αhν)^(1/n) against hν, with a linear fit extrapolated to the band gap;
- luminous and solar averages (T_lum with the CIE 1931 photopic ȳ(λ); T_sol / T_nir / T_vis / T_uv with
  the ASTM G173-03 AM1.5 global-tilt spectrum), as used for thermochromic and NIR-shielding films
  and for the NIR solar reflectance of pigments.

Nothing here changes the stored data of a spectrum; every function returns new arrays.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np


EV_NM = 1239.841984                      # h c / e in eV nm
DATA_DIR = Path(__file__).resolve().parent / "data"
X_UNITS = ("nm", "um", "eV", "cm-1")
# Axis titles in the graph app's markup (<i>…</i>, _{…}, ^{…}).
X_UNIT_LABELS = {"nm": "Wavelength (nm)", "um": "Wavelength (µm)", "eV": "Photon energy (eV)",
                 "cm-1": "Wavenumber (cm^{-1})"}
X_UNIT_TEXT = {"nm": "nm", "um": "µm", "eV": "eV", "cm-1": "cm⁻¹"}
Y_MODES = ("as_measured", "T", "A", "R", "KM", "tauc")
Y_MODE_LABELS = {"T": "Transmittance (%)", "A": "Absorbance", "R": "Reflectance (%)",
                 "KM": "Kubelka–Munk function, <i>F</i>(<i>R</i>)", "as_measured": "Intensity (a.u.)"}
# Tauc exponents: (αhν)^(1/n) with n = 1/2 (direct allowed), 2 (indirect allowed), 3/2 (direct forbidden), 3 (indirect forbidden).
TAUC_EXPONENTS = {"direct": 2.0, "indirect": 0.5, "direct_forbidden": 2.0 / 3.0, "indirect_forbidden": 1.0 / 3.0}
TAUC_LABELS = {"direct": "(<i>αhν</i>)^{2} (a.u.)", "indirect": "(<i>αhν</i>)^{1/2} (a.u.)",
               "direct_forbidden": "(<i>αhν</i>)^{2/3} (a.u.)", "indirect_forbidden": "(<i>αhν</i>)^{1/3} (a.u.)"}
_T_FLOOR = 1e-12                         # transmittance below this is treated as this (A would be infinite)
_R_FLOOR = 1e-9
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")      # NumPy 2 renamed trapz


# ------------------------------------------------------------------------------------------- units
def x_from_nm(x_nm, unit: str) -> np.ndarray:
    """Wavelengths in nm expressed in another axis unit (energy and wavenumber reverse the order of the points)."""
    x = np.asarray(x_nm, dtype=float)
    if unit == "nm":
        return x.copy()
    if unit == "um":
        return x / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if unit == "eV":
            return EV_NM / x
        if unit == "cm-1":
            return 1e7 / x
    raise ValueError(f"Unknown axis unit '{unit}'.")


def x_to_nm(x, unit: str) -> np.ndarray:
    """The inverse of x_from_nm (the conversions are their own inverses for eV and cm⁻¹)."""
    x = np.asarray(x, dtype=float)
    if unit == "nm":
        return x.copy()
    if unit == "um":
        return x * 1000.0
    return x_from_nm(x, unit)


def range_in_unit(x_min_nm: float, x_max_nm: float, unit: str) -> tuple[float, float]:
    """A nm range as the axis limits in another unit, low to high."""
    a, b = (float(v) for v in x_from_nm([x_min_nm, x_max_nm], unit))
    return (min(a, b), max(a, b))


# ------------------------------------------------------------------------------------------- ordinates
def _check_mode(mode: str):
    if mode not in ("T", "A", "R", "KM"):
        raise ValueError(f"Unknown ordinate kind '{mode}'.")


def transmittance_fraction(y, mode: str, percent: bool) -> np.ndarray:
    """T as a fraction (0–1) from T or A values."""
    y = np.asarray(y, dtype=float)
    if mode == "T":
        return y / 100.0 if percent else y.copy()
    if mode == "A":
        return np.power(10.0, -y)
    raise ValueError("Transmittance can be derived from transmittance or absorbance only.")


def reflectance_fraction(y, mode: str, percent: bool) -> np.ndarray:
    """R as a fraction (0–1) from R or F(R) values."""
    y = np.asarray(y, dtype=float)
    if mode == "R":
        return y / 100.0 if percent else y.copy()
    if mode == "KM":
        f = np.maximum(y, 0.0)
        return 1.0 + f - np.sqrt(f * f + 2.0 * f)
    raise ValueError("Reflectance can be derived from reflectance or the Kubelka–Munk function only.")


def convert_y(y, from_mode: str, from_percent: bool, to_mode: str, to_percent: bool = False) -> np.ndarray:
    """Convert between T ↔ A and R ↔ F(R). T/A and R/F(R) are different quantities: converting across is an error."""
    _check_mode(from_mode)
    _check_mode(to_mode)
    y = np.asarray(y, dtype=float)
    if from_mode in ("T", "A"):
        if to_mode not in ("T", "A"):
            raise ValueError("Transmittance / absorbance cannot be converted to reflectance.")
        t = transmittance_fraction(y, from_mode, from_percent)
        if to_mode == "T":
            return t * 100.0 if to_percent else t
        with np.errstate(divide="ignore"):
            return -np.log10(np.maximum(t, _T_FLOOR))
    if to_mode not in ("R", "KM"):
        raise ValueError("Reflectance / Kubelka–Munk cannot be converted to transmittance.")
    r = reflectance_fraction(y, from_mode, from_percent)
    if to_mode == "R":
        return r * 100.0 if to_percent else r
    r = np.clip(r, _R_FLOOR, 1.0)
    return (1.0 - r) ** 2 / (2.0 * r)


def kubelka_munk(r, percent: bool = True) -> np.ndarray:
    """F(R) = (1 − R)² / 2R from the reflectance (percent by default)."""
    return convert_y(r, "R", percent, "KM")


def absorption_like(y, mode: str, percent: bool) -> np.ndarray:
    """A quantity proportional to the absorption coefficient α: A (from T or A), or F(R) (from R or F(R)).

    Absolute α needs the thickness; the Tauc extrapolation to the band gap does not, so the
    proportionality constant is dropped.
    """
    _check_mode(mode)
    if mode in ("T", "A"):
        return convert_y(y, mode, percent, "A")
    return convert_y(y, mode, percent, "KM")


def tauc_transform(x_nm, y, mode: str, percent: bool, transition: str = "direct") -> tuple[np.ndarray, np.ndarray]:
    """(hν in eV, (αhν)^(1/n)) for a Tauc plot; points ordered by increasing energy."""
    if transition not in TAUC_EXPONENTS:
        raise ValueError(f"Unknown transition '{transition}'.")
    x_nm = np.asarray(x_nm, dtype=float)
    alpha = absorption_like(y, mode, percent)
    energy = EV_NM / x_nm
    with np.errstate(invalid="ignore"):
        value = np.power(np.maximum(alpha * energy, 0.0), TAUC_EXPONENTS[transition])
    order = np.argsort(energy, kind="stable")
    return energy[order], value[order]


def tauc_fit(energy, value, e_low: float, e_high: float) -> dict | None:
    """A least-squares line through the points with e_low ≤ hν ≤ e_high, extrapolated to zero.

    Returns {'slope', 'intercept', 'eg', 'r2', 'n', 'e_low', 'e_high'} or None when fewer than
    three points lie in the range or the slope is not positive (no absorption edge there).
    """
    energy, value = np.asarray(energy, dtype=float), np.asarray(value, dtype=float)
    if not (math.isfinite(e_low) and math.isfinite(e_high)) or e_low >= e_high:
        raise ValueError("The fit range must be two increasing finite energies.")
    inside = (energy >= e_low) & (energy <= e_high) & np.isfinite(value)
    if int(inside.sum()) < 3:
        return None
    e, v = energy[inside], value[inside]
    slope, intercept = np.polyfit(e, v, 1)
    if not slope > 0:
        return None
    predicted = slope * e + intercept
    total = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - float(np.sum((v - predicted) ** 2)) / total if total > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "eg": float(-intercept / slope),
            "r2": r2, "n": int(inside.sum()), "e_low": float(e_low), "e_high": float(e_high)}


# ------------------------------------------------------------------------------------------- processing
def smooth(y, window: int = 11, order: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing; the window is made odd and larger than the order."""
    y = np.asarray(y, dtype=float)
    window = int(window)
    order = int(order)
    if window % 2 == 0:
        window += 1
    if window <= order:
        window = order + 2 if (order + 2) % 2 else order + 3
    if len(y) <= window:
        return y.copy()
    from scipy.signal import savgol_filter
    return savgol_filter(y, window, order)


def normalize(x_nm, y, mode: str = "none", window=None, at: float | None = None) -> np.ndarray:
    """Scale a curve: 'none', 'max' (maximum = 1), 'window' (maximum within window = 1), 'minmax'
    (minimum 0, maximum 1), 'at' (value at wavelength ``at`` = 1), 'area' (unit area)."""
    x_nm, y = np.asarray(x_nm, dtype=float), np.asarray(y, dtype=float)
    if mode == "none" or not len(y):
        return y.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return y.copy()
    if mode == "max":
        top = float(np.max(np.abs(y[finite])))
        return y / top if top > 0 else y.copy()
    if mode == "window":
        if window is None:
            return normalize(x_nm, y, "max")
        lo, hi = sorted((float(window[0]), float(window[1])))
        inside = finite & (x_nm >= lo) & (x_nm <= hi)
        if not inside.any():
            return normalize(x_nm, y, "max")
        top = float(np.max(np.abs(y[inside])))
        return y / top if top > 0 else y.copy()
    if mode == "minmax":
        low, high = float(np.min(y[finite])), float(np.max(y[finite]))
        return (y - low) / (high - low) if high > low else y - low
    if mode == "at":
        if at is None:
            return normalize(x_nm, y, "max")
        value = interpolate_at(x_nm, y, float(at))
        return y / value if value and math.isfinite(value) and value != 0 else y.copy()
    if mode == "area":
        order = np.argsort(x_nm)
        area = float(_trapz(np.abs(y[order][finite[order]]), x_nm[order][finite[order]]))
        return y / area if area > 0 else y.copy()
    raise ValueError(f"Unknown normalisation '{mode}'.")


def interpolate_at(x, y, x0: float) -> float:
    """The linearly interpolated value at x0, or NaN outside the data."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if not len(x) or x0 < x[0] or x0 > x[-1]:
        return math.nan
    return float(np.interp(x0, x, y))


def detect_peaks(x, y, prominence_fraction: float = 0.05, min_distance: float = 10.0, kind: str = "max") -> np.ndarray:
    """Indices of local maxima (or minima, kind='min') by prominence relative to the curve's span,
    at least ``min_distance`` apart in the units of x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) < 3 or not np.isfinite(y).all() or np.ptp(y) == 0:
        return np.array([], dtype=int)
    from scipy.signal import find_peaks
    signal = -y if kind == "min" else y
    peaks, properties = find_peaks(signal, prominence=float(prominence_fraction) * float(np.ptp(y)))
    ranked = sorted(range(len(peaks)), key=lambda j: (-properties["prominences"][j], x[peaks[j]]))
    kept: list[int] = []
    for j in ranked:
        candidate = int(peaks[j])
        if all(abs(x[candidate] - x[other]) >= min_distance for other in kept):
            kept.append(candidate)
    return np.asarray(sorted(kept), dtype=int)


# ------------------------------------------------------------------------------------------- luminous / solar averages
@lru_cache(maxsize=None)
def photopic_table() -> tuple[np.ndarray, np.ndarray]:
    """CIE 1931 2° ȳ(λ) at 1 nm (360–830 nm)."""
    return _read_table(DATA_DIR / "cie1931_ybar.csv")


@lru_cache(maxsize=None)
def am15g_table() -> tuple[np.ndarray, np.ndarray]:
    """ASTM G173-03 global tilt irradiance (W m⁻² nm⁻¹), 280–4000 nm."""
    return _read_table(DATA_DIR / "astm_g173_global_tilt.csv")


def _read_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            a, b = line.split(",")[:2]
            rows.append((float(a), float(b)))
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


WEIGHTED_BANDS = {
    # name: (weight, low nm, high nm)
    "lum": ("photopic", 380.0, 780.0),
    "sol": ("am15g", 300.0, 2500.0),
    "vis": ("am15g", 380.0, 780.0),
    "nir": ("am15g", 780.0, 2500.0),
    "uv": ("am15g", 300.0, 380.0),
}


def weighted_average(x_nm, y, weight: str, low: float, high: float) -> tuple[float, float]:
    """∫ w(λ) y(λ) dλ / ∫ w(λ) dλ over [low, high] on a 1 nm grid, and the fraction of that band the data covers.

    ``weight`` is 'photopic' (CIE ȳ) or 'am15g' (ASTM G173 global tilt); the spectrum is interpolated onto
    the grid, and where it does not reach, the band is truncated to the covered part (the second value
    tells how much, 1.0 when the data covers the whole band).
    """
    x_nm, y = np.asarray(x_nm, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(x_nm, kind="stable")
    x_nm, y = x_nm[order], y[order]
    finite = np.isfinite(y)
    x_nm, y = x_nm[finite], y[finite]
    if len(x_nm) < 2:
        return math.nan, 0.0
    wx, wy = photopic_table() if weight == "photopic" else am15g_table()
    lo, hi = max(low, float(x_nm[0]), float(wx[0])), min(high, float(x_nm[-1]), float(wx[-1]))
    if hi - lo < 1.0:
        return math.nan, 0.0
    grid = np.arange(math.ceil(lo), math.floor(hi) + 1, 1.0)
    if len(grid) < 2:
        return math.nan, 0.0
    w = np.interp(grid, wx, wy)
    v = np.interp(grid, x_nm, y)
    denominator = float(_trapz(w, grid))
    if denominator <= 0:
        return math.nan, 0.0
    coverage = (float(grid[-1]) - float(grid[0])) / (high - low)
    return float(_trapz(w * v, grid)) / denominator, min(1.0, coverage)


def optical_indices(x_nm, y, mode: str, percent: bool) -> dict:
    """T_lum, T_sol, T_vis, T_nir, T_uv (as fractions) and their band coverages; for reflectance data the same
    averages of R. Returns {'quantity': 'T' or 'R', 'values': {name: value}, 'coverage': {name: fraction}}."""
    _check_mode(mode)
    if mode in ("T", "A"):
        fraction, quantity = transmittance_fraction(y, mode, percent), "T"
    else:
        fraction, quantity = reflectance_fraction(y, mode, percent), "R"
    values, coverage = {}, {}
    for name, (weight, low, high) in WEIGHTED_BANDS.items():
        values[name], coverage[name] = weighted_average(x_nm, fraction, weight, low, high)
    return {"quantity": quantity, "values": values, "coverage": coverage}
