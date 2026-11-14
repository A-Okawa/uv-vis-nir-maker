"""Offline readers for UV-Vis-NIR spectra (one or more spectra per file).

Supported layouts
-----------------
- JASCO Spectra Manager text / CSV exports (``TITLE … XUNITS … YUNITS … XYDATA``, then
  ``##### Extended Information``), as written by the V-series (V-650/V-670/V-770/V-780);
  tab- or comma-separated; several Y columns become separate spectra. The extended
  information (bandwidth, scan speed, light source, …) is kept as metadata.
- Shimadzu UVProbe text exports (``"sample.spc - RawData"`` title, ``"Wavelength nm."  "Abs."``
  header; several RawData blocks in one file become one spectrum each) and UVProbe CSV.
- Agilent Cary CSV exports (sample names on the first row, then paired
  ``Wavelength (nm),%T,Wavelength (nm),%T,…`` columns; shorter columns are allowed).
- Hitachi UH4150 / U-4100 and PerkinElmer Lambda text / CSV exports, and any generic table:
  a wavelength column followed by one or more value columns, with or without a header row
  (comma, semicolon, tab or space separated; decimal commas are accepted).
- JCAMP-DX (``.jdx`` / ``.dx``) with AFFN or ASDF (SQZ / DIF / DUP) ``XYDATA``, ``XYPOINTS``
  or ``PEAK TABLE``; compound files give one spectrum per block.
- Excel workbooks (``.xlsx``): every sheet is read as such a table.

Values are returned as stored (no smoothing, no unit conversion of the ordinate). Wavelengths
are converted to nanometres when the header names another unit (eV, cm⁻¹, µm); the original
unit is kept in the metadata. The ordinate kind (``T``, ``A``, ``R``, ``KM`` or ``unknown``) and
whether it is in percent come from the header text where it says so, and from the values
otherwise (this guess is recorded as ``y_mode_guessed``). Nothing is guessed about which
column holds what beyond "first column = wavelength, the others = values".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


SUPPORTED_EXTENSIONS = ("txt", "csv", "dat", "asc", "xy", "jdx", "dx", "jcamp", "xlsx", "tsv", "prn")
_MAX_POINTS = 2_000_000
_MAX_SPECTRA = 500
_EXPORT_HELP = "Export the spectrum as text/CSV (wavelength and value columns) from the instrument software."
EV_NM = 1239.841984                      # h c / e in eV nm (CODATA 2018)

_Y_MODE_KEYWORDS = (
    # (regex on the lower-cased header/unit text, mode); percent is read from a % sign in the text
    (r"kubelka|\bk[-/ ]?m\b|f\s*\(\s*r", "KM"),
    (r"reflect|\brefl\b", "R"),
    (r"transmit|\btrans\b|\btransm\b", "T"),
    (r"absorb|\babs\b|optical density|\bo\.?d\b", "A"),
)
# A header that is only the letter, with or without % and a unit in brackets: "T", "%T", "T%", "T (%)", "R%", "A", "Abs."
_Y_MODE_LETTER = re.compile(r"^%?\s*([tra])\s*%?\s*(?:\(.*\))?\.?$")
_X_UNIT_PATTERNS = (
    (r"\bev\b|electron ?volt|energy", "eV"),
    (r"cm\s*[-−‐]\s*1|cm⁻¹|wavenumber|1/cm", "cm-1"),
    (r"(?<![a-z])(?:um|µm|μm)(?![a-z])|micron", "um"),
    (r"\bnm\b|nanomet|wavelength|lambda|λ|波長", "nm"),
)


class UvVisParseError(ValueError):
    """The file is damaged, ambiguous, or uses an unsupported format."""


@dataclass
class Spectrum:
    name: str
    x: np.ndarray                     # wavelength in nm, ascending
    y: np.ndarray                     # as stored
    metadata: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------- helpers
def _text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        raise UvVisParseError("Binary data was found where a text export was expected. " + _EXPORT_HELP)
    for encoding in ("utf-8-sig", "cp932", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UvVisParseError("The text encoding could not be read. Save a UTF-8 text/CSV export.")


_NUMBER_RE = re.compile(r"^[+\-−]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+\-]?\d+)?$")
_DECIMAL_COMMA_RE = re.compile(r"^[+\-−]?\d+,\d+(?:[eE][+\-]?\d+)?$")


def _number(value: str) -> float:
    """A number as the instruments write them: 1.5, 1,5 (decimal comma), 1.5E+2, 1.5D+2, −1.5."""
    token = value.strip().strip('"').strip("'").replace("−", "-")
    if _DECIMAL_COMMA_RE.match(token):
        token = token.replace(",", ".")
    if not _NUMBER_RE.match(token):
        raise ValueError(token)
    return float(token.replace("D", "E").replace("d", "e"))


def _is_number(value: str) -> bool:
    try:
        _number(value)
        return True
    except ValueError:
        return False


def _unquote(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def _x_unit_from_text(text: str) -> str | None:
    lowered = str(text or "").lower()
    for pattern, unit in _X_UNIT_PATTERNS:
        if re.search(pattern, lowered):
            return unit
    return None


def y_mode_from_text(text: str) -> tuple[str, bool] | None:
    """(mode, percent) named by a header or unit text such as '%T', 'Abs.', 'R%', 'Transmittance (%)', 'F(R)';
    None if it says nothing (a sample name such as 'Probe A' or '20 °C' names no ordinate)."""
    lowered = _unquote(str(text or "")).lower()
    if not lowered:
        return None
    percent = "%" in lowered or "percent" in lowered
    match = _Y_MODE_LETTER.match(lowered)
    if match:
        return match.group(1).upper(), percent
    for pattern, mode in _Y_MODE_KEYWORDS:
        if re.search(pattern, lowered):
            return mode, percent and mode != "KM"
    return None


def guess_y_mode(y: np.ndarray) -> tuple[str, bool]:
    """A reading of the values when no header names the ordinate.

    Values up to about 1 → 'T' as a fraction; values reaching above 4 but staying within 0–110 → percent (returned
    as 'T'); anything else (values between 1 and 4, negative values, values above 110) → 'A' (absorbance is unbounded;
    a %T spectrum that never reaches 4 % would be read as absorbance, so such data should say %T in its header or be
    read with the app's input setting).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if not len(y):
        return "unknown", False
    low, high = float(np.min(y)), float(np.max(y))
    if -0.05 <= low and high <= 1.05:
        return "T", False
    if -0.5 <= low and 4.0 < high <= 110.0:
        return "T", True
    return "A", False


def to_nm(x: np.ndarray, unit: str) -> np.ndarray:
    """Wavelengths in nm from the stated unit."""
    x = np.asarray(x, dtype=float)
    if unit == "nm":
        return x
    if unit == "um":
        return x * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if unit == "eV":
            return EV_NM / x
        if unit == "cm-1":
            return 1e7 / x
    raise UvVisParseError(f"Unknown wavelength unit '{unit}'.")


def _spectrum(filename: str, name: str, x, y, metadata: dict, index: int = 1) -> Spectrum:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise UvVisParseError("A spectrum has mismatched wavelength/value columns.")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise UvVisParseError(f"'{name}' has fewer than two finite points.")
    if len(x) > _MAX_POINTS:
        raise UvVisParseError("The spectrum contains too many points.")
    unit = metadata.get("x_unit", "nm")
    x_nm = to_nm(x, unit)
    keep = np.isfinite(x_nm) & (x_nm > 0)
    x_nm, y = x_nm[keep], y[keep]
    if len(x_nm) < 2:
        raise UvVisParseError(f"'{name}' has no positive wavelengths after the unit conversion.")
    order = np.argsort(x_nm, kind="stable")
    meta = {"source_file": Path(filename).name, "index": index, "x_unit": unit, "y_mode": "unknown",
            "y_percent": False, "y_unit": "", "y_mode_guessed": False, "x_descending": bool(len(x) > 1 and x[0] > x[-1]),
            **metadata}
    if meta.get("y_mode") in (None, "unknown"):
        mode, percent = guess_y_mode(y)
        meta.update(y_mode=mode, y_percent=percent, y_mode_guessed=True)
    elif meta["y_mode"] in ("T", "R") and not meta.get("y_percent") and len(y) and float(np.nanmax(y)) > 1.5:
        # "TRANSMITTANCE" / "T" without a % sign, but values far above 1: they are percent.
        meta["y_percent"] = True
    return Spectrum(name, x_nm[order], y[order], meta)


def _stem(filename: str) -> str:
    return Path(filename).stem or "spectrum"


# ----------------------------------------------------------------------------------------- JASCO
_JASCO_KEYS = {"TITLE", "DATA TYPE", "ORIGIN", "OWNER", "DATE", "TIME", "SPECTROMETER/DATA SYSTEM",
               "LOCALE", "RESOLUTION", "DELTAX", "XUNITS", "YUNITS", "FIRSTX", "LASTX", "NPOINTS",
               "FIRSTY", "MAXY", "MINY"}


def looks_like_jasco(text: str) -> bool:
    head = text[:4000]
    return bool(re.search(r"(?m)^TITLE[\t,]", head)) and bool(re.search(r"(?m)^XYDATA\s*$", head))


def _read_jasco(text: str, filename: str) -> list[Spectrum]:
    header: dict[str, str] = {}
    rows: list[list[float]] = []
    extended: dict[str, dict[str, str]] = {}
    section = None
    stage = "header"
    delimiter = "\t" if "\t" in text[:2000] else ","
    width = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip("\r\n")
        if stage == "header":
            if line.strip() == "XYDATA":
                stage = "data"
                continue
            if not line.strip():
                continue
            key, _, value = line.partition(delimiter)
            header[key.strip().upper()] = value.strip()
            continue
        if stage == "data":
            if not line.strip():
                continue
            if line.startswith("#####"):
                stage = "extended"
                continue
            fields = [f for f in line.split(delimiter)] if delimiter in line else line.split()
            fields = [f.strip() for f in fields if f.strip()]
            try:
                vals = [_number(v) for v in fields]
            except ValueError as exc:
                raise UvVisParseError(f"Line {lineno}: a JASCO data row contains a nonnumeric value.") from exc
            if len(vals) < 2:
                raise UvVisParseError(f"Line {lineno}: a JASCO data row needs a wavelength and a value.")
            if width is None:
                width = len(vals)
            elif len(vals) != width:
                raise UvVisParseError(f"Line {lineno}: JASCO data rows have different numbers of columns.")
            rows.append(vals)
            continue
        # extended information: [Section] then "Key<tab>Value"
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            extended.setdefault(section, {})
            continue
        key, _, value = stripped.partition(delimiter)
        if section is None:
            section = "Information"
            extended.setdefault(section, {})
        extended[section][key.strip()] = value.strip()
    if not rows:
        raise UvVisParseError("The JASCO file has no XYDATA rows.")
    arr = np.asarray(rows)
    expected = header.get("NPOINTS")
    if expected and expected.isdigit() and int(expected) != len(arr):
        raise UvVisParseError("JASCO NPOINTS does not match the number of data rows (possibly truncated).")
    x_unit = _x_unit_from_text(header.get("XUNITS", "")) or "nm"
    conditions = {}
    for values in extended.values():
        conditions.update(values)
    spectra = []
    n_y = arr.shape[1] - 1
    for j in range(n_y):
        unit_key = "YUNITS" if j == 0 else f"Y{j + 1}UNITS"
        y_unit = header.get(unit_key, "")
        named = y_mode_from_text(y_unit)
        meta = {"format": "JASCO text/CSV", "x_unit": x_unit, "y_unit": y_unit, "title": header.get("TITLE", ""),
                "instrument": header.get("SPECTROMETER/DATA SYSTEM", ""), "date": header.get("DATE", ""),
                "time": header.get("TIME", ""), "vendor_header": dict(header), "conditions": conditions}
        if named:
            meta.update(y_mode=named[0], y_percent=named[1])
        name = _stem(filename) if n_y == 1 else f"{_stem(filename)} [{y_unit or f'Y{j + 1}'}]"
        spectra.append(_spectrum(filename, name, arr[:, 0], arr[:, j + 1], meta, j + 1))
    return spectra


# ----------------------------------------------------------------------------------------- JCAMP-DX
_SQZ = {c: i for i, c in enumerate("@ABCDEFGHI")}
_SQZ.update({c: -(i + 1) for i, c in enumerate("abcdefghi")})
_DIF = {c: i for i, c in enumerate("%JKLMNOPQR")}
_DIF.update({c: -(i + 1) for i, c in enumerate("jklmnopqr")})
_DUP = {c: i + 1 for i, c in enumerate("STUVWXYZs")}
_ASDF_TOKEN = re.compile(r"[@A-Ia-i%J-Rj-rS-Zs][0-9]*(?:\.[0-9]*)?|[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?")


def looks_like_jcamp(text: str) -> bool:
    return bool(re.search(r"(?mi)^\s*##TITLE\s*=", text[:4000])) and "##" in text


def _asdf_number(table: dict, token: str) -> float:
    """A SQZ/DIF/DUP token: the letter is the (signed) first digit, the rest of the token the remaining digits."""
    first = table[token[0]]
    text = f"{abs(first)}{token[1:]}"
    value = float(text)
    return -value if first < 0 else value


def _decode_asdf_line(line: str) -> tuple[float, list[float], bool]:
    """One ``X Y Y Y…`` line in AFFN/SQZ/DIF/DUP → (x, [y…], last_was_dif)."""
    tokens = _ASDF_TOKEN.findall(line.replace(",", " "))
    if not tokens:
        raise UvVisParseError("An empty JCAMP data line was found.")
    try:
        x = float(tokens[0])
    except ValueError as exc:
        raise UvVisParseError("A JCAMP data line does not start with an abscissa value.") from exc
    values: list[float] = []
    last_dif = False
    last_delta = 0.0
    for token in tokens[1:]:
        head = token[0]
        if head in _SQZ:
            values.append(_asdf_number(_SQZ, token))
            last_dif = False
        elif head in _DIF:
            if not values:
                raise UvVisParseError("A JCAMP DIF value has no preceding ordinate.")
            last_delta = _asdf_number(_DIF, token)
            values.append(values[-1] + last_delta)
            last_dif = True
        elif head in _DUP:
            count = int(_asdf_number(_DUP, token))
            if not values or count < 1:
                raise UvVisParseError("A JCAMP DUP count has nothing to repeat.")
            for _ in range(count - 1):
                values.append(values[-1] + (last_delta if last_dif else 0.0))
        else:
            values.append(float(token))
            last_dif = False
    return x, values, last_dif


def _read_jcamp_block(lines: list[str], filename: str, index: int) -> Spectrum | None:
    header: dict[str, str] = {}
    data_lines: list[str] = []
    data_kind = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("$$"):
            continue
        if line.startswith("##"):
            if data_kind is not None and data_lines:
                break
            key, _, value = line[2:].partition("=")
            key = re.sub(r"[\s_\-/]", "", key.upper())
            value = value.split("$$", 1)[0].strip()
            if key in ("XYDATA", "XYPOINTS", "PEAKTABLE"):
                data_kind = (key, value.replace(" ", "").upper())
                continue
            header[key] = value
            continue
        if data_kind is not None:
            data_lines.append(line.split("$$", 1)[0])
    if data_kind is None:
        return None
    kind, form = data_kind
    x_factor = float(header.get("XFACTOR", "1") or 1)
    y_factor = float(header.get("YFACTOR", "1") or 1)
    xs: list[float] = []
    ys: list[float] = []
    if kind == "XYDATA" and form.startswith("(X++(Y..Y))"):
        npoints = int(float(header["NPOINTS"])) if header.get("NPOINTS") else None
        first_x, last_x = header.get("FIRSTX"), header.get("LASTX")
        delta = None
        if header.get("DELTAX"):
            delta = float(header["DELTAX"])
        elif first_x is not None and last_x is not None and npoints and npoints > 1:
            delta = (float(last_x) - float(first_x)) / (npoints - 1)
        previous_dif = False
        for line in data_lines:
            x0, values, last_dif = _decode_asdf_line(line)
            if previous_dif and values:
                values = values[1:]                      # the check value repeats the last ordinate
            if delta is None:
                raise UvVisParseError("JCAMP (X++(Y..Y)) data needs DELTAX or FIRSTX/LASTX/NPOINTS.")
            for k, value in enumerate(values):
                xs.append((x0 + k * delta / x_factor) * x_factor)
                ys.append(value * y_factor)
            previous_dif = last_dif
    else:
        for line in data_lines:
            tokens = [t for t in re.split(r"[\s,;]+", line) if t]
            if len(tokens) % 2:
                raise UvVisParseError("A JCAMP (XY..XY) line has an odd number of values.")
            for a, b in zip(tokens[::2], tokens[1::2]):
                xs.append(float(a) * x_factor)
                ys.append(float(b) * y_factor)
    if npoints_text := header.get("NPOINTS"):
        try:
            if int(float(npoints_text)) != len(xs):
                raise UvVisParseError("JCAMP NPOINTS does not match the decoded data.")
        except ValueError:
            pass
    x_unit = _x_unit_from_text(header.get("XUNITS", "")) or "nm"
    y_unit = header.get("YUNITS", "")
    named = y_mode_from_text(y_unit)
    meta = {"format": "JCAMP-DX", "x_unit": x_unit, "y_unit": y_unit, "title": header.get("TITLE", ""),
            "instrument": header.get("SPECTROMETER/DATASYSTEM", header.get("SPECTROMETERDATASYSTEM", "")),
            "date": header.get("DATE", ""), "vendor_header": dict(header)}
    if named:
        meta.update(y_mode=named[0], y_percent=named[1])
    title = header.get("TITLE", "")
    name = _stem(filename) if index == 1 and not title else (title or f"{_stem(filename)} [{index}]")
    return _spectrum(filename, name, xs, ys, meta, index)


def _read_jcamp(text: str, filename: str) -> list[Spectrum]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    depth = 0
    for line in lines:
        key = line.strip()[2:].split("=", 1)[0].strip().upper() if line.strip().startswith("##") else None
        if key == "TITLE":
            if current and depth <= 1:
                blocks.append(current)
                current = []
            depth += 1
        current.append(line)
        if key == "END":
            depth = max(0, depth - 1)
            if depth == 0 or depth == 1 and len(blocks) == 0 and False:
                pass
    if current:
        blocks.append(current)
    spectra = []
    for block in blocks:
        spectrum = _read_jcamp_block(block, filename, len(spectra) + 1)
        if spectrum is not None:
            spectra.append(spectrum)
        if len(spectra) > _MAX_SPECTRA:
            raise UvVisParseError("The JCAMP file contains too many spectra.")
    if not spectra:
        raise UvVisParseError("The JCAMP file has no XYDATA / XYPOINTS / PEAK TABLE block.")
    if len(spectra) == 1 and not (spectra[0].metadata.get("title") or "").strip():
        spectra[0].name = _stem(filename)
    return spectra


# ----------------------------------------------------------------------------------------- tables
_SPLITTERS = {"\t": re.compile(r"\t"), ",": re.compile(r","), ";": re.compile(r";"), " ": re.compile(r"\s+")}


def _detect_delimiter(lines: list[str]) -> str:
    """The delimiter that splits the numeric rows most consistently: tab, semicolon, comma or whitespace."""
    sample = [line for line in lines if re.match(r"^\s*\"?[+\-−]?(?:\d|\.\d)", line)][:200]
    if not sample:
        sample = lines[:200]
    best, best_score = " ", -1.0
    for delimiter in ("\t", ";", ",", " "):
        counts = []
        numeric_fields = 0
        for line in sample:
            fields = [f for f in _SPLITTERS[delimiter].split(line.strip())]
            if delimiter == " ":
                fields = [f for f in fields if f]
            counts.append(len(fields))
            numeric_fields += sum(_is_number(f) for f in fields)
        if not counts:
            continue
        two_plus = sum(1 for c in counts if c >= 2) / len(counts)
        consistency = counts.count(max(set(counts), key=counts.count)) / len(counts)
        score = two_plus * 2 + consistency + numeric_fields / (sum(counts) or 1)
        if delimiter == "," and any(_DECIMAL_COMMA_RE.match(line.strip()) for line in sample):
            score -= 2           # "12,5" alone is a decimal comma, not two columns
        if score > best_score + 1e-9:
            best, best_score = delimiter, score
    return best


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return [f for f in re.split(r"\s+", line.strip()) if f]
    return [f.strip() for f in _SPLITTERS[delimiter].split(line.rstrip("\r\n"))]


def _row_values(fields: list[str]) -> list[float] | None:
    """Numbers of a data row (NaN for empty fields), or None if any filled field is not a number."""
    values = []
    filled = 0
    for f in fields:
        token = _unquote(f)
        if not token:
            values.append(math.nan)
            continue
        try:
            values.append(_number(token))
            filled += 1
        except ValueError:
            return None
    return values if filled >= 2 else None


def _read_table_rows(rows: list[list[str]], filename: str, sheet: str | None = None,
                     default_x_unit: str = "nm", default_y_mode: tuple[str, bool] | None = None) -> list[Spectrum]:
    """Spectra from a table (rows of fields): blocks of numeric rows, each with an optional header row above."""
    spectra: list[Spectrum] = []
    i, n = 0, len(rows)
    while i < n:
        values = _row_values(rows[i]) if rows[i] else None
        if values is None:
            i += 1
            continue
        width = len(rows[i])
        start = i
        block: list[list[float]] = []
        while i < n and len(rows[i]) == width:
            vals = _row_values(rows[i])
            if vals is None:
                break
            block.append(vals)
            i += 1
        if len(block) < 2:
            continue
        # The header row: the nearest non-numeric row above with the same number of (non-empty) fields.
        header = None
        title = None
        j = start - 1
        looked = 0
        while j >= 0 and looked < 6:
            fields = rows[j]
            if any(_unquote(f) for f in fields):
                filled = [_unquote(f) for f in fields]
                if header is None and (len(fields) == width or len([f for f in filled if f]) == width):
                    header = (filled + [""] * width)[:width]
                elif header is not None and title is None and len([f for f in filled if f]) >= 1:
                    title = filled
                    break
                elif header is None and len([f for f in filled if f]) == 1 and width == 2:
                    title = filled
                    break
                looked += 1
            j -= 1
        arr = np.asarray(block, dtype=float)
        spectra.extend(_table_block_spectra(arr, header, title, filename, sheet, len(spectra), default_x_unit, default_y_mode))
        if len(spectra) > _MAX_SPECTRA:
            raise UvVisParseError("The file contains too many spectra.")
    if not spectra:
        raise UvVisParseError("No table of wavelengths and values was found. " + _EXPORT_HELP)
    return spectra


def _paired_layout(header: list[str] | None, width: int) -> bool:
    """Cary-style x,y,x,y,… columns: an even width whose even-numbered headers all name the wavelength."""
    if header is None or width < 4 or width % 2:
        return False
    x_names = [header[k] for k in range(0, width, 2)]
    return all(_x_unit_from_text(name) is not None for name in x_names) and _x_unit_from_text(header[1]) is None


def _table_block_spectra(arr, header, title, filename, sheet, offset, default_x_unit, default_y_mode) -> list[Spectrum]:
    width = arr.shape[1]
    base = _stem(filename) if sheet is None else f"{_stem(filename)} [{sheet}]"
    spectra = []
    if _paired_layout(header, width):
        for pair in range(width // 2):
            xcol, ycol = 2 * pair, 2 * pair + 1
            x_unit = _x_unit_from_text(header[xcol]) or default_x_unit
            named = y_mode_from_text(header[ycol]) or default_y_mode
            sample = title[xcol] if title and xcol < len(title) and title[xcol] else f"{pair + 1}"
            meta = {"format": "paired-column table (Cary)", "x_unit": x_unit, "y_unit": header[ycol], "column_name": header[ycol]}
            if named:
                meta.update(y_mode=named[0], y_percent=named[1])
            spectra.append(_spectrum(filename, f"{base} [{sample}]" if width > 2 else base,
                                     arr[:, xcol], arr[:, ycol], meta, offset + len(spectra) + 1))
        return spectra
    x_name = header[0] if header else ""
    x_unit = _x_unit_from_text(x_name) or default_x_unit
    single = width == 2
    for col in range(1, width):
        y_name = header[col] if header else ""
        named = y_mode_from_text(y_name) or default_y_mode
        meta = {"format": "table", "x_unit": x_unit, "y_unit": y_name, "column_name": y_name}
        if title:
            meta["title"] = " ".join(t for t in title if t)
        if named:
            meta.update(y_mode=named[0], y_percent=named[1])
        if single:
            if title and len([t for t in title if t]) == 1 and re.search(r"\.spc|rawdata|smooth", " ".join(title), re.I):
                name = re.sub(r"\s*-\s*(?:RawData|SmoothedData|Data)\s*$", "", [t for t in title if t][0].replace(".spc", ""), flags=re.I)
                name = f"{base} [{name}]" if name and name != base else base
            else:
                name = base
        else:
            name = f"{base} [{y_name or f'Y{col}'}]"
        spectra.append(_spectrum(filename, name, arr[:, 0], arr[:, col], meta, offset + len(spectra) + 1))
    return spectra


def _read_table_text(text: str, filename: str, default_x_unit: str = "nm", default_y_mode=None) -> list[Spectrum]:
    lines = [line for line in text.splitlines()]
    delimiter = _detect_delimiter(lines)
    rows = [_split(line, delimiter) for line in lines]
    # Shimadzu/PerkinElmer put the column names in quotes; keep the quotes out of the names.
    return _read_table_rows(rows, filename, None, default_x_unit, default_y_mode)


def _read_xlsx(data: bytes, filename: str, default_x_unit: str = "nm", default_y_mode=None) -> list[Spectrum]:
    try:
        import openpyxl
    except ImportError as exc:
        raise UvVisParseError("Reading .xlsx needs the openpyxl package.") from exc
    try:
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:                                      # openpyxl raises many kinds
        raise UvVisParseError("The workbook could not be opened.") from exc
    per_sheet = []
    for sheet in book.sheetnames:
        rows = []
        for row in book[sheet].iter_rows(values_only=True):
            cells = ["" if v is None else (f"{v:.15g}" if isinstance(v, float) else str(v)) for v in row]
            while cells and cells[-1] == "":
                cells.pop()
            rows.append(cells)
        try:
            per_sheet.append((sheet, _read_table_rows(rows, filename, sheet, default_x_unit, default_y_mode)))
        except UvVisParseError:
            continue
    book.close()
    if not per_sheet:
        raise UvVisParseError("No sheet holds a table of wavelengths and values.")
    spectra = []
    for sheet, found in per_sheet:
        for spectrum in found:
            if len(per_sheet) == 1:
                spectrum.name = spectrum.name.replace(f" [{sheet}]", "", 1)
            spectrum.metadata["sheet"] = sheet
            spectrum.metadata["index"] = len(spectra) + 1
            spectra.append(spectrum)
    return spectra


# ----------------------------------------------------------------------------------------- entry point
def read_spectra(data: bytes, filename: str = "spectrum.csv", *, default_x_unit: str = "nm",
                 default_y_mode: tuple[str, bool] | None = None) -> list[Spectrum]:
    """Read every spectrum of a file, keeping stored values unchanged.

    ``default_x_unit`` is used when no header names the wavelength unit; ``default_y_mode``
    (mode, percent) when no header names the ordinate (otherwise the values are read, see
    ``guess_y_mode``).
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise UvVisParseError("The file is empty or is not byte data.")
    data = bytes(data)
    if default_x_unit not in ("nm", "um", "eV", "cm-1"):
        raise UvVisParseError("The default wavelength unit must be nm, um, eV or cm-1.")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".xlsx" or data.startswith(b"PK\x03\x04"):
            return _read_xlsx(data, filename, default_x_unit, default_y_mode)
        if suffix == ".xls":
            raise UvVisParseError("Legacy .xls workbooks are not read; save as .xlsx or CSV.")
        text = _text(data)
        if looks_like_jcamp(text):
            return _read_jcamp(text, filename)
        if looks_like_jasco(text):
            return _read_jasco(text, filename)
        return _read_table_text(text, filename, default_x_unit, default_y_mode)
    except UvVisParseError:
        raise
    except (ValueError, TypeError, OverflowError, KeyError, IndexError) as exc:
        raise UvVisParseError("The file layout could not be read. " + _EXPORT_HELP) from exc


def condition_summary(spectrum: Spectrum) -> str:
    """A one-line summary of the measurement conditions kept in the metadata (instrument, date, bandwidth, …)."""
    meta = spectrum.metadata
    parts = []
    if meta.get("instrument"):
        parts.append(str(meta["instrument"]))
    if meta.get("date"):
        parts.append(str(meta["date"]) + (" " + str(meta["time"]) if meta.get("time") else ""))
    conditions = meta.get("conditions") or {}
    for key in ("Bandwidth", "Band Width", "UV/Vis Bandwidth", "NIR Bandwidth", "Scanning Speed", "Scan Speed",
                "Response", "Light Source", "Light Source Change Wavelength", "Data Interval", "Accessory"):
        for name, value in conditions.items():
            if name.lower() == key.lower() and value:
                parts.append(f"{name} {value}")
                break
    unit = meta.get("y_unit") or ""
    if unit:
        parts.append(unit)
    if meta.get("x_unit") and meta["x_unit"] != "nm":
        parts.append(f"x in {meta['x_unit']} → nm")
    return " · ".join(parts)
