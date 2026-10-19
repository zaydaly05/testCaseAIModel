"""Load and combine labeled requirement datasets for training.

Supported inputs:
- PROMISE NFR ARFF-style text (lines like ``1,'The system shall ...',PE``)
- CSV / XLSX with a text column and a label column. Column names are
  auto-detected (text: text/requirement/requirementtext/sentence,
  label: category/label/type/class/_class_). PROMISE labels (F, SE, PE, ...)
  are mapped to our categories automatically.

Run ``python -m testcase_ai.model.datasets`` to rebuild the bundled
``promise_requirements.csv`` from the raw files in ``datasets/raw``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .taxonomy import CATEGORIES, PROMISE_LABELS

DATA_DIR = Path(__file__).parent / "data"
HANDCRAFTED = DATA_DIR / "handcrafted_requirements.csv"
PROMISE = DATA_DIR / "promise_requirements.csv"
RAW_DIR = Path(__file__).resolve().parents[2] / "datasets" / "raw"

_TEXT_COLS = ("text", "requirement", "requirementtext", "requirement_text", "sentence", "description")
_LABEL_COLS = ("category", "label", "type", "class", "_class_", "requirement_type")
_PROMISE_LINE = re.compile(r"^\s*\d+\s*,\s*'(.*)'\s*,\s*([A-Za-z]+)\s*$")


class DatasetError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def map_label(label: str) -> str | None:
    label = str(label).strip()
    if label.lower() in CATEGORIES:
        return label.lower()
    return PROMISE_LABELS.get(label.upper())


def load_promise_arff(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PROMISE_LINE.match(line)
        if m:
            rows.append((m.group(1).replace("\\'", "'"), m.group(2)))
    return _finalize(pd.DataFrame(rows, columns=["text", "label"]), source=Path(path).name)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, skipinitialspace=True)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in _TEXT_COLS if c in cols), None)
    label_col = next((cols[c] for c in _LABEL_COLS if c in cols), None)
    if text_col is None or label_col is None:
        raise DatasetError(
            f"{path.name}: need a text column ({', '.join(_TEXT_COLS)}) "
            f"and a label column ({', '.join(_LABEL_COLS)}); found {list(df.columns)}"
        )
    return _finalize(df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]], source=path.name)


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".txt", ".arff"):
        return load_promise_arff(path)
    return load_table(path)


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(normalize_text)
    df["category"] = df["label"].map(map_label)
    df = df[df["category"].notna() & (df["text"].str.len() > 3)]
    df["source"] = source
    return df[["text", "category", "source"]]


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate datasets; later frames win when the same text appears twice."""
    df = pd.concat(frames, ignore_index=True)
    df["_key"] = df["text"].map(_dedupe_key)
    df = df.drop_duplicates("_key", keep="last").drop(columns="_key")
    return df.reset_index(drop=True)


def load_training_data(extra: list[str | Path] | None = None, feedback: str | Path | None = None) -> pd.DataFrame:
    frames = [load_table(PROMISE), load_table(HANDCRAFTED)]
    frames += [load_any(p) for p in extra or []]
    if feedback and Path(feedback).exists():
        frames.append(load_table(feedback))
    return combine(frames)


def build_promise_csv(raw_dir: Path = RAW_DIR, out: Path = PROMISE) -> pd.DataFrame:
    frames = []
    if (raw_dir / "PROMISE_exp.txt").exists():
        frames.append(load_promise_arff(raw_dir / "PROMISE_exp.txt"))
    if (raw_dir / "software_requirements_extended.csv").exists():
        frames.append(load_table(raw_dir / "software_requirements_extended.csv"))
    if not frames:
        raise DatasetError(f"No raw datasets found in {raw_dir}")
    df = combine(frames)
    df[["text", "category"]].to_csv(out, index=False)
    return df


if __name__ == "__main__":
    built = build_promise_csv()
    print(f"Wrote {len(built)} requirements to {PROMISE}")
    print(built["category"].value_counts().to_string())
