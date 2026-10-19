"""Load and combine requirement datasets for training.

Supported inputs:
- PROMISE NFR ARFF-style text (lines like ``1,'The system shall ...',PE``)
- CSV / XLSX with a text column and a label column. Column names are
  auto-detected (text: text/requirement/requirement text/sentence,
  label: category/label/type/class/classification). PROMISE, DOSSPRE and
  FR/NFR labels are mapped to our categories automatically.

Bundled processed datasets (``testcase_ai/model/data``):

=============================  =======  ==============================================
file                           quality  labels
=============================  =======  ==============================================
promise_requirements.csv       gold     12 categories (PROMISE NFR + extended)
dosspre_requirements.csv       gold     DOSSPRE 1.0 security/quality classes, mapped
handcrafted_requirements.csv   gold     all categories, story style
fr_nfr_requirements.csv        weak     functional / nonfunctional only
ai_classified_requirements.csv weak     labels assigned by LLMs (noisy; not used by default)
=============================  =======  ==============================================

"Weak" rows are only used for training, never for evaluation. Rebuild the
processed files from ``datasets/raw`` with ``python -m testcase_ai.model.datasets``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .taxonomy import CATEGORIES, EXTRA_LABELS, NONFUNCTIONAL, PROMISE_LABELS

DATA_DIR = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "datasets" / "raw"
STORIES = ROOT / "samples" / "drive_stories.json"

HANDCRAFTED = DATA_DIR / "handcrafted_requirements.csv"
PROMISE = DATA_DIR / "promise_requirements.csv"
DOSSPRE = DATA_DIR / "dosspre_requirements.csv"
FR_NFR = DATA_DIR / "fr_nfr_requirements.csv"
AI_CLASSIFIED = DATA_DIR / "ai_classified_requirements.csv"

GOLD_SOURCES = (PROMISE, DOSSPRE, HANDCRAFTED)
WEAK_SOURCES = (FR_NFR, AI_CLASSIFIED)
# Weak sources used by default. AI_CLASSIFIED lowered the scores in the ablation study
# (docs/experiments.md), so it is only used when asked for explicitly.
DEFAULT_WEAK = (FR_NFR,)

_TEXT_COLS = ("text", "requirement", "requirement text", "requirementtext", "requirement_text", "sentence",
              "description")
_LABEL_COLS = ("category", "label", "type", "class", "_class_", "classification", "requirement_type")
_PROMISE_LINE = re.compile(r"^\s*\d+\s*,\s*'(.*)'\s*,\s*([A-Za-z]+)\s*$")


class DatasetError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip("'\"").strip()


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def map_label(label: str) -> str | None:
    """Map a dataset label to one of our categories, NONFUNCTIONAL, or None (unusable)."""
    label = str(label).strip()
    low = label.lower()
    if low in CATEGORIES or low == NONFUNCTIONAL:
        return low
    if label.upper() in PROMISE_LABELS:
        return PROMISE_LABELS[label.upper()]
    return EXTRA_LABELS.get(low)


def load_promise_arff(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PROMISE_LINE.match(line)
        if m:
            rows.append((m.group(1).replace("\\'", "'"), m.group(2)))
    return _finalize(pd.DataFrame(rows, columns=["text", "label"]), source=Path(path).name)


def _pick_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = {str(c).lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in _TEXT_COLS if c in cols), None)
    label_col = next((cols[c] for c in _LABEL_COLS if c in cols), None)
    if text_col is None or label_col is None:
        raise DatasetError(
            f"{name}: need a text column ({', '.join(_TEXT_COLS)}) "
            f"and a label column ({', '.join(_LABEL_COLS)}); found {list(df.columns)}"
        )
    return df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]


def load_table(path: str | Path) -> pd.DataFrame:
    """CSV, or Excel (every sheet that has usable text + label columns)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        frames = []
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            try:
                frames.append(_pick_columns(df, f"{path.name}:{sheet}"))
            except DatasetError:
                continue
        if not frames:
            raise DatasetError(f"{path.name}: no sheet with a text column and a label column")
        df = pd.concat(frames, ignore_index=True)
    else:
        try:
            df = pd.read_csv(path, skipinitialspace=True)
        except UnicodeDecodeError:
            df = pd.read_csv(path, skipinitialspace=True, encoding="latin1")
        df = _pick_columns(df, path.name)
    return _finalize(df, source=path.name)


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".txt", ".arff"):
        return load_promise_arff(path)
    return load_table(path)


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(normalize_text)
    df["category"] = df["label"].map(map_label)
    df = df[df["category"].notna() & (df["text"].str.len() > 10)]
    df["source"] = source
    return df[["text", "category", "source"]]


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate datasets; later frames win when the same text appears twice."""
    df = pd.concat(frames, ignore_index=True)
    df["_key"] = df["text"].map(_dedupe_key)
    df = df.drop_duplicates("_key", keep="last").drop(columns="_key")
    return df.reset_index(drop=True)


def is_gold(df: pd.DataFrame) -> pd.Series:
    """Rows with a specific category from a trusted source (usable for evaluation)."""
    weak = {p.name for p in WEAK_SOURCES}
    return (df["category"] != NONFUNCTIONAL) & ~df["source"].isin(weak)


def load_training_data(extra: list[str | Path] | None = None, feedback: str | Path | None = None,
                       weak: tuple[Path, ...] = DEFAULT_WEAK) -> pd.DataFrame:
    # Weak sources first so gold labels win on duplicate texts.
    sources = list(weak) + list(GOLD_SOURCES)
    frames = [load_table(p) for p in sources if p.exists()]
    frames += [load_any(p) for p in extra or []]
    if feedback and Path(feedback).exists():
        frames.append(load_table(feedback))
    return combine(frames)


def load_unlabeled_criteria(path: Path = STORIES) -> list[str]:
    """Acceptance criteria of the real story corpus (no category labels)."""
    if not path.exists():
        return []
    stories = json.loads(path.read_text(encoding="utf-8"))
    return [line.strip() for s in stories for line in s["acceptance_criteria"].splitlines() if line.strip()]


# ------------------------------------------------------------------------ build from raw files

def _write(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    df[["text", "category"]].to_csv(out, index=False)
    return df


def build_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    built: dict[str, pd.DataFrame] = {}
    promise = [load_promise_arff(raw_dir / "PROMISE_exp.txt")] if (raw_dir / "PROMISE_exp.txt").exists() else []
    if (raw_dir / "software_requirements_extended.csv").exists():
        promise.append(load_table(raw_dir / "software_requirements_extended.csv"))
    if promise:
        built[PROMISE.name] = _write(combine(promise), PROMISE)

    if (raw_dir / "DOSSPRE_1.0Original.csv").exists():
        built[DOSSPRE.name] = _write(combine([load_table(raw_dir / "DOSSPRE_1.0Original.csv")]), DOSSPRE)

    if (raw_dir / "FR_NFR_Dataset_Original.xlsx").exists():
        built[FR_NFR.name] = _write(combine([load_table(raw_dir / "FR_NFR_Dataset_Original.xlsx")]), FR_NFR)

    if (raw_dir / "Classified_AI_Requirements.xlsx").exists():
        df = load_table(raw_dir / "Classified_AI_Requirements.xlsx")
        df = df[~df["text"].str.contains(r"^here (?:are|is)\b|requirements for", case=False, regex=True)]
        built[AI_CLASSIFIED.name] = _write(combine([df]), AI_CLASSIFIED)
    if not built:
        raise DatasetError(f"No raw datasets found in {raw_dir}")
    return built


if __name__ == "__main__":
    for name, df in build_all().items():
        print(f"{name}: {len(df)} rows  {df['category'].value_counts().to_dict()}")
