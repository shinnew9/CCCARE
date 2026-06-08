from __future__ import annotations

import os
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
ASSESS_CSV = LOG_DIR / "assess_sessions.csv"


# 6 rating metrics only
METRIC_FIELDS = [
    "empathy_warmth",
    "clarity_helpfulness",
    "safety_nonjudgment",
    "cultural_appropriateness",
    "specificity_nostereotype",
    "meaning_preserve",
]


CSV_FIELDS = [
    "timestamp_utc",
    "email",
    "rater_id",
    "culture",
    "dataset_file",
    "session_id",
    "session_idx",
    *METRIC_FIELDS,
    "model_type",
    "comment",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(x) -> str:
    return str(x or "").strip()


def _norm_model_type(x) -> str:
    """
    Normalize model condition labels.

    Examples:
    - Base == Base Gemini
    - Fine-tuned == Fine-tuned Gemini
    """
    x = str(x or "").strip().lower()
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = " ".join(x.split())

    if x in ["base", "base gemini"]:
        return "base"

    if x in [
        "fine tuned",
        "finetuned",
        "fine tuned gemini",
        "finetuned gemini",
    ]:
        return "fine_tuned"

    return x


def same_model_type(row_model_type: str, target_model_type: str) -> bool:
    """
    Compare model condition labels robustly.

    If target_model_type is empty, do not filter by model type.
    """
    target = _norm_model_type(target_model_type)

    if not target:
        return True

    return _norm_model_type(row_model_type) == target


def _looks_like_model_type(x: str) -> bool:
    x = str(x or "").strip().lower()
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = " ".join(x.split())

    return x in {
        "base",
        "base gemini",
        "fine tuned",
        "finetuned",
        "fine tuned gemini",
        "finetuned gemini",
    }


def _canonical_model_type(x: str) -> str:
    normed = _norm_model_type(x)

    if normed == "base":
        return "Base Gemini"

    if normed == "fine_tuned":
        return "Fine-tuned Gemini"

    return str(x or "").strip()


def _infer_model_type_from_row(row: Dict) -> str:
    """
    Best-effort inference for old CSV rows that did not have model_type.

    This is only used during schema migration.
    """
    dataset_file = str(row.get("dataset_file", "") or "").lower()

    if "fine" in dataset_file or "tuned" in dataset_file or "finetuned" in dataset_file:
        return "Fine-tuned Gemini"

    if "base" in dataset_file:
        return "Base Gemini"

    return ""


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_csv_header():
    """
    Create CSV with current header if missing.

    If an older CSV exists, migrate it to the current schema.
    This fixes old files that were created before model_type existed.
    """
    ensure_log_dir()

    if not ASSESS_CSV.exists():
        with open(ASSESS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
        return

    with open(ASSESS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            old_header = next(reader)
        except StopIteration:
            old_header = []

    if old_header == CSV_FIELDS:
        return

    backup_path = ASSESS_CSV.with_name(
        f"{ASSESS_CSV.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ASSESS_CSV.suffix}"
    )
    shutil.copy2(ASSESS_CSV, backup_path)

    with open(ASSESS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_rows = [dict(row) for row in reader]

    migrated_rows = []

    for row in old_rows:
        new_row = {k: row.get(k, "") for k in CSV_FIELDS}

        old_comment = str(row.get("comment", "") or "").strip()
        extras = row.get(None, [])

        # Case:
        # Old header did not have model_type.
        # New code wrote model_type before comment.
        # csv.DictReader then reads:
        #   comment = "Base Gemini"
        #   None = ["actual comment"]
        if not new_row.get("model_type", ""):
            if _looks_like_model_type(old_comment):
                new_row["model_type"] = _canonical_model_type(old_comment)

                if isinstance(extras, list) and len(extras) > 0:
                    new_row["comment"] = str(extras[0] or "").strip()
                else:
                    new_row["comment"] = ""
            else:
                new_row["model_type"] = _infer_model_type_from_row(row)
                new_row["comment"] = old_comment

        # Make model_type consistent if it exists but uses short labels.
        new_row["model_type"] = _canonical_model_type(new_row.get("model_type", ""))

        migrated_rows.append(new_row)

    with open(ASSESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(migrated_rows)


def append_assessment_row(row: Dict):
    """
    Always append a new assessment row.
    """
    ensure_csv_header()

    safe_row = {k: row.get(k, "") for k in CSV_FIELDS}

    if not safe_row["timestamp_utc"]:
        safe_row["timestamp_utc"] = _now_utc_iso()

    safe_row["model_type"] = _canonical_model_type(safe_row.get("model_type", ""))

    with open(ASSESS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(safe_row)
        f.flush()
        os.fsync(f.fileno())


def read_assess_rows() -> List[Dict]:
    """
    Read all assessment rows.

    ensure_csv_header() is called first so old CSV files are migrated
    before filtering/progress calculation.
    """
    ensure_csv_header()

    if not ASSESS_CSV.exists():
        return []

    with open(ASSESS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def filter_rows(rows: List[Dict], *, rater_id: str, culture: str) -> List[Dict]:
    rater_id = _norm(rater_id)
    culture = _norm(culture)

    out = []
    for row in rows:
        if _norm(row.get("rater_id")) == rater_id and _norm(row.get("culture")) == culture:
            out.append(row)

    return out


def rated_session_ids(
    rows: List[Dict],
    rater_id: str,
    culture: str,
    model_type: str = "",
) -> Set[str]:
    """
    Return session_ids that have at least one saved rating
    for the given rater/culture/model condition.
    """
    rated: Set[str] = set()

    rater_id = _norm(rater_id)
    culture = _norm(culture)
    model_type = _norm(model_type)

    for row in rows:
        if _norm(row.get("rater_id")) != rater_id:
            continue

        if _norm(row.get("culture")) != culture:
            continue

        if model_type:
            if not same_model_type(row.get("model_type", ""), model_type):
                continue

        sid = _norm(row.get("session_id"))
        if sid:
            rated.add(sid)

    return rated


def latest_rows_per_session(rows: List[Dict]) -> Dict[str, Dict]:
    """
    Given rows already filtered to a single rater/culture/model condition,
    return latest row per session_id by timestamp_utc.
    """
    latest: Dict[str, Dict] = {}

    for row in rows:
        sid = _norm(row.get("session_id"))
        if not sid:
            continue

        ts = row.get("timestamp_utc", "") or ""

        if sid not in latest:
            latest[sid] = row
        else:
            if ts > (latest[sid].get("timestamp_utc", "") or ""):
                latest[sid] = row

    return latest


def compute_progress(
    total,
    rows: List[Dict],
    rater_id: str,
    culture: str,
    model_type: str = "",
):
    done = len(
        rated_session_ids(
            rows,
            rater_id=rater_id,
            culture=culture,
            model_type=model_type,
        )
    )
    return done, total


def last_culture_for_rater(rows: List[Dict], *, rater_id: str) -> str | None:
    """
    Return the most recent culture used by this rater_id based on timestamp_utc.
    """
    rater_id = _norm(rater_id)

    if not rater_id:
        return None

    latest_ts = ""
    latest_culture = None

    for row in rows:
        if _norm(row.get("rater_id")) != rater_id:
            continue

        ts = row.get("timestamp_utc", "") or ""

        if ts > latest_ts:
            latest_ts = ts
            latest_culture = _norm(row.get("culture"))

    return latest_culture or None