from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

APP_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = APP_ROOT / "logs"
ASSESS_CSV = LOG_DIR / "assess_sessions.csv"


# 6 metrics (session-level)
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


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_csv_header():
    """Create CSV with header if missing."""
    ensure_log_dir()
    if not ASSESS_CSV.exists():
        with open(ASSESS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()


def append_assessment_row(row: Dict):
    """
    Policy B: always append (history-preserving).
    """
    ensure_csv_header()

    safe_row = {k: row.get(k, "") for k in CSV_FIELDS}

    if not safe_row["timestamp_utc"]:
        safe_row["timestamp_utc"] = _now_utc_iso()

    with open(ASSESS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerow(safe_row)
        f.flush()
        os.fsync(f.fileno())


def read_assess_rows() -> List[Dict]:
    if not ASSESS_CSV.exists():
        return []
    with open(ASSESS_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def filter_rows(rows: List[Dict], *, rater_id: str, culture: str) -> List[Dict]:
    rater_id = (rater_id or "").strip()
    culture = (culture or "").strip()
    out = []
    for row in rows:
        if row.get("rater_id", "").strip() == rater_id and row.get("culture", "").strip() == culture:
            out.append(row)
    return out


def _norm(x) -> str:
    return str(x or "").strip()

def _norm_model_type(x) -> str:
    x = str(x or "").strip().lower()
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = " ".join(x.split())

    if x in ["base", "base gemini"]:
        return "base"
    if x in ["fine tuned", "finetuned", "fine tuned gemini", "fine tuned gemini", "fine-tuned gemini"]:
        return "fine_tuned"
    return x


def same_model_type(row_model_type: str, target_model_type: str) -> bool:
    """
    Accept aliases such as:
    Base == Base Gemini
    Fine-tuned == Fine-tuned Gemini
    """
    target = _norm_model_type(target_model_type)

    # If no target condition is selected, don't filter by model_type.
    if not target:
        return True

    return _norm_model_type(row_model_type) == target


def rated_session_ids(rows, rater_id: str, culture: str, model_type: str = ""):
    rated = set()

    rater_id = _norm(rater_id)
    culture = _norm(culture)
    model_type = _norm(model_type)

    for r in rows:
        if _norm(r.get("rater_id")) != rater_id:
            continue
        if _norm(r.get("culture")) != culture:
            continue

        # model_type이 명시된 경우에만 Base/Fine-tuned 구분
        if model_type:
            if _norm(r.get("model_type")) != model_type:
                continue

        sid = _norm(r.get("session_id"))
        if sid:
            rated.add(sid)

    return rated


def latest_rows_per_session(rows: List[Dict]) -> Dict[str, Dict]:
    """
    Given rows already filtered to a single (rater_id, culture),
    return latest row per session_id by timestamp_utc.
    """
    latest: Dict[str, Dict] = {}
    for row in rows:
        sid = str(row.get("session_id", "")).strip()
        if not sid:
            continue
        ts = row.get("timestamp_utc", "")
        if sid not in latest:
            latest[sid] = row
        else:
            # ISO string compare works if consistent isoformat; otherwise fallback
            if ts > (latest[sid].get("timestamp_utc", "") or ""):
                latest[sid] = row
    return latest


def compute_progress(total, rows, rater_id: str, culture: str, model_type: str = ""):
    done = len(rated_session_ids(rows, rater_id, culture, model_type))
    return done, total


def last_culture_for_rater(rows: List[Dict], *, rater_id: str) -> str | None:
    """Return the most recent culture used by this rater_id based on timestamp_utc."""
    rater_id = (rater_id or "").strip()
    if not rater_id:
        return None

    latest_ts = ""
    latest_culture = None
    for row in rows:
        if row.get("rater_id", "").strip() != rater_id:
            continue
        ts = row.get("timestamp_utc", "") or ""
        if ts > latest_ts:
            latest_ts = ts
            latest_culture = (row.get("culture", "") or "").strip()

    return latest_culture or None
