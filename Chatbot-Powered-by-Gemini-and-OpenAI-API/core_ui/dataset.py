from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

DATASET_FILES = {
    "Chinese": ROOT / "data" / "psydial4" / "student_only_100.jsonl",
    "Hispanic": ROOT / "data" / "psydial4" / "student_only_rewrite_hispanic_college_grad_100.jsonl",
    "African American": ROOT / "data" / "psydial4" / "student_only_rewrite_african_american_college_grad_100.jsonl",
    "Korean": {
        "Base Gemini": ROOT / "data" / "korean" / "korean_base_app_15.json",
        "Fine-tuned Gemini": ROOT / "data" / "korean" / "korean_finetuned_app_15.json",
    },
    "Others": None,
}


def load_json(path: Path):
    if not path or not path.exists():
        st.error(f"Dataset file not found: {path}")
        st.stop()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    if not path or not path.exists():
        st.error(f"Dataset file not found: {path}")
        st.stop()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_session_psydial(raw: dict):
    sid = str(raw.get("session_id", raw.get("id", "unknown")))
    turns = raw.get("turns", [])
    norm = []

    for t in turns:
        role = (t.get("role") or "").lower().strip()
        text = t.get("text") or ""

        if not text:
            continue

        if role == "system":
            continue

        if role in ["user", "client", "patient", "seeker", "human"]:
            norm.append({"speaker": "client", "text": text})
        else:
            norm.append({"speaker": "counselor", "text": text})

    return {
        "session_id": sid,
        "turns": norm,
        "topic": raw.get("topic", ""),
        "psychotherapy": raw.get("psychotherapy", ""),
        "theme": raw.get("theme", ""),
        "reference_ko": raw.get("reference_ko", ""),
        "reference_en": raw.get("reference_en", ""),
    }


# Korean dataset is structured as a single text input, so we need logic to split it into turns based on speaker labels.
# Assumimng that the conversation looks like "Client: ... Counselor: ..." and splitting turns based on these speaker labels. 
# If there are lines without speaker labels, we will attach them to the previous turn. 
def parse_korean_input_to_turns(input_text: str):
    turns = []

    for line in input_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("내담자") or line.startswith("Client"):
            speaker = "client"
        elif line.startswith("상담사") or line.startswith("Counselor") or line.startswith("Therapist"):
            speaker = "counselor"
        else:
            # speaker label이 없으면 이전 turn에 붙이기
            if turns:
                turns[-1]["text"] += "\n" + line
                continue
            speaker = "client"

        turns.append({
            "speaker": speaker,
            "text": line,
        })

    return turns


def parse_session_korean_output(raw: dict):
    sid = str(raw.get("session_id", raw.get("dialog_id", raw.get("id", "unknown"))))
    turns = raw.get("turns", [])
    norm = []

    for t in turns:
        speaker = t.get("speaker", "client")
        text = str(t.get("text", ""))

        # remove leading indentation from every line
        text = "\n".join(line.strip() for line in text.splitlines()).strip()

        if text:
            norm.append({
                "speaker": speaker,
                "text": text,
            })

    return {
        "session_id": sid,
        "turns": norm,
        "topic": raw.get("topic", ""),
        "psychotherapy": raw.get("psychotherapy", ""),
        "theme": raw.get("theme", ""),
        "reference_ko": raw.get("reference_ko", ""),
        "reference_en": raw.get("reference_en", ""),
    }


def resolve_dataset_path(culture: str):
    ds_conf = DATASET_FILES.get(culture)

    if culture == "Korean" and isinstance(ds_conf, dict):
        model_type = st.session_state.get("korean_model_type", "Base Gemini")
        return Path(ds_conf[model_type])

    if ds_conf:
        return Path(ds_conf)

    return None


def _load_json(path: Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_sessions(data: Any) -> List[Dict]:
    """
    Support common JSON structures:
    1. [session, session, ...]
    2. {"sessions": [...]}
    3. {"data": [...]}
    4. {"conversations": [...]}
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["sessions", "data", "conversations"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        "Unsupported dataset JSON format. Expected a list or a dict containing "
        "'sessions', 'data', or 'conversations'."
    )


def get_dataset_file_for_culture(culture: str, model_type: Optional[str] = None) -> Path:
    """
    Return the dataset path for a culture.

    For Korean, model_type must distinguish Base Gemini vs Fine-tuned Gemini.
    """
    conf = DATASET_FILES.get(culture)

    if conf is None:
        raise ValueError(f"No dataset configured for culture: {culture}")

    if culture == "Korean":
        if not isinstance(conf, dict):
            raise ValueError(
                "Korean dataset config must be a dict with Base Gemini and Fine-tuned Gemini files."
            )

        model_type = model_type or "Base Gemini"

        if model_type not in conf:
            raise ValueError(
                f"No Korean dataset configured for model_type={model_type}. "
                f"Available: {list(conf.keys())}"
            )

        return Path(conf[model_type])

    if isinstance(conf, dict):
        raise ValueError(
            f"Dataset config for {culture} is a dict, but no model_type was provided."
        )

    return Path(conf)


def get_sessions_for_culture(culture: str, model_type: Optional[str] = None) -> List[Dict]:
    """
    Load sessions for the selected culture/model condition.
    """
    dataset_file = get_dataset_file_for_culture(culture, model_type=model_type)
    data = _load_json(dataset_file)
    sessions = _extract_sessions(data)

    return sessions