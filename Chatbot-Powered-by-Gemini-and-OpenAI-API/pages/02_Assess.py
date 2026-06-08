import streamlit as st

from core_ui.layout import set_base_page_config, inject_base_css, render_top_right_signout
from core_ui.auth import require_signed_in
from core_ui.dataset import get_sessions_for_culture, DATASET_FILES
from core_ui.session_sampling import select_fixed_korean_sessions
from core_ui.chat_view import render_chat

from core.logs_assess import (
    append_assessment_row,
    read_assess_rows,
    rated_session_ids,
    compute_progress,
    latest_rows_per_session,
    METRIC_FIELDS,
    ASSESS_CSV,
)

set_base_page_config()
inject_base_css()


def _get_sessions(culture: str):
    model_type = st.session_state.get("korean_model_type", "") if culture == "Korean" else ""

    cache_key = f"_sessions_cache_{culture}_{model_type}_fixed_6"

    cached = st.session_state.get(cache_key)
    if cached and isinstance(cached, list) and len(cached) > 0:
        return cached

    if culture == "Korean":
        sessions = get_sessions_for_culture(culture, model_type=model_type)
        sessions = select_fixed_korean_sessions(sessions)
    else:
        sessions = get_sessions_for_culture(culture)

    st.session_state[cache_key] = sessions
    return sessions


def _condition_label(model_type: str) -> str:
    if model_type == "Base Gemini":
        return "Base"
    if model_type == "Fine-tuned Gemini":
        return "Fine-tuned"
    return model_type or "Unknown"


def main():
    require_signed_in()
    render_top_right_signout(key="signout_assess")

    culture = st.session_state.get("culture")
    if not culture:
        st.warning("Please select a dataset first.")
        st.switch_page("pages/01_Dataset.py")
        st.stop()

    rater_id = st.session_state.get("rater_id", "").strip()
    email = st.session_state.get("email", "").strip()

    ds_conf = DATASET_FILES.get(culture) or ""

    if culture == "Korean" and isinstance(ds_conf, dict):
        model_type = st.session_state.get("korean_model_type", "Base Gemini")
        ds_file = str(ds_conf.get(model_type, ""))
    else:
        model_type = ""
        ds_file = str(ds_conf)

    sessions = _get_sessions(culture)
    st.caption(f"DEBUG model_type = {model_type}")
    st.caption(f"DEBUG dataset_file = {ds_file}")
    st.caption(f"DEBUG first session id = {sessions[0].get('session_id', '') if sessions else 'NO SESSION'}")
    st.write("DEBUG first session sample:", sessions[0].get("turns", [])[:2] if sessions else [])

    if sessions:
        first_turns = sessions[0].get("turns", [])
        st.caption(f"DEBUG first session turns count = {len(first_turns)}")
        st.write("DEBUG first session sample:", first_turns[:2])

        total = len(sessions)

    if total == 0:
        st.error("No sessions found for this dataset.")
        st.stop()

    # Do not auto-resume during data collection.
    # Keep session_idx simple and stable.
    if "session_idx" not in st.session_state:
        st.session_state["session_idx"] = 0

    idx = int(st.session_state.get("session_idx", 0) or 0)
    idx = max(0, min(idx, total - 1))
    st.session_state["session_idx"] = idx

    session = sessions[idx]
    sid = str(session.get("session_id", "")).strip()

    all_rows = read_assess_rows()
    done, total = compute_progress(
        total,
        all_rows,
        rater_id=rater_id,
        culture=culture,
        model_type=model_type,
    )

    # Flash message after save + rerun
    if st.session_state.pop("_rating_saved_flash", False):
        st.success("✅ Your answer has been saved! Moving to the next session.")

    st.markdown("## Conversation Assess")

    if culture == "Korean":
        condition_label = _condition_label(model_type)
        st.info(
            f"**Current condition: {condition_label}**  \n"
            f"Progress: **{done}/{total}** completed."
        )
    else:
        st.caption(f"Dataset: {culture} • Progress: {done}/{total} completed")

    st.caption(f"DEBUG: CSV path = {ASSESS_CSV}")
    st.caption(f"DEBUG: CSV exists = {ASSESS_CSV.exists()}")
    st.caption(f"DEBUG: total rows currently loaded = {len(all_rows)}")

    st.markdown("---")
    st.subheader(f"Session {idx + 1} / {len(sessions)}")

    psychotherapy = session.get("psychotherapy", "")

    st.markdown(
        f"""
        <span style="font-size: 0.85rem; opacity: 0.7;">
        Session ID: {sid}
        </span>

        <span style="background-color:#2a2f3a; padding:4px 10px; border-radius:10px; font-size:0.8rem; margin-left:6px;">
        {psychotherapy}
        </span>
        """,
        unsafe_allow_html=True,
    )

    render_chat(session.get("turns", []), culture=culture)

    st.markdown("---")

    # Check whether this exact session has been rated for this rater/culture/model_type
    filtered = [
        r for r in all_rows
        if str(r.get("rater_id", "")).strip() == rater_id
        and str(r.get("culture", "")).strip() == str(culture).strip()
        and (
            culture != "Korean"
            or str(r.get("model_type", "")).strip() == str(model_type).strip()
        )
    ]

    latest_map = latest_rows_per_session(filtered)
    already = sid in latest_map

    if already:
        st.info(
            "This session has been rated before. "
            "If you save again, a new row will be appended and the latest rating will be used for summary."
        )
        last = latest_map[sid]
        with st.expander("Show latest saved rating for this session"):
            st.write({k: last.get(k, "") for k in ["timestamp_utc", *METRIC_FIELDS, "model_type", "comment"]})

    st.markdown("### Rate this conversation (1–5)")

    st.markdown(
        """
        Please evaluate the counseling response using the six dimensions below.

        **Scale:**  
        1 = Very poor  
        2 = Poor  
        3 = Acceptable  
        4 = Good  
        5 = Excellent
        """
    )

    safe_model_type = str(model_type or "default").replace(" ", "_").replace("-", "_")
    form_key_suffix = f"{culture}_{safe_model_type}_{sid}_{idx}"

    EVAL_QUESTIONS = [
        {
            "field": "empathy_warmth",
            "label": "1. Empathy",
            "question": "This response shows understanding, care, and emotional support for the client.",
            "key_prefix": "empathy",
        },
        {
            "field": "clarity_helpfulness",
            "label": "2. Clarity / Helpfulness",
            "question": "This response is clear, understandable, and practically helpful.",
            "key_prefix": "clarity",
        },
        {
            "field": "safety_nonjudgment",
            "label": "3. Safety / Non-judgment",
            "question": "This response avoids judgment, blame, harmful advice, or pressure.",
            "key_prefix": "safety",
        },
        {
            "field": "cultural_appropriateness",
            "label": "4. Cultural Appropriateness",
            "question": "This response feels appropriate for Korean-speaking clients and their cultural context.",
            "key_prefix": "cultural",
        },
        {
            "field": "specificity_nostereotype",
            "label": "5. Specificity / Not Stereotypical",
            "question": "This response is specific to the client’s situation and does not sound generic or stereotypical.",
            "key_prefix": "specificity",
        },
        {
            "field": "meaning_preserve",
            "label": "6. Maintains Original Meaning",
            "question": "This Korean response preserves the intended meaning of the original counseling response.",
            "key_prefix": "meaning",
        },
    ]

    with st.form(f"rating_form_{form_key_suffix}", clear_on_submit=False):
        scores = {}

        for q in EVAL_QUESTIONS:
            st.markdown(f"**{q['label']}**")
            st.caption(q["question"])

            scores[q["field"]] = st.radio(
                label=q["question"],
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True,
                key=f"{q['key_prefix']}_{form_key_suffix}",
                label_visibility="collapsed",
            )

            st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

        comment = st.text_area(
            "Optional comment",
            placeholder="Please share any additional thoughts about this response.",
            height=90,
            key=f"comment_{form_key_suffix}",
        )

        submit = st.form_submit_button("Save rating")

        if submit:
            row = {
                "timestamp_utc": "",
                "email": email,
                "rater_id": rater_id,
                "culture": culture,
                "dataset_file": ds_file,
                "session_id": sid,
                "session_idx": str(idx),

                "empathy_warmth": str(scores["empathy_warmth"]),
                "clarity_helpfulness": str(scores["clarity_helpfulness"]),
                "safety_nonjudgment": str(scores["safety_nonjudgment"]),
                "cultural_appropriateness": str(scores["cultural_appropriateness"]),
                "specificity_nostereotype": str(scores["specificity_nostereotype"]),
                "meaning_preserve": str(scores["meaning_preserve"]),

                "model_type": model_type,
                "comment": comment.strip(),
            }

            append_assessment_row(row)

            # Save 성공 메시지를 다음 render에서 보여주기
            st.session_state["_rating_saved_flash"] = True

            # Save rating을 누른 경우에만 다음 session으로 이동
            if idx < len(sessions) - 1:
                st.session_state["session_idx"] = idx + 1
            else:
                st.session_state["session_idx"] = idx

            st.rerun()

    st.markdown("---")

    nav = st.columns([1, 1, 2, 2])

    with nav[0]:
        if st.button("← Previous", disabled=(idx <= 0), use_container_width=True):
            st.session_state["session_idx"] = idx - 1
            st.rerun()

    with nav[1]:
        if st.button("Next →", disabled=(idx >= len(sessions) - 1), use_container_width=True):
            # Next는 저장 없이 이동만 함
            st.session_state["session_idx"] = idx + 1
            st.rerun()

    with nav[2]:
        if st.button("Back to dataset select", use_container_width=True):
            st.switch_page("pages/01_Dataset.py")

    with nav[3]:
        if st.button("Go to results →", use_container_width=True):
            st.switch_page("pages/03_results.py")


if __name__ == "__main__":
    main()