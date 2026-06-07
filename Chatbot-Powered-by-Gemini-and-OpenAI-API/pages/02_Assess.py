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


def scroll_to_top():
    st.components.v1.html(
        """
        <script>
        function goTop() {
          const el = window.parent.document.getElementById("TOP");
          if (el) { el.scrollIntoView({behavior: "instant", block: "start"}); }
          else { window.parent.scrollTo(0, 0); }
        }
        goTop();
        setTimeout(goTop, 50);
        </script>
        """,
        height=0,
    )


def _session_num_messages(session: dict) -> int:
    """
    Count how many dialogue messages/turns exist in a session.
    This is intentionally flexible because dataset structures may differ.
    """
    for key in ["messages", "conversation", "dialogue", "turns"]:
        value = session.get(key)

        if isinstance(value, list):
            return len(value)

        if isinstance(value, str):
            # fallback: count rough speaker markers if conversation is stored as text
            return value.count("Client") + value.count("Counselor") + value.count("Therapist")

    return 0


def _get_sessions(culture: str):
    model_type = st.session_state.get("korean_model_type", "") if culture == "Korean" else ""

    # IMPORTANT:
    # model_type must be included in cache key so Base and Fine-tuned load different JSON contents.
    # The selected session IDs remain the same because select_fixed_korean_sessions uses a fixed seed.
    cache_key = f"_sessions_cache_{culture}_{model_type}_fixed_6"

    cached = st.session_state.get(cache_key)
    if cached and isinstance(cached, list) and len(cached) > 0:
        return cached

    sessions = get_sessions_for_culture(culture)

    if culture == "Korean":
        sessions = select_fixed_korean_sessions(sessions)

    st.session_state[cache_key] = sessions
    return sessions



def _find_next_unrated_index(sessions, rated_ids_set):
    for i, s in enumerate(sessions):
        sid = str(s.get("session_id", "")).strip()
        if sid and sid not in rated_ids_set:
            return i
    return None


def _ensure_resume_pointer(sessions, rater_id: int, culture: str, model_type: str = ""):
    """
    Only initialize the pointer when session_idx does not exist.
    Do NOT automatically move away from the current session after saving.
    """
    cur_idx = st.session_state.get("session_idx", None)

    # 이미 사용자가 어느 세션에 있으면 그대로 둔다.
    if cur_idx is not None:
        try:
            cur_idx = int(cur_idx)
        except Exception:
            cur_idx = 0

        cur_idx = max(0, min(cur_idx, len(sessions) - 1))
        st.session_state["session_idx"] = cur_idx
        return

    # 처음 들어왔을 때만 next unrated로 시작한다.
    all_rows = read_assess_rows()
    rated_ids_set = rated_session_ids(
        all_rows,
        rater_id=rater_id,
        culture=culture,
        model_type=model_type,
    )
    nxt = _find_next_unrated_index(sessions, rated_ids_set)
    st.session_state["session_idx"] = nxt if nxt is not None else 0


def main():
    require_signed_in()
    render_top_right_signout(key="signout_assess")

    st.markdown('<div id="TOP"></div>', unsafe_allow_html=True)

    # consume scroll flag exactly once per render
    if st.session_state.pop("_scroll_top", False):
        scroll_to_top()

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
    total = len(sessions)

    model_type = st.session_state.get("korean_model_type", "") if culture == "Korean" else ""

    # Resume logic (based on CSV)
    _ensure_resume_pointer(sessions, rater_id=rater_id, culture=culture, model_type=model_type)

    # Progress UI
    all_rows = read_assess_rows()
    done, total = compute_progress(
        total,
        all_rows,
        rater_id=rater_id,
        culture=culture,
        model_type=model_type,
    )

    st.markdown("## Conversation Assess")
    if culture == "Korean":
        model_label = st.session_state.get("korean_model_type", "Base Gemini")

        if model_label == "Base Gemini":
            condition_label = "Base"
        elif model_label == "Fine-tuned Gemini":
            condition_label = "Fine-tuned"
        else:
            condition_label = model_label

        st.info(
            f"**Current condition: {condition_label}**  \n"
            f"Progress: **{done}/{total}** completed."
        )
    else:
        st.caption(f"Dataset: {culture} • Progress: {done}/{total} completed")



    # Controls: Resume / Start Over
    ctrl = st.columns([1.2, 1.2, 3])
    with ctrl[0]:
        if st.button("Resume next unrated", use_container_width=True):
            rated_ids_set = rated_session_ids(all_rows, rater_id=rater_id, culture=culture)
            nxt = _find_next_unrated_index(sessions, rated_ids_set)
            if nxt is not None:
                st.session_state["_scroll_top"] = True
                st.session_state["session_idx"] = nxt
                st.rerun()
            else:
                st.toast("All sessions have been rated at least once.", icon="✅")
    with ctrl[1]:
        if st.button("Start from first", use_container_width=True):
            st.session_state["_scroll_top"] = True
            st.session_state["session_idx"] = 0
            st.rerun()

    # Current session
    idx = int(st.session_state.get("session_idx", 0) or 0)
    idx = max(0, min(idx, len(sessions) - 1))
    st.session_state["session_idx"] = idx
    session = sessions[idx]
    sid = str(session.get("session_id", "")).strip()

    st.markdown("---")
    st.subheader(f"Session {idx + 1} / {len(sessions)}")

    psychotherapy = session.get("psychotherapy", "")

    st.markdown(f"""
    <span style="font-size: 0.85rem; opacity: 0.7;">
    Session ID: {sid}
    </span>

    <span style="background-color:#2a2f3a; padding:4px 10px; border-radius:10px; font-size:0.8rem; margin-left:6px;">
    {psychotherapy}
    </span>
    </div>
    """, unsafe_allow_html=True)



    # Chat (left/right bubbles)
    render_chat(session.get("turns", []), culture=culture)

    st.markdown("---")

    # If already rated, show info + last rating preview (latest row)
    filtered = [
        r for r in all_rows
        if r.get("rater_id", "").strip() == rater_id
        and r.get("culture", "").strip() == culture
        and (
            culture != "Korean"
            or r.get("model_type", "").strip() == model_type
        )
    ]
    latest_map = latest_rows_per_session(filtered)
    already = sid in latest_map
    if already:
        st.info("This session has been rated before (history is preserved). You can rate again; a new row will be appended.")
        last = latest_map[sid]
        with st.expander("Show latest saved rating for this session"):
            st.write({k: last.get(k, "") for k in ["timestamp_utc", *METRIC_FIELDS, "comment"]})

    # Rating form
    st.markdown("### Rate this conversation (1–5)")

    st.markdown("""
        Please evaluate the counseling response using the six dimensions below.

        **Scale:**  
        1 = Very poor  
        2 = Poor  
        3 = Acceptable  
        4 = Good  
        5 = Excellent
        """)

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

        model_type = st.session_state.get("korean_model_type", "") if culture == "Korean" else ""

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

                "comment": comment.strip(),
                "model_type": model_type,
            }

            append_assessment_row(row)
            st.success("Your answer has been saved! You can click **Next →** below when you are ready.")

            # Stay on the current session.
            # Do not auto-scroll.
            # Do not auto-move to the next session.
            st.session_state["session_idx"] = idx

            # DEBUG: 저장 확인용
            debug_rows = read_assess_rows()
            st.success(f"DEBUG: Saved locally. Current saved rows: {len(debug_rows)}")
            st.code(str(ASSESS_CSV) if "ASSESS_CSV" in globals() else "ASSESS_CSV not imported")

            append_assessment_row(row)

            all_rows = read_assess_rows()
            rated_ids_set = rated_session_ids(all_rows, rater_id=rater_id, culture=culture, model_type=model_type)
            nxt = _find_next_unrated_index(sessions, rated_ids_set)

            st.session_state["_scroll_top"] = True

            if nxt is None:
                st.toast("Saved! All sessions rated at least once.", icon="✅")
                st.session_state["session_idx"] = idx
            else:
                st.toast("Saved! Moving to next unrated session…", icon="✅")
                st.session_state["session_idx"] = nxt
                st.rerun()


    # Navigation (manual)
    st.markdown("---")
    nav = st.columns([1, 1, 2, 2])
    with nav[0]:
        if st.button("← Previous", disabled=(idx <= 0), use_container_width=True):
            st.session_state["_scroll_top"] = True
            st.session_state["session_idx"] = idx - 1
            st.rerun()
    with nav[1]:
        if st.button("Next →", disabled=(idx >= len(sessions) - 1), use_container_width=True):
            st.session_state["_scroll_top"] = True
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
