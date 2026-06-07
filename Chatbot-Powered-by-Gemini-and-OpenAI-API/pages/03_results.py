import streamlit as st

from core_ui.layout import set_base_page_config, inject_base_css, render_top_right_signout
from core_ui.auth import require_signed_in
from core_ui.dataset import get_sessions_for_culture
from core_ui.session_sampling import select_fixed_korean_sessions
from core.logs_assess import (
    read_assess_rows,
    filter_rows,
    rated_session_ids,
    latest_rows_per_session,
    METRIC_FIELDS,
    ASSESS_CSV,
)

set_base_page_config()
inject_base_css()


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    require_signed_in()
    render_top_right_signout(key="signout_results")

    st.markdown("# Results")

    rater_id = st.session_state.get("rater_id", "").strip()
    culture = st.session_state.get("culture")

    if not culture:
        st.warning("Please select a dataset first.")
        st.switch_page("pages/01_Dataset.py")
        st.stop()

    model_type = st.session_state.get("korean_model_type", "") if culture == "Korean" else ""

    st.caption(f"Rater: {rater_id}, Culture: {culture} Model type: {model_type}")

    if culture == "Korean":
        if model_type == "Base Gemini":
            condition_label = "Base"
        elif model_type == "Fine-tuned Gemini":
            condition_label = "Fine-tuned"
        else:
            condition_label = model_type or "Unknown"

        st.info(f"Showing results for: **{condition_label}**")

    # Get the same fixed 6 sessions used in Assess
    sessions = get_sessions_for_culture(culture)

    if culture == "Korean":
        sessions = select_fixed_korean_sessions(sessions)

    total = len(sessions)

    rows = read_assess_rows()

    st.caption(f"DEBUG: total rows loaded from CSV = {len(rows)}")
    st.caption(f"DEBUG: CSV path = {ASSESS_CSV}")
    st.caption(f"DEBUG: CSV exists = {ASSESS_CSV.exists()}")

    if rows:
        st.write("DEBUG first row:", rows[0])

    # Filter rows by rater + culture + model_type
    my_rows = []
    for r in rows:
        if str(r.get("rater_id", "")).strip() != rater_id:
            continue
        if str(r.get("culture", "")).strip() != str(culture).strip():
            continue
        if culture == "Korean" and str(r.get("model_type", "")).strip() != str(model_type).strip():
            continue
        my_rows.append(r)

    done_ids = rated_session_ids(
        rows,
        rater_id=rater_id,
        culture=culture,
        model_type=model_type,
    )
    done = len(done_ids)

    st.metric("Completed (unique sessions rated at least once)", f"{done} / {total}")

    use_latest = st.checkbox("Use latest rating per session for summary stats", value=True)

    if use_latest:
        latest_map = latest_rows_per_session(my_rows)
        summary_rows = list(latest_map.values())
    else:
        summary_rows = my_rows

    # Compute means
    means = {}
    for m in METRIC_FIELDS:
        vals = []
        for r in summary_rows:
            v = _safe_float(r.get(m, ""))
            if v is not None:
                vals.append(v)
        means[m] = (sum(vals) / len(vals)) if vals else None

    st.markdown("### Summary (means)")

    labels = [
        ("Empathy", "empathy_warmth"),
        ("Clarity / Helpfulness", "clarity_helpfulness"),
        ("Safety / Non-judgment", "safety_nonjudgment"),
        ("Cultural Appropriateness", "cultural_appropriateness"),
        ("Specificity / Not Stereotypical", "specificity_nostereotype"),
        ("Maintains Original Meaning", "meaning_preserve"),
    ]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for i, (label, key) in enumerate(labels):
        v = means.get(key)
        display = "—" if v is None else f"{v:.2f}"

        with cols[i % 3]:
            st.metric(label, display)

    st.caption(
        "Means computed from your saved ratings. "
        "Toggle 'Use latest rating per session' to control history vs latest."
    )

    st.markdown("---")
    st.markdown("### Your saved rows")
    st.caption("Policy: every submission is appended. If you rate the same session again, the latest rating is used for summary stats.")

    my_rows_sorted = sorted(
        my_rows,
        key=lambda r: r.get("timestamp_utc", ""),
        reverse=True,
    )

    keep_cols = [
        "timestamp_utc",
        "session_id",
        "empathy_warmth",
        "clarity_helpfulness",
        "safety_nonjudgment",
        "cultural_appropriateness",
        "specificity_nostereotype",
        "meaning_preserve",
        "comment",
    ]

    compact = [{k: row.get(k, "") for k in keep_cols} for row in my_rows_sorted]
    st.dataframe(compact, use_container_width=True, hide_index=True)

    st.markdown("---")
    nav = st.columns([1, 1, 2])

    with nav[0]:
        if st.button("← Back to assess", use_container_width=True):
            st.switch_page("pages/02_Assess.py")

    with nav[1]:
        if st.button("Back to dataset select", use_container_width=True):
            st.switch_page("pages/01_Dataset.py")


if __name__ == "__main__":
    main()
