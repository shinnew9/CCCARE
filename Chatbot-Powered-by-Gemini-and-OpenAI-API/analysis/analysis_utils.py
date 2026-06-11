from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# 1. Load and combine CSVs
def load_all_csvs(data_dir, pattern="*.csv"):
    """
    Load all CSV files from a folder and combine them into one dataframe.

    Parameters
    ----------
    data_dir : str or Path
        Folder path containing participant CSV files.
    pattern : str
        File pattern. Default is "*.csv".

    Returns
    -------
    pd.DataFrame
        Combined dataframe with a source_file column.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Failed to read {file.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Loaded {len(csv_files)} CSV files")
    print(f"[OK] Combined shape: {combined.shape}")

    return combined


# 2. Column Name Standardization
def standardize_columns(df, column_map=None):
    """
    Standardize column names.

    Example column_map:
    {
        "email": "participant_id",
        "condition_name": "condition",
        "session": "session_id"
    }
    """
    df = df.copy()

    # Basic cleanup
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    if column_map:
        df = df.rename(columns=column_map)
    return df

column_map = {
    "email": "participant_id",
    "model_condition": "condition",
    "session": "session_id"
}

# df = standardize_columns(df, column_map)


# 3. 6-dimensiona automatic searching and rating column detection
def get_rating_columns(df, rating_cols=None):
    """
    Return rating columns.

    If rating_cols is given, use it.
    Otherwise, try to detect numeric Likert-scale columns.
    """
    if rating_cols is not None:
        missing = [col for col in rating_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing rating columns: {missing}")
        return rating_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude common non-rating numeric columns
    exclude_keywords = ["timestamp", "time", "index", "id", "session"]
    rating_cols = [
        col for col in numeric_cols
        if not any(key in col.lower() for key in exclude_keywords)
    ]

    print("[Detected rating columns]")
    for col in rating_cols:
        print("-", col)

    return rating_cols

rating_cols = [
    "empathy",
    "naturalness",
    "cultural_appropriateness",
    "relevance",
    "helpfulness",
    "overall_quality"
]


# 4. Statistical analysis
def check_data_overview(df, participant_col="participant_id", condition_col="condition", session_col="session_id"):
    """
    Print basic overview of the evaluation data.
    """
    print("========== DATA OVERVIEW ==========")
    print(f"Shape: {df.shape}")
    print()

    if participant_col in df.columns:
        print(f"Unique participants: {df[participant_col].nunique()}")
        print(df[participant_col].value_counts().sort_index())
        print()

    if condition_col in df.columns:
        print("Condition counts:")
        print(df[condition_col].value_counts(dropna=False))
        print()

    if session_col in df.columns:
        print("Session counts:")
        print(df[session_col].value_counts(dropna=False).sort_index())
        print()

    print("Missing values:")
    print(df.isna().sum())


# 5. particiant-level, checking whether it's all done
def participant_completion_table(
    df,
    participant_col="participant_id",
    session_col="session_id",
    condition_col=None
):
    """
    Create a completion table by participant.

    If condition_col is provided, count sessions by participant and condition.
    """
    if condition_col and condition_col in df.columns:
        completion = (
            df.groupby([participant_col, condition_col])[session_col]
            .nunique()
            .reset_index(name="n_sessions_completed")
        )
    else:
        completion = (
            df.groupby(participant_col)[session_col]
            .nunique()
            .reset_index(name="n_sessions_completed")
        )

    return completion.sort_values(by=participant_col)

# completion = participant_completion_table(df)
# display(completion)


# 6. Dimensional analysis: t-test, ANOVA, etc.
def descriptive_by_condition(
    df,
    rating_cols,
    condition_col="condition"
):
    """
    Compute mean, std, median, and count for each rating dimension by condition.
    """
    results = []

    for col in rating_cols:
        temp = (
            df.groupby(condition_col)[col]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        temp.insert(0, "dimension", col)
        results.append(temp)

    result_df = pd.concat(results, ignore_index=True)
    return result_df

# desc = descriptive_by_condition(df, rating_cols)
# display(desc)


# 7. Total Average Score Analysis
def add_average_human_score(df, rating_cols, new_col="human_avg_score"):
    """
    Add average human rating score across selected dimensions.
    """
    df = df.copy()
    df[new_col] = df[rating_cols].mean(axis=1, skipna=True)
    return df

df = add_average_human_score(df, rating_cols)


# 8. Base vs Fine-tuned Model Comparison
def compare_conditions_independent(
    df,
    rating_cols,
    condition_col="condition",
    condition_a="base",
    condition_b="fine_tuned"
):
    """
    Compare two conditions using independent t-test and Mann-Whitney U test.

    Use this when responses are not clearly paired.
    """
    results = []

    for col in rating_cols:
        a = df[df[condition_col] == condition_a][col].dropna()
        b = df[df[condition_col] == condition_b][col].dropna()

        if len(a) < 2 or len(b) < 2:
            results.append({
                "dimension": col,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": a.mean() if len(a) else np.nan,
                "mean_b": b.mean() if len(b) else np.nan,
                "mean_diff_b_minus_a": np.nan,
                "t_pvalue": np.nan,
                "mw_pvalue": np.nan,
                "cohens_d": np.nan
            })
            continue

        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")

        pooled_sd = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
        cohens_d = (b.mean() - a.mean()) / pooled_sd if pooled_sd != 0 else np.nan

        results.append({
            "dimension": col,
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "mean_diff_b_minus_a": b.mean() - a.mean(),
            "t_stat": t_stat,
            "t_pvalue": t_p,
            "mw_stat": mw_stat,
            "mw_pvalue": mw_p,
            "cohens_d": cohens_d
        })

    return pd.DataFrame(results)

comparison = compare_conditions_independent(
    df,
    rating_cols,
    condition_col="condition",
    condition_a="Base",
    condition_b="Fine-tuned"
)

# display(comparison)


# 9. Paired Comparison (if applicable)
def compare_conditions_paired(
    df,
    rating_cols,
    participant_col="participant_id",
    session_col="session_id",
    condition_col="condition",
    condition_a="base",
    condition_b="fine_tuned"
):
    """
    Compare two conditions using paired t-test and Wilcoxon signed-rank test.

    This assumes the same participant evaluated both conditions
    for the same session.
    """
    results = []

    id_cols = [participant_col, session_col]

    for col in rating_cols:
        wide = df.pivot_table(
            index=id_cols,
            columns=condition_col,
            values=col,
            aggfunc="mean"
        ).reset_index()

        if condition_a not in wide.columns or condition_b not in wide.columns:
            raise ValueError(f"Condition names not found in data: {condition_a}, {condition_b}")

        paired = wide[[condition_a, condition_b]].dropna()
        a = paired[condition_a]
        b = paired[condition_b]
        diff = b - a

        if len(diff) < 2:
            results.append({
                "dimension": col,
                "n_pairs": len(diff),
                "mean_a": a.mean() if len(a) else np.nan,
                "mean_b": b.mean() if len(b) else np.nan,
                "mean_diff_b_minus_a": np.nan,
                "paired_t_pvalue": np.nan,
                "wilcoxon_pvalue": np.nan,
                "cohens_dz": np.nan
            })
            continue

        t_stat, t_p = stats.ttest_rel(b, a)

        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        cohens_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan

        results.append({
            "dimension": col,
            "n_pairs": len(diff),
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "mean_diff_b_minus_a": diff.mean(),
            "paired_t_stat": t_stat,
            "paired_t_pvalue": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_p,
            "cohens_dz": cohens_dz
        })

    return pd.DataFrame(results)


# 10. NLP-based qualitative analysis (if applicable)
def correlation_with_human_scores(
    df,
    metric_cols,
    human_cols,
    method="spearman"
):
    """
    Compute correlations between automatic metrics and human rating dimensions.

    method: "spearman" or "pearson"
    """
    results = []

    for metric in metric_cols:
        for human in human_cols:
            temp = df[[metric, human]].dropna()

            if len(temp) < 3:
                corr, p = np.nan, np.nan
            else:
                if method == "spearman":
                    corr, p = stats.spearmanr(temp[metric], temp[human])
                elif method == "pearson":
                    corr, p = stats.pearsonr(temp[metric], temp[human])
                else:
                    raise ValueError("method must be 'spearman' or 'pearson'")

            results.append({
                "metric": metric,
                "human_dimension": human,
                "n": len(temp),
                "correlation": corr,
                "pvalue": p,
                "method": method
            })

    return pd.DataFrame(results)

# metric_cols = ["bleu", "rouge_l", "meteor", "bertscore_f1"]
# human_cols = rating_cols + ["human_avg_score"]

# corr = correlation_with_human_scores(df, metric_cols, human_cols)
# display(corr)


# 11. Human-Metric Gap Analysis
def min_max_normalize(series):
    """
    Min-max normalize a pandas Series.
    """
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.nan, index=series.index)

    return (series - min_val) / (max_val - min_val)


def add_human_metric_gap(
    df,
    metric_col,
    human_col="human_avg_score",
    new_col="human_metric_gap"
):
    """
    Add gap score:
    normalized automatic metric - normalized human rating.

    High positive value means:
    automatic metric is high, but human rating is relatively low.
    """
    df = df.copy()

    metric_norm = min_max_normalize(df[metric_col])
    human_norm = min_max_normalize(df[human_col])

    df[new_col] = metric_norm - human_norm

    return df

# df = add_human_metric_gap(
#     df,
#     metric_col="bertscore_f1",
#     human_col="human_avg_score",
#     new_col="bertscore_human_gap"
# )

# gap_cases = df.sort_values("bertscore_human_gap", ascending=False).head(10)
# display(gap_cases)


# 12. Error Taxonomy(if applicable)
def get_low_scoring_cases(
    df,
    score_col="human_avg_score",
    n=10,
    text_cols=None
):
    """
    Return lowest-scoring cases for qualitative analysis.
    """
    if text_cols is None:
        text_cols = []

    cols = [
        col for col in [
            "participant_id", "session_id", "condition", score_col
        ] if col in df.columns
    ]

    cols += [col for col in text_cols if col in df.columns]

    return df.sort_values(score_col, ascending=True)[cols].head(n)


# low_cases = get_low_scoring_cases(
#     df,
#     score_col="human_avg_score",
#     n=10,
#     text_cols=["client_message", "llm_response"]
# )

# display(low_cases)


# 13. The whole pipeline
def run_preliminary_analysis(
    data_dir,
    rating_cols,
    column_map=None,
    condition_col="condition",
    participant_col="participant_id",
    session_col="session_id",
    condition_a="Base",
    condition_b="Fine-tuned",
    paired=False
):
    """
    Full preliminary analysis pipeline.
    """
    df = load_all_csvs(data_dir)
    df = standardize_columns(df, column_map=column_map)

    check_data_overview(
        df,
        participant_col=participant_col,
        condition_col=condition_col,
        session_col=session_col
    )

    df = add_average_human_score(df, rating_cols)

    completion = participant_completion_table(
        df,
        participant_col=participant_col,
        session_col=session_col,
        condition_col=condition_col
    )

    descriptive = descriptive_by_condition(
        df,
        rating_cols=rating_cols + ["human_avg_score"],
        condition_col=condition_col
    )

    if paired:
        comparison = compare_conditions_paired(
            df,
            rating_cols=rating_cols + ["human_avg_score"],
            participant_col=participant_col,
            session_col=session_col,
            condition_col=condition_col,
            condition_a=condition_a,
            condition_b=condition_b
        )
    else:
        comparison = compare_conditions_independent(
            df,
            rating_cols=rating_cols + ["human_avg_score"],
            condition_col=condition_col,
            condition_a=condition_a,
            condition_b=condition_b
        )

    return {
        "data": df,
        "completion": completion,
        "descriptive": descriptive,
        "comparison": comparison
    }


# rating_cols = [
#     "empathy",
#     "naturalness",
#     "cultural_appropriateness",
#     "relevance",
#     "helpfulness",
#     "overall_quality"
# ]

# results = run_preliminary_analysis(
#     data_dir="./responses",
#     rating_cols=rating_cols,
#     condition_col="condition",
#     participant_col="participant_id",
#     session_col="session_id",
#     condition_a="Base",
#     condition_b="Fine-tuned",
#     paired=False
# )

# df_clean = results["data"]
# completion = results["completion"]
# descriptive = results["descriptive"]
# comparison = results["comparison"]

# display(completion)
# display(descriptive)
# display(comparison)


def save_analysis_outputs(results, output_dir="./analysis_outputs"):
    """
    Save analysis outputs as CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in results.items():
        if isinstance(obj, pd.DataFrame):
            path = output_dir / f"{name}.csv"
            obj.to_csv(path, index=False)
            print(f"[SAVED] {path}")

# save_analysis_outputs(results, output_dir="./analysis_outputs")