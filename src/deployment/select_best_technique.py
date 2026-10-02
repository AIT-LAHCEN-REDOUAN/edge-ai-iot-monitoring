from pathlib import Path
import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================
INPUT_CSV = Path(r"D:\github\IOT_Project\results\deployment\optimization_comparison_table.csv")

OUTPUT_DIR = Path(r"D:\github\IOT_Project\results\deployment")
SCORED_CSV = OUTPUT_DIR / "optimization_scored_table.csv"
SELECTION_TXT = OUTPUT_DIR / "best_techniques_per_vm.txt"


# ===============================
# HELPERS
# ===============================
def min_max_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Normalize to [0,1].
    If higher_is_better=False, smaller values become better scores.
    """
    s = series.astype(float)

    if s.isna().all():
        return pd.Series([np.nan] * len(s), index=s.index)

    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(s), index=s.index)

    if higher_is_better:
        return (s - min_val) / (max_val - min_val)
    else:
        return (max_val - s) / (max_val - min_val)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only techniques with all core metrics available
    required_cols = ["effective_size_mb", "accuracy", "avg_image_time_sec"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Normalized metrics
    df["size_score"] = min_max_normalize(df["effective_size_mb"], higher_is_better=False)
    df["speed_score"] = min_max_normalize(df["avg_image_time_sec"], higher_is_better=False)
    df["accuracy_score"] = min_max_normalize(df["accuracy"], higher_is_better=True)

    # VM scoring rules
    # VM1: constrained -> prioritize size and speed
    df["vm1_score"] = (
        0.45 * df["size_score"] +
        0.35 * df["speed_score"] +
        0.20 * df["accuracy_score"]
    )

    # VM2: balanced
    df["vm2_score"] = (
        0.30 * df["size_score"] +
        0.30 * df["speed_score"] +
        0.40 * df["accuracy_score"]
    )

    # VM3: accuracy-first
    df["vm3_score"] = (
        0.20 * df["size_score"] +
        0.20 * df["speed_score"] +
        0.60 * df["accuracy_score"]
    )

    return df


def select_best(df: pd.DataFrame, score_column: str) -> pd.Series:
    return df.sort_values(by=score_column, ascending=False).iloc[0]


# ===============================
# MAIN
# ===============================
def main():
    print("🏆 SELECTING BEST TECHNIQUE PER VM PROFILE...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    scored_df = compute_scores(df)
    scored_df.to_csv(SCORED_CSV, index=False)

    best_vm1 = select_best(scored_df, "vm1_score")
    best_vm2 = select_best(scored_df, "vm2_score")
    best_vm3 = select_best(scored_df, "vm3_score")

    summary = []
    summary.append("=== BEST TECHNIQUES PER VM PROFILE ===\n\n")

    summary.append("VM1 (highly constrained / size-speed priority)\n")
    summary.append(f"Selected technique: {best_vm1['technique']}\n")
    summary.append(f"Score: {best_vm1['vm1_score']:.6f}\n")
    summary.append(f"Size (MB): {best_vm1['effective_size_mb']:.4f}\n")
    summary.append(f"Accuracy: {best_vm1['accuracy']:.6f}\n")
    summary.append(f"Avg image time (sec): {best_vm1['avg_image_time_sec']:.6f}\n\n")

    summary.append("VM2 (balanced profile)\n")
    summary.append(f"Selected technique: {best_vm2['technique']}\n")
    summary.append(f"Score: {best_vm2['vm2_score']:.6f}\n")
    summary.append(f"Size (MB): {best_vm2['effective_size_mb']:.4f}\n")
    summary.append(f"Accuracy: {best_vm2['accuracy']:.6f}\n")
    summary.append(f"Avg image time (sec): {best_vm2['avg_image_time_sec']:.6f}\n\n")

    summary.append("VM3 (accuracy-first profile)\n")
    summary.append(f"Selected technique: {best_vm3['technique']}\n")
    summary.append(f"Score: {best_vm3['vm3_score']:.6f}\n")
    summary.append(f"Size (MB): {best_vm3['effective_size_mb']:.4f}\n")
    summary.append(f"Accuracy: {best_vm3['accuracy']:.6f}\n")
    summary.append(f"Avg image time (sec): {best_vm3['avg_image_time_sec']:.6f}\n\n")

    summary.append("=== FULL SCORED TABLE ===\n")
    display_cols = [
        "technique",
        "device_context",
        "effective_size_mb",
        "accuracy",
        "avg_image_time_sec",
        "size_score",
        "speed_score",
        "accuracy_score",
        "vm1_score",
        "vm2_score",
        "vm3_score",
    ]
    summary.append(scored_df[display_cols].to_string(index=False))

    SELECTION_TXT.write_text("".join(summary), encoding="utf-8")

    print("\n==============================")
    print("✅ TECHNIQUE SELECTION COMPLETED")
    print("==============================")
    print(f"Scored table saved to: {SCORED_CSV}")
    print(f"Selection summary saved to: {SELECTION_TXT}")

    print("\nSelected techniques:")
    print(f"VM1 -> {best_vm1['technique']}")
    print(f"VM2 -> {best_vm2['technique']}")
    print(f"VM3 -> {best_vm3['technique']}")


if __name__ == "__main__":
    main()