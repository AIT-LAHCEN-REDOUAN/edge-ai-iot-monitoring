from pathlib import Path
import re
import pandas as pd

# ===============================
# CONFIG
# ===============================
RESULTS_DIR = Path(r"D:\github\IOT_Project\results\optimization")
OUTPUT_DIR = Path(r"D:\github\IOT_Project\results\deployment")
OUTPUT_CSV = OUTPUT_DIR / "optimization_comparison_table.csv"
OUTPUT_TXT = OUTPUT_DIR / "optimization_comparison_summary.txt"

REPORT_FILES = {
    "baseline_cpu": RESULTS_DIR / "baseline_benchmark_cpu.txt",
    "baseline_gpu": RESULTS_DIR / "baseline_benchmark.txt",
    "q1_dynamic_quantization": RESULTS_DIR / "q1_dynamic_quantization_report.txt",
    "q2_static_ptq_fx": RESULTS_DIR / "q2_static_ptq_fx_report.txt",
    "q3_qat": RESULTS_DIR / "q3_qat_report.txt",
    "q4_weight_only": RESULTS_DIR / "q4_weight_only_report.txt",
    "q5_mixed_precision": RESULTS_DIR / "q5_mixed_precision_report.txt",
    "p1_unstructured_pruning": RESULTS_DIR / "p1_unstructured_pruning_report.txt",
    "p2_structured_pruning": RESULTS_DIR / "p2_structured_pruning_report.txt",
    "p3_magnitude_pruning": RESULTS_DIR / "p3_magnitude_pruning_report.txt",
}

METRIC_PATTERNS = {
    "model_size_mb": [
        r"Model size \(MB\):\s*([0-9.]+)",
        r"Baseline checkpoint size \(MB\):\s*([0-9.]+)",
    ],
    "optimized_size_mb": [
        r"Q1 model size \(MB\):\s*([0-9.]+)",
        r"Q2 FX model size \(MB\):\s*([0-9.]+)",
        r"Q3 model size \(MB\):\s*([0-9.]+)",
        r"Q4 model size \(MB\):\s*([0-9.]+)",
        r"Q5 FP16 checkpoint size \(MB\):\s*([0-9.]+)",
        r"P1 model size \(MB\):\s*([0-9.]+)",
        r"P2 model size \(MB\):\s*([0-9.]+)",
        r"P3 model size \(MB\):\s*([0-9.]+)",
    ],
    "accuracy": [
        r"Test accuracy:\s*([0-9.]+)",
        r"Q1 test accuracy:\s*([0-9.]+)",
        r"Q2 FX test accuracy:\s*([0-9.]+)",
        r"Q3 test accuracy:\s*([0-9.]+)",
        r"Q4 test accuracy:\s*([0-9.]+)",
        r"FP32 test accuracy:\s*([0-9.]+)",
        r"AMP test accuracy:\s*([0-9.]+)",
        r"P1 test accuracy:\s*([0-9.]+)",
        r"P2 test accuracy:\s*([0-9.]+)",
        r"P3 test accuracy:\s*([0-9.]+)",
    ],
    "avg_batch_time_sec": [
        r"Average batch inference time \(sec\):\s*([0-9.]+)",
        r"FP32 avg batch inference time \(sec\):\s*([0-9.]+)",
        r"AMP avg batch inference time \(sec\):\s*([0-9.]+)",
    ],
    "avg_image_time_sec": [
        r"Average image inference time \(sec\):\s*([0-9.]+)",
        r"FP32 avg image inference time \(sec\):\s*([0-9.]+)",
        r"AMP avg image inference time \(sec\):\s*([0-9.]+)",
    ],
    "throughput_img_sec": [
        r"Throughput \(images/sec\):\s*([0-9.]+)",
        r"FP32 throughput \(images/sec\):\s*([0-9.]+)",
        r"AMP throughput \(images/sec\):\s*([0-9.]+)",
    ],
    "peak_gpu_memory_mb": [
        r"Peak GPU memory \(MB\):\s*([0-9.]+)",
        r"FP32 peak GPU memory \(MB\):\s*([0-9.]+)",
        r"AMP peak GPU memory \(MB\):\s*([0-9.]+)",
    ],
    "sparsity": [
        r"Measured sparsity:\s*([0-9.]+)",
    ],
    "finetune_time_sec": [
        r"Fine-tuning time \(sec\):\s*([0-9.]+)",
        r"QAT total training time \(sec\):\s*([0-9.]+)",
    ],
    "compression_ratio": [
        r"Compression ratio:\s*([0-9.]+)",
        r"Checkpoint compression ratio:\s*([0-9.]+)",
    ],
}


# ===============================
# HELPERS
# ===============================
def extract_first_match(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def infer_device(technique_name):
    if technique_name == "baseline_gpu":
        return "gpu"
    if technique_name == "q5_mixed_precision":
        return "gpu"
    return "cpu"


def infer_family(technique_name):
    if technique_name.startswith("q"):
        return "quantization"
    if technique_name.startswith("p"):
        return "pruning"
    return "baseline"


def read_report(report_path: Path):
    if not report_path.exists():
        return None
    return report_path.read_text(encoding="utf-8")


# ===============================
# MAIN
# ===============================
def main():
    print("📋 BUILDING GLOBAL OPTIMIZATION COMPARISON TABLE...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for technique_name, report_path in REPORT_FILES.items():
        print(f"Reading: {report_path.name}")

        text = read_report(report_path)
        if text is None:
            print(f"⚠️ Missing file: {report_path}")
            continue

        row = {
            "technique": technique_name,
            "family": infer_family(technique_name),
            "device_context": infer_device(technique_name),
            "source_file": str(report_path),
        }

        for metric_name, patterns in METRIC_PATTERNS.items():
            row[metric_name] = extract_first_match(text, patterns)

        # Normalize which size column is used
        if technique_name.startswith("baseline"):
            row["effective_size_mb"] = row["model_size_mb"]
        else:
            row["effective_size_mb"] = row["optimized_size_mb"]

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No reports were found or parsed successfully.")

    # Order columns
    preferred_columns = [
        "technique",
        "family",
        "device_context",
        "effective_size_mb",
        "accuracy",
        "avg_batch_time_sec",
        "avg_image_time_sec",
        "throughput_img_sec",
        "peak_gpu_memory_mb",
        "sparsity",
        "compression_ratio",
        "finetune_time_sec",
        "source_file",
    ]

    for col in preferred_columns:
        if col not in df.columns:
            df[col] = None

    df = df[preferred_columns]

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)

    # Build human-readable summary
    best_cpu_speed = df[df["device_context"] == "cpu"].sort_values(
        by="throughput_img_sec", ascending=False
    ).head(1)

    best_cpu_accuracy = df[df["device_context"] == "cpu"].sort_values(
        by="accuracy", ascending=False
    ).head(1)

    best_gpu_speed = df[df["device_context"] == "gpu"].sort_values(
        by="throughput_img_sec", ascending=False
    ).head(1)

    smallest_model = df.sort_values(by="effective_size_mb", ascending=True).head(1)

    summary_lines = []
    summary_lines.append("=== OPTIMIZATION COMPARISON SUMMARY ===\n")
    summary_lines.append(f"Total techniques parsed: {len(df)}\n")

    if not best_cpu_speed.empty:
        row = best_cpu_speed.iloc[0]
        summary_lines.append(
            f"Best CPU throughput: {row['technique']} "
            f"({row['throughput_img_sec']:.2f} img/s)\n"
        )

    if not best_cpu_accuracy.empty:
        row = best_cpu_accuracy.iloc[0]
        summary_lines.append(
            f"Best CPU accuracy: {row['technique']} "
            f"({row['accuracy']:.6f})\n"
        )

    if not best_gpu_speed.empty:
        row = best_gpu_speed.iloc[0]
        summary_lines.append(
            f"Best GPU throughput: {row['technique']} "
            f"({row['throughput_img_sec']:.2f} img/s)\n"
        )

    if not smallest_model.empty:
        row = smallest_model.iloc[0]
        summary_lines.append(
            f"Smallest model/checkpoint: {row['technique']} "
            f"({row['effective_size_mb']:.4f} MB)\n"
        )

    summary_lines.append("\n=== FULL TABLE PREVIEW ===\n")
    summary_lines.append(df.to_string(index=False))

    OUTPUT_TXT.write_text("".join(summary_lines), encoding="utf-8")

    print("\n==============================")
    print("✅ COMPARISON TABLE COMPLETED")
    print("==============================")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Summary saved to: {OUTPUT_TXT}")
    print("\nPreview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()