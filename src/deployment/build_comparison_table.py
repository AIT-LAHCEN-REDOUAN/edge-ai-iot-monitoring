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


# ===============================
# HELPERS
# ===============================
def extract_value(text, pattern):
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def infer_device(technique_name):
    if technique_name in {"baseline_gpu", "q5_mixed_precision"}:
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


def parse_report(text, technique_name):
    row = {
        "technique": technique_name,
        "family": infer_family(technique_name),
        "device_context": infer_device(technique_name),
        "effective_size_mb": None,
        "accuracy": None,
        "avg_batch_time_sec": None,
        "avg_image_time_sec": None,
        "throughput_img_sec": None,
        "peak_gpu_memory_mb": None,
        "sparsity": None,
        "compression_ratio": None,
        "finetune_time_sec": None,
    }

    if technique_name == "baseline_cpu":
        row["effective_size_mb"] = extract_value(
            text, r"Model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )

    elif technique_name == "baseline_gpu":
        row["effective_size_mb"] = extract_value(
            text, r"Model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["peak_gpu_memory_mb"] = extract_value(
            text, r"Peak GPU memory \(MB\):\s*([0-9.]+)"
        )

    elif technique_name == "q1_dynamic_quantization":
        row["effective_size_mb"] = extract_value(
            text, r"Quantized model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Q1 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )

    elif technique_name == "q2_static_ptq_fx":
        row["effective_size_mb"] = extract_value(
            text, r"Q2 FX model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Q2 FX test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )

    elif technique_name == "q3_qat":
        row["effective_size_mb"] = extract_value(
            text, r"Q3 model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Q3 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )
        row["finetune_time_sec"] = extract_value(
            text, r"QAT total training time \(sec\):\s*([0-9.]+)"
        )

    elif technique_name == "q4_weight_only":
        row["effective_size_mb"] = extract_value(
            text, r"Q4 model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"Q4 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )

    elif technique_name == "q5_mixed_precision":
        # IMPORTANT: use ONLY AMP metrics, not FP32 metrics
        row["effective_size_mb"] = extract_value(
            text, r"Q5 FP16 checkpoint size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"AMP test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"AMP avg batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"AMP avg image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"AMP throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["peak_gpu_memory_mb"] = extract_value(
            text, r"AMP peak GPU memory \(MB\):\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Checkpoint compression ratio:\s*([0-9.]+)"
        )

    elif technique_name == "p1_unstructured_pruning":
        row["effective_size_mb"] = extract_value(
            text, r"P1 model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"P1 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["sparsity"] = extract_value(
            text, r"Measured sparsity:\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )
        row["finetune_time_sec"] = extract_value(
            text, r"Fine-tuning time \(sec\):\s*([0-9.]+)"
        )

    elif technique_name == "p2_structured_pruning":
        row["effective_size_mb"] = extract_value(
            text, r"P2 model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"P2 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["sparsity"] = extract_value(
            text, r"Measured sparsity:\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )
        row["finetune_time_sec"] = extract_value(
            text, r"Fine-tuning time \(sec\):\s*([0-9.]+)"
        )

    elif technique_name == "p3_magnitude_pruning":
        row["effective_size_mb"] = extract_value(
            text, r"P3 model size \(MB\):\s*([0-9.]+)"
        )
        row["accuracy"] = extract_value(
            text, r"P3 test accuracy:\s*([0-9.]+)"
        )
        row["avg_batch_time_sec"] = extract_value(
            text, r"Average batch inference time \(sec\):\s*([0-9.]+)"
        )
        row["avg_image_time_sec"] = extract_value(
            text, r"Average image inference time \(sec\):\s*([0-9.]+)"
        )
        row["throughput_img_sec"] = extract_value(
            text, r"Throughput \(images/sec\):\s*([0-9.]+)"
        )
        row["sparsity"] = extract_value(
            text, r"Measured sparsity:\s*([0-9.]+)"
        )
        row["compression_ratio"] = extract_value(
            text, r"Compression ratio:\s*([0-9.]+)"
        )
        row["finetune_time_sec"] = extract_value(
            text, r"Fine-tuning time \(sec\):\s*([0-9.]+)"
        )

    return row


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

        row = parse_report(text, technique_name)
        row["source_file"] = str(report_path)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No reports were found or parsed successfully.")

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

    # Build readable summary
    best_cpu_speed = df[df["device_context"] == "cpu"].sort_values(
        by="throughput_img_sec", ascending=False
    ).head(1)

    best_cpu_accuracy = df[df["device_context"] == "cpu"].sort_values(
        by="accuracy", ascending=False
    ).head(1)

    best_gpu_speed = df[df["device_context"] == "gpu"].sort_values(
        by="throughput_img_sec", ascending=False
    ).head(1)

    smallest_model = df.sort_values(
        by="effective_size_mb", ascending=True
    ).head(1)

    summary_lines = []
    summary_lines.append("=== OPTIMIZATION COMPARISON SUMMARY ===\n")
    summary_lines.append(f"Total techniques parsed: {len(df)}\n")

    if not best_cpu_speed.empty:
        row = best_cpu_speed.iloc[0]
        summary_lines.append(
            f"Best CPU throughput: {row['technique']} ({row['throughput_img_sec']:.2f} img/s)\n"
        )

    if not best_cpu_accuracy.empty:
        row = best_cpu_accuracy.iloc[0]
        summary_lines.append(
            f"Best CPU accuracy: {row['technique']} ({row['accuracy']:.6f})\n"
        )

    if not best_gpu_speed.empty:
        row = best_gpu_speed.iloc[0]
        summary_lines.append(
            f"Best GPU throughput: {row['technique']} ({row['throughput_img_sec']:.2f} img/s)\n"
        )

    if not smallest_model.empty:
        row = smallest_model.iloc[0]
        summary_lines.append(
            f"Smallest model/checkpoint: {row['technique']} ({row['effective_size_mb']:.4f} MB)\n"
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