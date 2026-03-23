from pathlib import Path
import pandas as pd
import json

# ===============================
# CONFIG
# ===============================
INPUT_CSV = Path(r"D:\github\IOT_Project\results\deployment\optimization_scored_table.csv")

OUTPUT_DIR = Path(r"D:\github\IOT_Project\results\deployment")
OUTPUT_JSON = OUTPUT_DIR / "vm_profiles.json"
OUTPUT_TXT = OUTPUT_DIR / "vm_profiles_summary.txt"


# ===============================
# HELPERS
# ===============================
def select_row(df: pd.DataFrame, technique_name: str):
    match = df[df["technique"] == technique_name]
    if match.empty:
        raise ValueError(f"Technique not found in scored table: {technique_name}")
    return match.iloc[0]


# ===============================
# MAIN
# ===============================
def main():
    print("🖥️ PREPARING VM DEPLOYMENT PROFILES...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Scored table not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Official automatic selection from your previous step
    vm1_technique = "q3_qat"
    vm2_technique = "q3_qat"
    vm3_technique = "q3_qat"

    vm1 = select_row(df, vm1_technique)
    vm2 = select_row(df, vm2_technique)
    vm3 = select_row(df, vm3_technique)

    vm_profiles = {
        "vm1": {
            "profile_name": "highly_constrained",
            "cpu_limit": 1,
            "memory_limit_mb": 512,
            "selected_technique": vm1["technique"],
            "device_context": vm1["device_context"],
            "effective_size_mb": float(vm1["effective_size_mb"]),
            "accuracy": float(vm1["accuracy"]),
            "avg_image_time_sec": float(vm1["avg_image_time_sec"]),
            "throughput_img_sec": float(vm1["throughput_img_sec"]),
            "selection_score": float(vm1["vm1_score"]),
        },
        "vm2": {
            "profile_name": "balanced",
            "cpu_limit": 2,
            "memory_limit_mb": 1024,
            "selected_technique": vm2["technique"],
            "device_context": vm2["device_context"],
            "effective_size_mb": float(vm2["effective_size_mb"]),
            "accuracy": float(vm2["accuracy"]),
            "avg_image_time_sec": float(vm2["avg_image_time_sec"]),
            "throughput_img_sec": float(vm2["throughput_img_sec"]),
            "selection_score": float(vm2["vm2_score"]),
        },
        "vm3": {
            "profile_name": "accuracy_first",
            "cpu_limit": 2,
            "memory_limit_mb": 2048,
            "selected_technique": vm3["technique"],
            "device_context": vm3["device_context"],
            "effective_size_mb": float(vm3["effective_size_mb"]),
            "accuracy": float(vm3["accuracy"]),
            "avg_image_time_sec": float(vm3["avg_image_time_sec"]),
            "throughput_img_sec": float(vm3["throughput_img_sec"]),
            "selection_score": float(vm3["vm3_score"]),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(vm_profiles, f, indent=4)

    summary_lines = []
    summary_lines.append("=== VM DEPLOYMENT PROFILES ===\n\n")

    for vm_name, info in vm_profiles.items():
        summary_lines.append(f"{vm_name.upper()}\n")
        summary_lines.append(f"Profile: {info['profile_name']}\n")
        summary_lines.append(f"CPU limit: {info['cpu_limit']}\n")
        summary_lines.append(f"Memory limit (MB): {info['memory_limit_mb']}\n")
        summary_lines.append(f"Selected technique: {info['selected_technique']}\n")
        summary_lines.append(f"Device context: {info['device_context']}\n")
        summary_lines.append(f"Model size (MB): {info['effective_size_mb']:.4f}\n")
        summary_lines.append(f"Accuracy: {info['accuracy']:.6f}\n")
        summary_lines.append(f"Avg image time (sec): {info['avg_image_time_sec']:.6f}\n")
        summary_lines.append(f"Throughput (img/sec): {info['throughput_img_sec']:.2f}\n")
        summary_lines.append(f"Selection score: {info['selection_score']:.6f}\n\n")

    OUTPUT_TXT.write_text("".join(summary_lines), encoding="utf-8")

    print("\n==============================")
    print("✅ VM PROFILES PREPARED")
    print("==============================")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Summary saved to: {OUTPUT_TXT}")

    print("\nSelected deployment plan:")
    for vm_name, info in vm_profiles.items():
        print(f"{vm_name} -> {info['selected_technique']} | "
              f"CPU={info['cpu_limit']} | RAM={info['memory_limit_mb']}MB")


if __name__ == "__main__":
    main()