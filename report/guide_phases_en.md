# End‑to‑End Runbook — Phases 1 → 6

Operational guide to run the entire project on Windows/PowerShell, from data preprocessing to IoT supervision with ThingsBoard.

## 0. Prerequisites
- Python 3.11 installed.
- Docker Desktop running.
- Install host Python deps (used by orchestrator and utility scripts):

```powershell
python -m pip install -r requirements.txt
```

- Dataset available or prepared with scripts in `src/dataset`.

## 1. Dataset Preparation
- Inventory and inspection:

```powershell
python src\dataset\build_metadata.py
python src\dataset\inspect_dataset.py
```

- Train/val/test split (70/15/15):

```powershell
python src\dataset\split_dataset.py
```

- Volume preprocessing to `.npy` (128×128×128):

```powershell
python src\dataset\preprocess_nii.py
```

- 2D slice extraction:

```powershell
python src\dataset\extract_slices.py
```

Outputs:
- Metadata in `data/metadata`
- Processed data in `data/processed`

## 2. Phase 1 — Baseline Model
- Train:

```powershell
python src\baseline\train_baseline.py
```

- Evaluate (metrics + confusion matrix):

```powershell
python src\baseline\evaluate_baseline.py
```

Key artifacts:
- `models/baseline/best_model.pth`
- `results/baseline/metrics.txt`, `results/baseline/confusion_matrix.png`

## 3. Phase 2 — Optimizations (Q1–Q5, P1–P3)
- Baseline benchmarks:

```powershell
python src\optimization\benchmark_baseline.py
python src\optimization\benchmark_baseline_cpu.py
```

- Quantization:

```powershell
python src\optimization\q1_dynamic_quantization\apply_q1.py
python src\optimization\q2_static_ptq\apply_q2.py
python src\optimization\q3_qat\apply_q3.py
python src\optimization\q4_weight_only\apply_q4.py
python src\optimization\q5_mixed_precision\apply_q5.py
```

- Pruning:

```powershell
python src\optimization\p1_unstructured_pruning\apply_p1.py
python src\optimization\p2_structured_pruning\apply_p2.py
python src\optimization\p3_magnitude_pruning\apply_p3.py
```

Outputs:
- Optimized checkpoints in `models/optimized`
- Reports in `results/optimization`

## 4. Phase 3 — VM Deployment (Docker)
- Start ThingsBoard and the three constrained VMs:

```powershell
docker compose up -d thingsboard
docker compose up -d vm1 vm2 vm3
docker compose ps
```

- Health checks:

```powershell
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

Notes:
- For env changes or code under mounted volumes (`scripts/`, `src/`, `data/`, `models/`, `results/`), do not rebuild. Use `docker compose up -d vmX` or `docker compose restart vmX`.

## 5. Phase 4 — Best Technique per VM
1) Build the comparison table from per‑technique reports:

```powershell
python src\deployment\build_comparison_table.py
```

2) Score and select per VM profile, export VM profiles:

```powershell
python src\deployment\select_best_technique.py
python src\deployment\prepare_vm_profiles.py
```

Outputs:
- `results/deployment/optimization_comparison_table.csv`
- `results/deployment/optimization_scored_table.csv`
- `results/deployment/best_techniques_per_vm.txt`
- `results/deployment/vm_profiles.json`

## 6. Phase 5 — Collective Inference
### 6.1 VM HTTP inference servers
- Each VM exposes `POST /infer` on host ports 5001/5002/5003. Example request:

```powershell
$p="/app/data/processed/slices/test/0/<your_file>.npy"
$body=@{ slice_path=$p; patient_id="P-TEST-001"} | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:5001/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
```

### 6.2 Orchestrator (weighted vote, confidence revalidation, load‑aware)

```powershell
python src\collective\run_collective.py
```

Outputs:
- `results/collective/detail.csv`
- `results/collective/summary.txt` with collective_accuracy, best_individual_accuracy, consensus_rate, revalidation_improvements

## 7. Phase 6 — ThingsBoard Supervision (MQTT)
### 7.1 UI and device creation
- Open `http://localhost:9090`
- Login as tenant
- Create devices `vm1`, `vm2`, `vm3` and copy their Access tokens

### 7.2 Enable MQTT telemetry from VMs
- For each VM in `docker-compose.yml` set:
  - `TB_ENABLE: "true"`
  - `TB_HOST: "thingsboard"`
  - `TB_PORT: 1883`
  - `TB_TOKEN: "<YOUR_DEVICE_TOKEN>"`
- Apply without rebuild:

```powershell
docker compose up -d vm1 vm2 vm3
```

- Trigger one inference per VM to push telemetry:

```powershell
$p="/app/data/processed/slices/test/0/<your_file>.npy"
$body=@{ slice_path=$p; patient_id="P-DEMO-001"} | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:5001/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
Invoke-WebRequest -Method POST -Uri http://localhost:5002/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
Invoke-WebRequest -Method POST -Uri http://localhost:5003/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
```

- In ThingsBoard → Devices → `vm1`/`vm2`/`vm3` → Latest telemetry you should see:
  - `vm_id`, `timestamp`, `technique`, `prediction`, `confidence`, `inference_time_ms`, `cpu_usage_pct`, `ram_usage_mb`, `patient_id`

### 7.3 Collective telemetry (optional)
- Create a `collective` device and copy its token, then:

```powershell
$env:TB_COLLECTIVE_ENABLE="true"
$env:TB_COLLECTIVE_HOST="localhost"
$env:TB_COLLECTIVE_PORT="1883"
$env:TB_COLLECTIVE_TOKEN="<COLLECTIVE_TOKEN>"
python src\collective\run_collective.py
```

## 8. Quick Checks
- VM endpoints:

```powershell
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

- ThingsBoard HTTP test (replace `<TOKEN>`):

```powershell
$b=@{ temperature=25 } | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:9090/api/v1/<TOKEN>/telemetry -Body $b -ContentType 'application/json'
```

## 9. Troubleshooting
- TB UI not on 8080: use `http://localhost:9090`.
- No device telemetry: ensure `TB_ENABLE="true"` and the correct `TB_TOKEN` in `docker-compose.yml`, then `docker compose up -d vmX` and POST `/infer`.
- 400 from `/infer`: make sure you send a container path under `/app/...`, not a Windows host path.
- Slow builds: avoid `--no-cache`; rebuild only if the base image or `requirements.txt` changed. Mounted volumes let you iterate without rebuilding.

---

Project structure:
- Dataset scripts: `src/dataset`
- Baseline: `src/baseline`
- Optimizations: `src/optimization`
- Deployment & selection: `src/deployment`
- Orchestrator & collective: `src/collective`
- ThingsBoard (MQTT client): `src/thingsboard`

