# Guide de mise en œuvre — Phases 1 → 6

Ce document décrit, de façon opérationnelle, toutes les étapes pour exécuter le projet de bout en bout sur Windows/PowerShell, du prétraitement des données jusqu’à la supervision IoT via ThingsBoard.

## 0. Prérequis
- Python 3.11 installé (utilisation locale pour l’orchestrateur et les scripts).
- Docker Desktop démarré.
- Dépendances Python (pour l’hôte) :

```powershell
python -m pip install -r requirements.txt
```

- Données présentes ou à préparer via les scripts du dossier `src/dataset`.

## 1. Préparation du jeu de données
- Inventaire et inspection :

```powershell
python src\dataset\build_metadata.py
python src\dataset\inspect_dataset.py
```

- Découpage train/val/test (70/15/15) :

```powershell
python src\dataset\split_dataset.py
```

- Prétraitement des volumes NIfTI vers `.npy` (128×128×128) :

```powershell
python src\dataset\preprocess_nii.py
```

- Extraction des coupes 2D (slices) :

```powershell
python src\dataset\extract_slices.py
```

Les métadonnées produites sont dans `data/metadata` et les sorties traitées dans `data/processed`.

## 2. Phase 1 — Modèle de base (Baseline)
- Entraînement :

```powershell
python src\baseline\train_baseline.py
```

- Évaluation (métriques + matrice de confusion) :

```powershell
python src\baseline\evaluate_baseline.py
```

Sorties clés :
- Modèle : `models/baseline/best_model.pth`
- Résultats : `results/baseline/metrics.txt`, `results/baseline/confusion_matrix.png`

## 3. Phase 2 — Optimisations (Q1–Q5, P1–P3)
- Référence de performance du baseline :

```powershell
python src\optimization\benchmark_baseline.py
python src\optimization\benchmark_baseline_cpu.py
```

- Quantification :

```powershell
python src\optimization\q1_dynamic_quantization\apply_q1.py
python src\optimization\q2_static_ptq\apply_q2.py
python src\optimization\q3_qat\apply_q3.py
python src\optimization\q4_weight_only\apply_q4.py
python src\optimization\q5_mixed_precision\apply_q5.py
```

- Élagage :

```powershell
python src\optimization\p1_unstructured_pruning\apply_p1.py
python src\optimization\p2_structured_pruning\apply_p2.py
python src\optimization\p3_magnitude_pruning\apply_p3.py
```

Sorties :
- Checkpoints optimisés : `models/optimized/…`
- Rapports : `results/optimization/…`

## 4. Phase 3 — Déploiement VM (Docker)
- Démarrage de ThingsBoard (port UI 9090, MQTT 1883) et des VMs contraintes :

```powershell
docker compose up -d thingsboard
docker compose up -d vm1 vm2 vm3
docker compose ps
```

- Vérification des VM :

```powershell
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

Remarques :
- Inutile de reconstruire pour des changements d’ENV ou de code monté en volume (`scripts/`, `src/`, `data/`, `models/`, `results/`). Utiliser `docker compose up -d vmX` ou `docker compose restart vmX`.

## 5. Phase 4 — Sélection de la meilleure technique par VM
1) Agrégation des rapports en tableau comparatif :

```powershell
python src\deployment\build_comparison_table.py
```

2) Scoring et sélection par profil VM :

```powershell
python src\deployment\select_best_technique.py
python src\deployment\prepare_vm_profiles.py
```

Sorties :
- `results/deployment/optimization_comparison_table.csv`
- `results/deployment/optimization_scored_table.csv`
- `results/deployment/best_techniques_per_vm.txt`
- `results/deployment/vm_profiles.json`

## 6. Phase 5 — Intelligence collective
### 6.1 Serveurs d’inférence (VM)
- Les 3 services exposent `POST /infer` (port hôte 5001/5002/5003). Exemple d’appel :

```powershell
$p="/app/data/processed/slices/test/0/<votre_fichier>.npy"
$body=@{ slice_path=$p; patient_id="P-TEST-001"} | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:5001/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
```

### 6.2 Orchestrateur (vote pondéré, revalidation par confiance, équilibrage)

```powershell
python src\collective\run_collective.py
```

Sorties :
- `results/collective/detail.csv`
- `results/collective/summary.txt` (collective_accuracy, best_individual_accuracy, consensus_rate, revalidation_improvements)

## 7. Phase 6 — Supervision ThingsBoard (MQTT)
### 7.1 Interface et création des devices
- Ouvrir l’UI : `http://localhost:9090` (tenant par défaut : `tenant@thingsboard.org` / `tenant`).
- Créer 3 devices : `vm1`, `vm2`, `vm3` et copier leurs tokens.

### 7.2 Activer la télémétrie MQTT côté VM
- Éditer `docker-compose.yml` pour chaque service VM :
  - `TB_ENABLE: "true"`
  - `TB_HOST: "thingsboard"`
  - `TB_PORT: 1883`
  - `TB_TOKEN: "<VOTRE_TOKEN_DEVICE>"`
- Appliquer sans rebuild :

```powershell
docker compose up -d vm1 vm2 vm3
```

- Déclencher une inférence pour pousser des points :

```powershell
$p="/app/data/processed/slices/test/0/<votre_fichier>.npy"
$body=@{ slice_path=$p; patient_id="P-DEMO-001"} | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:5001/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
Invoke-WebRequest -Method POST -Uri http://localhost:5002/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
Invoke-WebRequest -Method POST -Uri http://localhost:5003/infer -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
```

- Vérifier dans ThingsBoard → Devices → `vm1`/`vm2`/`vm3` → `Latest telemetry` :
  - `vm_id`, `timestamp`, `technique`, `prediction`, `confidence`, `inference_time_ms`, `cpu_usage_pct`, `ram_usage_mb`, `patient_id`.

### 7.3 Télémétrie collective (option)
- Créer un device `collective`, récupérer son token, puis :

```powershell
$env:TB_COLLECTIVE_ENABLE="true"
$env:TB_COLLECTIVE_HOST="localhost"
$env:TB_COLLECTIVE_PORT="1883"
$env:TB_COLLECTIVE_TOKEN="<TOKEN_COLLECTIVE>"
python src\collective\run_collective.py
```

## 8. Vérifications rapides
- VM actives :

```powershell
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

- Test HTTP ThingsBoard (remplacer `<TOKEN>` par le token device) :

```powershell
$b=@{ temperature=25 } | ConvertTo-Json -Compress
Invoke-WebRequest -Method POST -Uri http://localhost:9090/api/v1/<TOKEN>/telemetry -Body $b -ContentType 'application/json'
```

## 9. Dépannage synthétique
- UI TB inaccessible sur 8080 : utiliser `http://localhost:9090`.
- Pas de télémétrie device : vérifier `TB_ENABLE="true"` + `TB_TOKEN` correct dans `docker-compose.yml`, puis `docker compose up -d vmX` et relancer `/infer`.
- Erreurs 400 sur `/infer` : s’assurer que le chemin est bien un chemin **conteneur** (préfixé par `/app/…`), pas un chemin Windows hôte.
- Builds lents : éviter `--no-cache`, ne reconstruire que si `requirements.txt` ou l’image de base changent. Les volumes montés évitent la reconstruction pour le code/source/données.

---

Ce guide reflète la structure actuelle du dépôt :
- Scripts de données : `src/dataset`
- Baseline : `src/baseline`
- Optimisations : `src/optimization`
- Déploiement & sélection : `src/deployment`
- Orchestrateur & collectif : `src/collective`
- ThingsBoard (client MQTT) : `src/thingsboard`

