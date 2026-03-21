# IOT Project - Inference Medicale Embarquee

Ce depot suit le sujet `projet_iot.pdf`.

## Objectif

Comparer des techniques d'optimisation de modeles de deep learning
(quantification et pruning), puis deployer sur 3 VM contraintes et
mettre en place une intelligence collective avec supervision ThingsBoard.

## Structure

- `data/`: jeux de donnees (non versionnes si volumineux)
- `models/`: modeles entraines et optimises
- `src/baseline/`: entrainement du modele de base (Phase 1)
- `src/optimization/`: quantification Q1-Q5 et pruning P1-P3 (Phase 2)
- `src/deployment/`: bench sur VM (Phase 3)
- `src/orchestrator/`: agregation des predictions (Phase 5)
- `thingsboard/`: client MQTT + dashboards JSON (Phase 6)
- `results/`: matrices CSV et metriques
- `report/`: rapport final
- `scripts/`: scripts PowerShell utilitaires

## Demarrage rapide

1. Installer les dependances:

```powershell
"c:/Users/houdh/Desktop/Data Science/IOT/PFA/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

2. Lancer le squelette baseline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

3. Lancer le squelette orchestrateur:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_orchestrator.ps1
```

## Livrables a produire

- Scripts d'entrainement et optimisation
- Matrice 3x8 dans `results/vm_matrix_template.csv`
- Orchestrateur de vote pondere
- Integration ThingsBoard
- Rapport technique final (10-12 pages)

