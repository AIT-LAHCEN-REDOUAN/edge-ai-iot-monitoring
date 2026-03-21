# Phase 2 - Application des techniques d'optimisation (Q1 a Q5, P1 a P3)

## 1. Objectif

Cette phase applique les 8 techniques d'optimisation demandees (Q1 a Q5 et P1 a P3) au modele baseline MobileNetV2, puis mesure empiriquement leur impact sur :
- la precision (accuracy, F1-score),
- la taille du modele,
- la vitesse d'inference,
- la consommation memoire RAM.

## 2. Configuration experimentale

- Modele de base : `models/baseline_mobilenet_v2.pt`
- Dataset d'evaluation : split test du manifest de Phase 1
- Nombre d'echantillons testes pour Phase 2 locale : `400`
- Plateforme de mesure : machine hote CPU (resultats marques `measured_host_cpu`)

## 3. Techniques implementees

### 3.1 Quantification

- Q1 : quantification dynamique (`int8`) sur couches lineaires.
- Q2 : quantification statique PTQ (FX graph mode) avec calibration.
- Q3 : QAT avec courte phase de fine-tuning puis conversion.
- Q4 : weight-only (simulation quantize-dequantize des poids, activations en fp32).
- Q5 : precision mixte (8 bits pour couches sensibles, 4 bits pour les autres, simulation).

### 3.2 Pruning

- P1 : elagage non structure global L1 (50%).
- P2 : elagage structure par canaux sur couches convolutives (30%).
- P3 : elagage par magnitude iteratif (3 etapes, 20% par etape).

## 4. Resultats quantification (Q1-Q5)

| ID | Technique | Accuracy | F1 pondere | Taille (MB) | Inference (ms) | RAM (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | Quantification dynamique | 0.9950 | 0.9950 | 8.717 | 127.098 | 707.33 |
| Q2 | Quantification statique (PTQ) | 0.9950 | 0.9950 | 2.518 | 39.004 | 750.00 |
| Q3 | QAT | 0.9950 | 0.9950 | 2.518 | 43.294 | 1570.34 |
| Q4 | Weight-Only | 0.9950 | 0.9950 | 8.724 | 124.478 | 1565.83 |
| Q5 | Precision mixte | 0.9950 | 0.9950 | 8.724 | 126.066 | 1561.61 |

Observation principale : Q2 offre le meilleur compromis local (forte compression et latence minimale) tout en conservant la precision.

## 5. Resultats pruning (P1-P3)

| ID | Technique | Accuracy | F1 pondere | Taille stockee (MB) | Taille effective (MB) | Compression | Inference (ms) | RAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Elagage non structure | 0.5225 | 0.35863 | 8.724 | 4.429 | 1.970x | 122.957 | 736.86 |
| P2 | Elagage structure | 0.4775 | 0.30864 | 8.724 | 6.149 | 1.419x | 116.936 | 735.24 |
| P3 | Elagage par magnitude | 0.5225 | 0.35863 | 8.724 | 4.532 | 1.925x | 121.989 | 753.21 |

Observation principale : le pruning tel qu'applique ici degrade fortement la precision sur ce modele et ce jeu de test local.

## 6. Tableau comparatif complet et graphes

Livrables produits :
- Tableau comparatif complet : `results/phase2_comparative_table.csv`
- Graphe taille vs precision : `results/phase2_size_vs_precision.png`
- Graphe vitesse vs precision : `results/phase2_speed_vs_precision.png`

Interpretation synthese :
- Les techniques de quantification (surtout Q2/Q3) preservent bien la precision dans cette configuration.
- Les techniques de pruning testees necessitent un reglage plus fin (taux d'elagage, fine-tuning post-pruning) avant de pouvoir rivaliser avec la quantification.

## 7. Etat de la matrice 3x8

Le fichier `results/vm_matrix_template.csv` a ete partiellement rempli :
- VM1 : Q1 a Q5 et P1 a P3 renseignes (mesures host CPU).
- VM2/VM3 : encore `pending` (a mesurer apres deploiement dans les environnements contraints de Phase 3).

## 8. Fichiers generes Phase 2

- `src/optimization/apply_quantization.py`
- `src/optimization/apply_pruning.py`
- `results/quantization_results.csv`
- `results/quantization_results.json`
- `results/pruning_results.csv`
- `results/pruning_results.json`
- `results/phase2_comparative_table.csv`
- `results/phase2_size_vs_precision.png`
- `results/phase2_speed_vs_precision.png`
- `results/vm_matrix_template.csv`
