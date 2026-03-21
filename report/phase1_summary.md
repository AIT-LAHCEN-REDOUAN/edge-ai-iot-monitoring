# Phase 1 - Modele de base (Baseline)

## 1. Dataset medical choisi

Le jeu de donnees retenu est un dataset d'IRM cerebrales provenant de Kaggle :
https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi/data

Dans l'etat actuellement utilise dans le projet, les classes exploitables detectees automatiquement sont :
- `mr_IXI_480i`
- `mri_IXI_480_2`

Le dossier `a` a ete ignore car il ne contenait pas de donnees NIfTI exploitables pour l'entrainement.

## 2. Pretraitement applique

Le pretraitement a ete realise sur l'ensemble du dataset, avec une approche adapteee aux volumes IRM 3D :
- verification des fichiers `.nii` et exclusion des fichiers corrompus ou non supportes ;
- decoupage des volumes en echantillons 2.5D ;
- extraction de 8 slices par volume ;
- construction d'une entree a 3 canaux a partir de la slice cible et de ses deux slices voisines ;
- normalisation min-max de chaque slice dans l'intervalle `[0, 1]` ;
- redimensionnement en `224 x 224` ;
- augmentation de donnees sur l'ensemble d'entrainement uniquement : flips legers et ajustement simple d'intensite.

Cette strategie permet de transformer les volumes IRM en echantillons compatibles avec un reseau de classification 2D tout en conservant un contexte local autour de chaque coupe.

## 3. Respect des criteres du sujet

Les criteres imposes dans le PDF sont respectes :
- au moins `10 000` echantillons : `12 360` echantillons generes ;
- division `train / validation / test = 70% / 15% / 15%` ;
- au moins `2 classes` clairement identifiees.

Repartition finale :
- train : `8 648` echantillons (`69.97%`) ;
- validation : `1 856` echantillons (`15.02%`) ;
- test : `1 856` echantillons (`15.02%`).

Repartition par volumes :
- train : `1 081` volumes ;
- validation : `232` volumes ;
- test : `232` volumes.

Important : la separation a ete faite au niveau des volumes avant la generation des slices. Cela evite toute fuite de donnees entre train, validation et test.

## 4. Architecture du modele de base

Le modele baseline retenu est `MobileNetV2`, choisi pour les raisons suivantes :
- architecture legere adaptee aux contraintes embarquees ;
- taille reduite du modele ;
- bon compromis entre cout de calcul et performance ;
- pertinence pour les phases suivantes de quantification et de pruning.

Le classifieur final a ete adapte a une sortie a `2 classes`.

## 5. Parametres d'entrainement

Configuration de l'experience baseline :
- optimiseur : `Adam` ;
- fonction de cout : `CrossEntropyLoss` ;
- batch size : `8` ;
- nombre d'epochs : `3` ;
- execution : `CPU`.

## 6. Resultats obtenus

### 6.1 Metriques globales

| Metrique | Valeur |
| --- | --- |
| Accuracy | `0.99246` |
| F1-score pondere | `0.99246` |
| Taille du modele | `8.727 MB` |
| Temps moyen d'inference sur 100 images | `10.668 ms` |
| RAM maximale observee | `881.38 MB` |

### 6.2 Resultats par classe

| Classe | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| `mr_IXI_480i` | `0.98514` | `1.00000` | `0.99251` | `928` |
| `mri_IXI_480_2` | `1.00000` | `0.98491` | `0.99240` | `928` |

## 7. Discussion

Les performances du baseline sont elevees sur le jeu de test, avec une accuracy proche de `99.25%`. Le point important est que ces resultats ont ete obtenus apres correction methodologique du pipeline : la separation des donnees est maintenant realisee au niveau des volumes et non au niveau des slices.

Cette correction etait necessaire pour eviter qu'un meme volume contribue a la fois a l'entrainement et a l'evaluation. Les resultats obtenus sont donc beaucoup plus credibles pour la suite du projet.

Le choix de `MobileNetV2` est pertinent dans ce contexte, car il servira de base a la Phase 2 pour comparer les techniques d'optimisation suivantes :
- quantification ;
- pruning ;
- deploiement sur des machines virtuelles a ressources contraintes.

## 8. Livrables Phase 1 disponibles

Les fichiers produits automatiquement sont :
- `results/dataset_distribution.json`
- `results/preprocessing_manifest.csv`
- `results/baseline_summary.json`
- `results/baseline_metrics.csv`
- `results/baseline_classification_report.json`
- `models/baseline_mobilenet_v2.pt`

## 9. Limites et remarques

Deux remarques doivent etre mentionnees dans le rapport :
- `24` fichiers NIfTI corrompus ou non supportes ont ete ignores ;
- les noms de classes utilises actuellement proviennent directement des dossiers du dataset (`mr_IXI_480i`, `mri_IXI_480_2`). Si une interpretation medicale plus precise est disponible dans la documentation d'origine du dataset, il faudra la reprendre explicitement dans la version finale du rapport.

## 10. Comparaison avec la litterature

Pour respecter le critere du sujet, les resultats du baseline sont compares a deux travaux de reference en classification IRM cerebrale.

### 10.1 Article 1

Payan, A., Montana, G. (2015). *Predicting Alzheimer's disease: a neuroimaging study with 3D convolutional neural networks*. arXiv:1502.02506.

Points importants :
- approche CNN 3D sur IRM cerebrales ;
- exploitation du contexte volumique complet ;
- objectif de classification diagnostique sur donnees medicales similaires en modalite (IRM).

### 10.2 Article 2

Basaia, S., et al. (2019). *Automated classification of Alzheimer's disease and mild cognitive impairment using a single MRI and deep neural networks*. Frontiers in Neuroscience, 13:509.

Points importants :
- classification IRM avec deep learning sur cohortes cliniques ;
- performance elevee rapportee pour la discrimination des classes ;
- pipeline orientee aide au diagnostic, comparable en objectif applicatif.

### 10.3 Positionnement de notre baseline

Comparaison synthetique :

| Critere | Notre baseline (Phase 1) | Article 1 (Payan & Montana, 2015) | Article 2 (Basaia et al., 2019) |
| --- | --- | --- | --- |
| Modalite | IRM cerebrale | IRM cerebrale | IRM cerebrale |
| Type de modele | MobileNetV2 (2D, entree 2.5D) | CNN 3D | Deep neural network |
| Echelle des donnees | 12 360 echantillons (slices) | Etude neuroimagerie (cohorte clinique) | Cohortes cliniques AD/MCI |
| Metrique principale | Accuracy = 0.99246, F1 = 0.99246 | Resultats eleves rapportes | Resultats eleves rapportes |
| Contrainte embarquee | Oui (modele leger) | Non ciblee explicitement | Non ciblee explicitement |

Analyse :
- les articles de reference sont methodologiquement plus orientes diagnostic clinique pur ;
- notre baseline privilegie un reseau leger et un pipeline compatible avec les phases embarquees suivantes (quantification, pruning, deploiement VM) ;
- en termes de protocole, la separation par volume appliquee dans notre pipeline renforce la validite de l'evaluation ;
- la comparaison quantitative stricte reste limitee par la difference de cohortes, protocoles et definitions de classes entre jeux de donnees.

## 11. Bibliographie

[1] Payan, A., & Montana, G. (2015). *Predicting Alzheimer's disease: a neuroimaging study with 3D convolutional neural networks*. arXiv:1502.02506. https://arxiv.org/abs/1502.02506

[2] Basaia, S., Agosta, F., Wagner, L., Canu, E., Magnani, G., Santangelo, R., Filippi, M. (2019). *Automated classification of Alzheimer's disease and mild cognitive impairment using a single MRI and deep neural networks*. Frontiers in Neuroscience, 13, 509. https://doi.org/10.3389/fnins.2019.00509
