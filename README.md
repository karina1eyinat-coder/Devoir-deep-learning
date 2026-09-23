# CNN from scratch vs Transfer Learning — Cats vs Dogs

## Objectif

Comparer un modèle **CNN entraîné from scratch** et un modèle en **transfert d'apprentissage** (ResNet18 pré-entraîné) sur le jeu de données Cats vs Dogs, et mesurer l'impact du transfer learning sur la convergence, la performance et la robustesse.

## Environnement

```bash
pip install -r requirements.txt
```

Ou avec conda :

```bash
conda create -n catsdogs python=3.10
conda activate catsdogs
pip install -r requirements.txt
```

Testé avec Python 3.10, PyTorch >= 2.0.

## Organisation des données

Télécharger le jeu de données Cat vs Dog (déjà séparé en train/test) :
https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip

Dézipper à la racine du projet de façon à obtenir :

```
Cat_Dog_data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

Ce dossier n'est **pas** versionné (voir `.gitignore`).

## Lancer les entraînements

Ouvrir `notebook.ipynb` (Google Colab ou Jupyter local) et exécuter les cellules dans l'ordre :

1. Montage des données (`Cat_Dog_data/train`, `Cat_Dog_data/test`)
2. Section "Reproductibilité (seed)" — seed = 42
3. Section "Vérification du GPU"
4. Section "Données : split train/val + data augmentation"
5. **Expérience A — CNN from scratch** : entraîné successivement avec `SGD` (lr=0.01, momentum=0.9, StepLR) puis `Adam` (lr=1e-3, CosineAnnealingLR), 10 époques chacun, batch_size=32.
6. **Expérience B — Transfer Learning (ResNet18)** : couches convolutionnelles gelées, tête `fc` réentraînée, mêmes optimiseurs et hyperparamètres que ci-dessus.
7. Comparaison des courbes + matrice de confusion.
8. Rechargement des checkpoints `.pth` et évaluation finale sur le jeu de test.

### Hyperparamètres clés

| Paramètre | CNN from scratch | Transfer Learning |
|---|---|---|
| Batch size | 32 | 32 |
| Époques | 10 | 10 |
| Optimiseurs testés | SGD (lr=0.01) / Adam (lr=1e-3) | SGD (lr=0.01) / Adam (lr=1e-3) |
| Scheduler | StepLR / CosineAnnealingLR | StepLR / CosineAnnealingLR |
| Régularisation | BatchNorm + Dropout (0.5 / 0.25) | Dropout (0.5) sur la tête |
| Gel des couches | — | Backbone ResNet18 gelé |

## Évaluer / recharger un modèle

Les meilleurs checkpoints sont sauvegardés localement pendant l'entraînement (non versionnés) :

- `scratch_sgd_best.pth`, `scratch_adam_best.pth`
- `tl_sgd_best.pth`, `tl_adam_best.pth`

La section "Persistance du modèle" du notebook recharge automatiquement le meilleur checkpoint de chaque expérience et calcule les métriques finales sur le jeu de test.

## Résultats

*(À compléter après exécution du notebook — copier les valeurs finales imprimées dans la section 10, et insérer les images `comparison_curves.png` et `confusion_*.png` générées.)*

| Expérience | Accuracy (test) | Précision (test) | Recall (test) | Loss (test) |
|---|---|---|---|---|
| CNN from scratch (meilleur optimiseur) | — | — | — | — |
| Transfer Learning ResNet18 (meilleur optimiseur) | — | — | — | — |

![Courbes de comparaison](comparison_curves.png)

### Analyse (2–3 paragraphes)

*(À compléter — voir la section 11 du notebook pour la trame d'analyse : convergence, performance finale, robustesse/généralisation.)*

## Limites & pistes d'amélioration

- *(à compléter)*

## Reproductibilité

Seed fixée à 42 pour Python, NumPy et PyTorch (CPU/GPU) — voir section 2 du notebook.
