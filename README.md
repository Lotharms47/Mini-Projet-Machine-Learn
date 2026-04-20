# Mini-Projet-Machine-Learning
# Prédiction de la Valeur Marchande des Joueurs de Football
---

## Problématique

Développer un modèle ML pour prédire la **valeur marchande d'un joueur de football** à partir de ses statistiques FIFA 24, dans le contexte d'une agence de scouting fictive (**ScoutAI**).

## Dataset

**EA Sports FC 24 Complete Player Dataset** — Kaggle  
~18 000 joueurs | 70+ features | Variable cible : `value_eur`

Télécharger ici : https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset  
Placer le fichier `man_players_24.csv` à la racine du projet.

## Modèles comparés

| Modèle | Type |
|---|---|
| Régression Linéaire | Baseline linéaire |
| Random Forest | Ensembliste non-linéaire |

## Structure du projet

```
football_ml_project/
├── football_value_prediction.ipynb   # Notebook principal (code complet)
├── rapport.md                        # Rapport Phase 1 + Phase 2
├── README.md                         # Ce fichier
└── man_players_24.csv                    # Dataset (à télécharger sur Kaggle)
```

## Lancer le projet

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook football_value_prediction.ipynb
```

Ou ouvrir directement dans **VS Code** avec l'extension Jupyter.

## Métriques d'évaluation

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score
- Cross-Validation R² (5-fold)
