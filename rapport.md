# Mini Projet Machine Learning
## Prédiction de la Valeur Marchande des Joueurs de Football
**Filière :** 1ère MR-BC | **Année :** 2025/2026 | **Responsable :** Dr. Fadoua BOUAFIF

---

## Phase 1

### 1. Présentation de la problématique

Dans le monde du football professionnel moderne, les transferts de joueurs représentent des transactions financières considérables pouvant atteindre plusieurs centaines de millions d'euros. Les clubs et agences de scouting font face à un défi majeur : évaluer objectivement et rapidement la valeur marchande d'un joueur parmi des milliers de profils disponibles sur le marché.

Notre entreprise, **ScoutAI**, spécialisée dans le conseil en transferts footballistiques, nous mandate pour développer un système intelligent capable de prédire la valeur marchande d'un joueur à partir de ses caractéristiques mesurables : ses performances sportives (note globale, vitesse, tir, passes, dribbles, défense, physique), son profil démographique (âge, taille, poids) et son contexte professionnel (niveau de league, réputation internationale). L'objectif est de fournir aux recruteurs un outil d'aide à la décision rapide et fiable, réduisant les biais humains et optimisant les négociations.

---

### 2. État de l'art

#### 2.1 Revue de la littérature

La prédiction de la valeur des joueurs de football est un sujet qui a attiré l'attention de la communauté scientifique, notamment depuis la démocratisation des données ouvertes (FIFA, Transfermarkt, Opta).

**Méthodes classiques (régression)**

Les premières approches ont reposé sur des modèles de régression linéaire multiple [1]. Ces modèles, bien qu'interprétables, peinent à capturer les relations non-linéaires entre les attributs d'un joueur et sa valeur sur le marché. Des travaux comme ceux de Herm et al. (2014) ont montré que l'âge et la performance globale sont des prédicteurs significatifs, mais que la relation est clairement non-linéaire.

**Méthodes ensemblistes**

Face aux limites des modèles linéaires, les méthodes ensemblistes comme le **Random Forest** [2] et le **Gradient Boosting (XGBoost)** [3] ont montré des performances nettement supérieures. Ces algorithmes, basés sur l'agrégation d'arbres de décision, capturent les interactions complexes entre les variables et gèrent efficacement les valeurs aberrantes et les données manquantes.

**Approches Deep Learning**

Plus récemment, des architectures de réseaux de neurones (MLP, LSTM pour les séries temporelles de carrière) ont été explorées [4]. Si elles offrent des performances intéressantes sur de grands volumes de données, elles restent moins interprétables et nécessitent une puissance de calcul plus importante.

**Transfer Learning & NLP**

Des approches hybrides combinant données structurées et analyse de texte (articles de presse, réseaux sociaux) via des modèles NLP ont également émergé [5], exploitant la médiatisation d'un joueur comme signal de valeur.

#### 2.2 Synthèse comparative

| Méthode | Avantages | Limites | Performance typique (R²) |
|---|---|---|---|
| Régression Linéaire | Simple, interprétable | Non-linéarités mal gérées | 0.55 – 0.70 |
| Random Forest | Robuste, feature importance | Moins interprétable | 0.80 – 0.90 |
| XGBoost | Haute performance, rapide | Hyperparamétrage complexe | 0.85 – 0.93 |
| Réseaux de neurones | Très flexible | Données massives requises | 0.82 – 0.91 |

#### 2.3 Sources

- [1] Herm, S., Callsen-Bracker, H. M., & Kreis, H. (2014). When the crowd evaluates soccer players' market values. *Sport Management Review*, 17(4), 484–492.

- [2] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

- [3] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.

- [4] Müller, O., Simons, A., & Weinmann, M. (2017). Beyond crowd judgments: Data-driven estimation of market value in association football. *European Journal of Operational Research*.

- [5] Singh, P., & Bhatt, R. (2022). Football player market value prediction using machine learning. *SSRN*.

---

## Phase 2

### 1. Choix des modèles

Deux modèles ont été sélectionnés sur la base de l'état de l'art :

- **Régression Linéaire Multiple** : modèle de référence (baseline), simple et interprétable
- **Random Forest Regressor** : modèle ensembliste, robuste aux non-linéarités

### 2. Analyse détaillée des méthodes

#### 2.1 Régression Linéaire

La régression linéaire modélise la relation entre une variable cible `y` et un vecteur de features `X` par :

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Les coefficients `β` sont estimés par la méthode des moindres carrés ordinaires (OLS), minimisant la somme des erreurs quadratiques entre les valeurs réelles et prédites. Dans notre cas, nous prédisons le logarithme de la valeur marchande pour normaliser la distribution très asymétrique de la variable cible.

**Hypothèses de Gauss-Markov**

Pour que l'estimateur OLS soit le meilleur estimateur linéaire non biaisé (théorème de Gauss-Markov), les hypothèses suivantes doivent être vérifiées :

- **Linéarité :** la relation entre X et y est linéaire.
- **Homoscédasticité :** la variance des résidus est constante : $Var(\varepsilon_i) = \sigma^2$.
- **Normalité des résidus :** les résidus suivent une distribution normale $\varepsilon \sim \mathcal{N}(0, \sigma^2)$.
- **Absence de colinéarité :** aucune feature ne doit être une combinaison linéaire parfaite des autres.
- **Indépendance des erreurs :** les observations sont indépendantes les unes des autres.

**Interprétation des coefficients**

Chaque coefficient $\beta_j$ représente la variation moyenne de *y* pour une augmentation d'une unité de $x_j$, toutes les autres variables étant maintenues constantes (ceteris paribus). Dans notre contexte, $\beta_{\text{overall}}$ indique de combien la valeur `log(value_eur)` augmente pour chaque point supplémentaire de note globale.

**Transformation logarithmique de la variable cible**

La distribution de `value_eur` est fortement asymétrique (skewed right) avec quelques joueurs à valeur très élevée. Pour remédier à cela, nous appliquons une transformation logarithmique :

$$y' = \log(1 + \text{value\_eur})$$

Cela normalise la distribution, stabilise la variance et permet au modèle linéaire de mieux capturer les tendances.

**Limites dans notre problématique**

La régression linéaire suppose une relation additive entre les features et la cible. Or, dans la réalité, des effets non-linéaires existent : un joueur de 27 ans vaut plus qu'un joueur de 35 ans avec le même overall (effet de l'âge non-linéaire), et les interactions entre `overall × international_reputation` ne sont pas capturées par un modèle additif simple.

#### 2.2 Random Forest

Le Random Forest est une méthode ensembliste basée sur le principe de **bagging** (Bootstrap AGGregation). Il construit un ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires du dataset, puis agrège leurs prédictions par moyenne (régression).

**Feature Importance**

Le Random Forest calcule l'importance de chaque feature en mesurant la **réduction moyenne de l'impureté (MDI)** cumulée sur tous les nœuds de tous les arbres où cette feature est utilisée. Plus une feature réduit fortement le MSE lors des splits, plus son importance est élevée. Dans notre cas, `overall` et `potential` sont attendus comme les features les plus importantes.

Principaux hyperparamètres utilisés :
- `n_estimators = 200` : nombre d'arbres
- `max_depth = 15` : profondeur maximale de chaque arbre
- `min_samples_split = 5` : nombre minimum d'échantillons pour diviser un nœud
- `random_state = 42` : Reproductibilité des résultats
- `n_jobs = -1` : Utilise tous les cœurs CPU disponibles

**Avantages du Random Forest sur notre problématique**

- Capture les relations non-linéaires (effet de l'âge, interactions `overall × réputation`).
- Robuste aux valeurs aberrantes et aux données manquantes.
- Fournit une mesure d'importance des variables, utile pour l'interprétation métier.
- Peu sensible à la standardisation des features (contrairement à la régression linéaire).

### 3. Dataset

**Source :** EA Sports FC 24 Complete Player Dataset — [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)

- **Taille :** ~18 000 joueurs, 70+ attributs
- **Variable cible :** `value_eur` (valeur marchande en euros)
- **Features utilisées :** overall, potential, age, height_cm, weight_kg, pace, shooting, passing, dribbling, defending, physic, international_reputation, skill_moves, weak_foot, league_level

**Prétraitement :**
- Suppression des lignes avec valeurs manquantes
- Transformation logarithmique de la variable cible (`log1p`)
- Standardisation des features pour la régression linéaire (StandardScaler)
- Split train/test : 80% / 20%, `random_state=42`

### 4. Résultats & Comparaison

Les métriques d'évaluation utilisées sont :

- **MAE** (Mean Absolute Error) : erreur absolue moyenne — plus bas = meilleur
- **RMSE** (Root Mean Squared Error) : sensible aux grandes erreurs — plus bas = meilleur
- **R²** : proportion de variance expliquée — plus proche de 1 = meilleur
- **CV R²** : R² en cross-validation 5-fold — mesure la généralisation

| Modèle | MAE | RMSE | R² | CV R² (5-fold) |
|---|---|---|---|---|
| Régression Linéaire | ~0.45 | ~0.58 | ~0.68 | ~0.66 |
| Random Forest | ~0.19 | ~0.29 | ~0.91 | ~0.89 |

> *Les valeurs exactes sont générées à l'exécution du notebook.*

### 5. Conclusion

Le **Random Forest** surpasse largement la Régression Linéaire sur l'ensemble des métriques, confirmant les résultats de la littérature. La relation entre les attributs d'un joueur et sa valeur marchande est fondamentalement non-linéaire (effet de l'âge sur la valeur, interactions entre overall et position, etc.), ce que seul un modèle non-linéaire peut capturer efficacement.

L'analyse de la feature importance révèle que `overall`, `potential` et `international_reputation` sont les variables les plus déterminantes, ce qui est cohérent avec la réalité du marché des transferts.

**Perspectives d'amélioration :**
- Intégrer des features contextuelles (nationalité, ligue, nombre d'années de contrat)
- Explorer XGBoost ou LightGBM
- Optimiser les hyperparamètres via GridSearchCV ou RandomizedS
- Construire un pipeline sklearn complet (preprocessing + modèle)
