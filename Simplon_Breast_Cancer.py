#!/usr/bin/env python
# coding: utf-8

# # Introduction à la classification binaire
# # Application au diagnostic du cancer du sein
# 
# Travail en groupe 3/4, sur la semaine 50
# 
# ## Projet Breast Cancer Wisconsin
# Les données: 
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# 
# Nous allons étudier le concept de classification qui est avec la régression l'autre tâche principale du Machine Learning supervisé. Nous allons particulièrement nous concentrer sur la classification binaire et l'illustrer sur un exemple d'application concret, le diagnostic du cancer du sein. En outre, nous allons découvrir et utiliser la régression logistique comme modèle de Machine Learning pour effectuer notre tâche de classification binaire.
# 

# ## Contexte
# Le Breast Cancer Wisconsin (diagnostic) Dataset est un jeu de données classique du Machine Learning. Il est composé de 569 exemples, chaque exemple étant défini par 30 caractéristiques ; ces caractéristiques correspondent principalement aux propriétés géométriques mesurées sur les cellules issues de la biopsie. Chaque exemple est classé dans la catégorie "Benign" (n=357) si la tumeur est bénigne et dans la catégorie "Malignant" (n=212) si la tumeur est maline. Le nombre de catégories de la classification est de 2, on parle alors de classification binaire. La finalité de ce travail est ainsi d'entrainer un modèle de Machine Learning à identifier le type de tumeur (bénigne ou maline) en fonction des propriétés géométriques mesurées sur les cellules issues de la biopsie.
# 

# ### concepts de base de la classification binaire :
# Lister:
# 
#     En quoi consiste la classification binaire ?
#     Qu'est-ce qu'une matrice de confusion ?
#         Qu'est-ce qu'un faux négatif ? un vrai négatif ? un faux positif ? un vrai positif ?
#     Qu'est-ce que le taux de classification (accuracy) ?
#     Qu'est-ce que le rappel (recall) ?
#     Qu'est-ce que la précision (precision) ?
#     Qu'est-ce que le F1-score ?
#     Qu'est-ce qu'une courbe ROC ?
#         Qu'est-ce que l'Area Under the Curve (AUC) ?
# 
La classification binaire consiste à entrainer le modèle de machine learning à identifier 2 catégories en fonction des données quantitatives que nous avons dans notre dataset.
Une matrice de confusion est un outil permettant de mesurer les performances d’un modèle de Machine Learning en vérifiant notamment à quelle fréquence ses prédictions sont exactes par rapport à la réalité dans des problèmes de classification.Ces résultats peuvent être l’indication correcte d’une prédiction positive comme  » vraie positive  » (true positive) et d’une prédiction négative comme  » vraie négative  » (true negative), ou une prédiction positive incorrecte comme  » fausse positive  » (false positive) et une prédiction négative incorrecte comme  » fausse négative  » (false negative).L’indicateur le plus simple est l’accuracy : il indique le pourcentage de bonnes prédictions. C’est un très bon indicateur parce qu’il est très simple à comprendre.Pour compléter l’accuracy, on calcule également le recall : il se concentre uniquement sur les id  qui ont réellement le cancer et donne une indication sur la part de faux négatifs. Les faux négatifs ce sont les id qui ont le cancer mais qui ne sont pas détectés par le score. Concrètement ce sont des id que vous ne détectez pas et pour lesquels vous ne pourrez pas agir pour éviter leur départ.Enfin, un 3ème indicateur vient compléter l’accuracy et le recall, c’est la precision : il se concentre uniquement sur les id pour lesquels le modèle a prédit un cancer et donne une indication sur les faux positifs. Les faux positifs ce sont les id pour lesquels le score a prédit un cancer mais qui ont une tumeur begnine. Le F1 Score permet d’effectuer une bonne évaluation de la performance de notre modèle.
Le F1-Score combine subtilement la précision et le rappel. Il est intéressant et plus intéressant que l’accuracy car le nombre de vrais négatifs (tn) n’est pas pris en compte. Courbe ROC est utilisé pour evaluer les performances du modèle.
# Area Under the Curve (AUC) , L’AUC correspond à la probabilité pour qu’un événement positif soit classé comme positif par le test sur l’étendue des valeurs seuil possibles. 
# ### Etude
# décrire:
# 
#     Qu'est-ce qu'une régression logistique ?
#     Qu'est-ce que le Feature Scaling ?
#         A quoi sert-il et quels sont ses avantages ?
#         Qu'est-ce que la normalisation des données ?
#         Qu'est-ce que la standardisation des données ?
#         Comment l'utilise-t-on lorsque l'on a un jeu de train et un jeu de test ?
# 
Un modèle de régression logistique permet aussi de prédire la probabilité qu’un événement arrive (valeur de 1) ou non (valeur de 0) à partir de l’optimisation des coefficients de régression. Ce résultat varie toujours entre 0 et 1. Lorsque la valeur prédite est supérieure à un seuil, l’événement est susceptible de se produire, alors que lorsque cette valeur est inférieure au même seuil, il ne l’est pas.
# In[61]:


Le Feature Scaling  est une bonne pratique, pour ne pas dire obligatoire, lors de la modélisation avec du Machine Learning.

Les algorithmes pour lesquels le feature scaling s’avère nécessaire, sont ceux pour lesquels il faudra

    Calculer un vecteur de poids (weights) theta
    Calculer des distances pour déduire le degrée de similarité de deux items
    Certains algorithmes de Clustering

Plus concrétement, voici une liste d’algorithmes non exhaustive pour lesquels il faudra procéder au Feature Scaling :

    Logistic Regression
    Regression Analysis (polynomial, multivariate regression…)
    Support Vector Machines (SVM)
    K-Nearest Neighbors (KNN)
    K-Means (clustering…)
    Principal Component Analysis (PCA)


# In[ ]:


La Normalisation

Min-Max Scaling peut- être appliqué quand les données varient dans des échelles différentes. A l’issue de cette transformation, les features seront comprises dans un intervalle fixe [0,1]. Le but d’avoir un tel intervalle restreint est de réduire l’espace de variation des valeurs d’une feature et par conséquent réduire l’effet des outliers.

La normalisation peut- être effectuée par la technique du Min-Max Scaling. La transformation se fait grâce à la formule suivante :

    \[X_{normalise} = \frac{X - X_{min}}{X_{max} - X_{min}}\]

Avec :

    X_{min} : la plus petite valeur observée pour la feature X
    X_{min} : la plus grande valeur observée pour la feature X
    X : La valeur de la feature qu’on cherche à normaliser


# In[ ]:


La Standardisation

La standardisation (aussi appelée Z-Score normalisation  à ne pas confondre avec la normalisation du paragraphe précendent) peut- être appliquée quand les input features répondent à des distributions normales (Distributions Gaussiennes) avec des moyennes et des écart-types différents. Par conséquent, cette transformation aura pour impact d’avoir toutes nos features répondant à la même loi normale X \sim \mathcal{N} (0, \, 1).

La standardisation peut également être appliquée quand les features ont des unités différentes.

La Standardisation est le processus de transformer une feature en une autre qui répondra à la loi normale (Gaussian Distribution) X \sim \mathcal{N} (\mu, \, \sigma) avec :

    \mu = 0  La moyenne de la loi de distribution
    \sigma = 1 est l’Écart-type (Standard Deviation)

La formule de standardisation d’une feature est la suivante :

        \[z = \frac{ x - \mu} {\sigma} \]

avec :

    x la valeur qu’on veut standardiser (input variable)
    \mu la moyenne (mean) des observations pour cette feature
    \sigma est l’ecart-type (Standard Deviation) des observations pour cette feature


# ## Implémentation python
# Entrainement d'un modèle de type régression logistique à diagnostiquer l'absence ou la présence d'un cancer du sein
# Les étapes:
# 
#     Importation des librairies Python
#     Chargement des données du Breast Cancer Wisconsin (diagnostic) Dataset
#     Mise au format Numpy des données
#         Par défaut, patients sains = 0, patients malades = 1.
#     Echantillonnage des données
#         NB test_size = 113
#     Afficher sous forme d'histogrammes la distribution du jeu de données initial, du jeu de train et du jeu de test en fonction de chaque catégorie (bénigne et maline)
#     Effectuer le Feature Scaling
#     Entrainer le modèle de régression logistique
#         model = LogisticRegression(C = 0.1, max_iter = 10000)
#     Calculer et afficher les performances obtenues sur le jeu d'apprentissage
#         Matrice de confusion
#         Taux de classification, Rappel, Précision et F1-Score
#         Courbe ROC, AUC
#     Calculer et afficher les performances obtenues sur le jeu de test
#         Matrice de confusion
#         Taux de classification, Rappel, Précision et F1-Score
#         Courbe ROC, AUC

# In[ ]:


#Importation librairies Python

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import seaborn as sns
import plotly 
import plotly.graph_objects as go
import chart_studio
import chart_studio.plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
plotly.offline.init_notebook_mode()
import plotly.express as px
from pySankey.sankey import sankey
init_notebook_mode(connected=True)         # initiate notebook for offline plot
from plotly.subplots import make_subplots
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
import plotly.offline as pyo
pyo.init_notebook_mode()
from IPython.display import Image, HTML, display, SVG
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pywaffle import Waffle
import random
from pandas_profiling import ProfileReport
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
        'width': 1600,
        'height': 900,
        'scroll': True,
})

display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


#Chargement jeu de données Breast Cancer

df = pd.read_csv('data.csv')
df


# In[ ]:


#Exploration Data Analysis

profile = ProfileReport(df, title='Analyse du fichier Breast Cancer', html={'style':{'full_width':True}})
profile.to_notebook_iframe()


# In[ ]:


#Valeurs manquantes :
na_values=msno.matrix(df,figsize=(10,3))
na_values


# In[ ]:


#Retrait de la colonne avec valeurs manquantes
# Drop last column of a dataframe
df = df.iloc[: , :-1]
df


# In[ ]:



#Mise au format Numpy des données
    #Par défaut, patients sains = 0, patients malades = 1

df= df.replace({'M':1,'B':0})
df


# In[ ]:


df.columns


# In[ ]:


#Echantillonage des données
# Load module from scikit-learn
from sklearn.model_selection import train_test_split


# In[ ]:


# Split data into 2 parts : train & test
X=df[['id', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
y=df[['diagnosis']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.198, random_state=42) 


# In[ ]:


# Check the size of subset of data
print("The length of the initial dataset is :", len(X))
print("The length of the train dataset is   :", len(X_train))
print("The length of the test dataset is    :", len(X_test))


# In[59]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

modele_logit = LogisticRegression(penalty='none',solver='newton-cg')
modele_logit.fit(X,y)


# In[56]:


#comprendre les coefficients du modèle, scikit-learn stocke les informations dans .coef_, nous allons les afficher de manière plus agréable dans un DataFrame avec la constante du modèle :
coef = pd.DataFrame(np.concatenate([modele_logit.intercept_.reshape(-1,1),
                             modele_logit.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T

coef


# In[62]:


#importation de l'outil 
from statsmodels.tools import add_constant 
 
#données X avec la constante 
XTrainBis = sm.tools.add_constant(X_train) 
 
#vérifier la structure 
print(XTrainBis.info())


# In[63]:


#visualisation des premières lignes de la structure 
#premières lignes 
print(XTrainBis.head()) 


# In[ ]:




