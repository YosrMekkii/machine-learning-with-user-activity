# Importation des bibliothèques nécessaires
import pandas as pd  # Manipulation de données sous forme de DataFrame
import numpy as np  # Gestion des tableaux et calculs numériques
import matplotlib.pyplot as plt  # Visualisation des données
import seaborn as sns  # Amélioration des visualisations avec Matplotlib
from datetime import datetime  # Gestion des dates et heures
from sklearn.model_selection import train_test_split  # Séparation des données en ensembles d'entraînement et de test
from sklearn.preprocessing import StandardScaler  # Standardisation des données
from sklearn.neighbors import KNeighborsClassifier  # Modèle K-Nearest Neighbors
from sklearn.svm import SVC  # Support Vector Machine pour la classification
from sklearn.tree import DecisionTreeClassifier  # Arbre de décision pour la classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Évaluation des modèles
import os
def convertir_duree_en_mois(duree_str):
    """Convertit une durée de formation exprimée en texte en un nombre moyen de mois."""
    if pd.isna(duree_str):  # Vérifie si la valeur est NaN (valeur manquante)
        return np.nan  # Retourne NaN si la valeur est manquante
    # Extraction des valeurs numériques des durées (ex: "3 mois, 6 mois" → [3, 6])
    mois = [int(s.split()[0]) for s in duree_str.split(',') if s.split()[0].isdigit()]
    return np.mean(mois) if mois else np.nan  # Retourne la moyenne des valeurs extraites

def pretraiter_donnees(df):
    """Effectue le prétraitement des données en transformant les dates et les durées en valeurs numériques."""
    df_processed = df.copy()  # Copie du DataFrame pour éviter de modifier l'original
    # Suppression des doublons
    df_processed = df_processed.drop_duplicates()
    # Transformation des dates en nombre de jours écoulés depuis la dernière activité
    if 'Dernière_activité' in df.columns:
        df_processed['jours_depuis_derniere_activite'] = df_processed['Dernière_activité'].apply(
            lambda x: (datetime.now() - pd.to_datetime(x)).days if pd.notna(x) else np.nan
        )

    # Conversion des durées de formation en mois
    if 'Durée_apprentissage_par_compétence' in df.columns:
        df_processed['Durée_apprentissage_par_compétence'] = df_processed['Durée_apprentissage_par_compétence'].apply(
            convertir_duree_en_mois
        )
    
    # Sélection des caractéristiques à utiliser
    features = ['jours_depuis_derniere_activite', 'Durée_apprentissage_par_compétence']
    if 'Score' in df.columns:  # Ajout du score si disponible
        features.append('Score')
    
    X = df_processed[features].copy()
    
    # Remplacement des valeurs manquantes par la moyenne de chaque colonne
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())
    
    return X  # Retourne le DataFrame prétraité

def categoriser_utilisateurs(df, seuil_actif=7, seuil_moderement_actif=30):
    """Catégorise les utilisateurs selon leur activité et leur score."""
    if df.empty:
        return pd.Series(dtype=str)  # Retourne une série vide si le DataFrame est vide
    
    # Calcul des moyennes globales pour comparer les utilisateurs
    moyenne_duree = df['Durée_apprentissage_par_compétence'].mean()
    moyenne_score = df['Score'].mean()
    
    # Fonction interne pour classer chaque utilisateur
    def determiner_categorie(row):
        jours, duree, score = row['jours_depuis_derniere_activite'], row['Durée_apprentissage_par_compétence'], row['Score']
        if jours < seuil_actif and duree > moyenne_duree and score > moyenne_score:
            return 'Actif'
        elif jours < seuil_moderement_actif:
            return 'Modérément Actif'
        return 'Inactif'
    
    return df.apply(determiner_categorie, axis=1)  # Applique la classification à chaque ligne

def entrainer_modeles(X_train, y_train):
    """Entraîne plusieurs modèles de classification et les retourne."""
    modeles = {
        'KNN': KNeighborsClassifier(n_neighbors=3),  # Algorithme des k plus proches voisins
        'SVM': SVC(kernel='linear'),  # Support Vector Machine avec un noyau linéaire
        'Decision Tree': DecisionTreeClassifier()  # Arbre de décision
    }
    
    # Entraînement de chaque modèle sur les données d'entraînement
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
    
    return modeles  # Retourne les modèles entraînés

import os

def evaluer_modeles(modeles, X_test, y_test):
    """
    Évalue les performances de plusieurs modèles de classification sur des données de test.

    Paramètres :
    - modeles (dict) : Un dictionnaire contenant les modèles sous la forme {nom: modèle}
    - X_test (array) : Les données de test (features)
    - y_test (array) : Les étiquettes réelles des données de test

    Retourne :
    - resultats (dict) : Un dictionnaire contenant les scores d'exactitude (accuracy) de chaque modèle
    """
    
    resultats = {}  # Dictionnaire pour stocker les scores d'accuracy de chaque modèle
    accuracy_scores = {}  # Dictionnaire pour stocker les scores pour la visualisation

    # Création du dossier "figures" s'il n'existe pas pour stocker les images générées
    os.makedirs("figures", exist_ok=True)

    # Création d'une figure contenant plusieurs sous-graphiques pour afficher les matrices de confusion
    fig, axes = plt.subplots(1, len(modeles), figsize=(15, 5))  

    # Boucle sur chaque modèle pour évaluer ses performances
    for i, (nom, modele) in enumerate(modeles.items()):
        # Prédiction des étiquettes sur les données de test
        y_pred = modele.predict(X_test)
        
        # Calcul de l'exactitude du modèle
        accuracy = accuracy_score(y_test, y_pred)
        
        # Stockage des résultats dans les dictionnaires
        resultats[nom] = accuracy
        accuracy_scores[nom] = accuracy
        
        # Affichage du rapport de classification du modèle
        print(f'\nModèle: {nom}')
        print(classification_report(y_test, y_pred))

        # Affichage de la matrice de confusion sous forme graphique avec seaborn
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Matrice - {nom}')  # Titre de la matrice
        axes[i].set_xlabel('Prédictions')  # Nom de l'axe des abscisses
        axes[i].set_ylabel('Réelles')  # Nom de l'axe des ordonnées

    # Ajustement de la disposition des sous-graphiques pour éviter les chevauchements
    plt.tight_layout()

    # Affichage des matrices de confusion
    plt.show(block=True)

    # Enregistrement de la figure contenant les matrices de confusion
    confusion_matrix_path = "figures/matrices_confusion.png"
    fig.savefig(confusion_matrix_path)
    print(f"✅ Matrices de confusion enregistrées sous '{confusion_matrix_path}'")

    # Génération du graphique de comparaison des scores des modèles
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette='viridis')
    plt.title('Comparaison des performances des modèles')  # Titre du graphique
    plt.xlabel('Modèles')  # Nom de l'axe des abscisses
    plt.ylabel('Précision (Accuracy)')  # Nom de l'axe des ordonnées
    plt.ylim(0, 1)  # Définition de l'échelle de l'axe des ordonnées entre 0 et 1

    # Affichage du graphique de comparaison
    plt.show(block=True)

    # Enregistrement du graphique de comparaison des performances
    performance_graph_path = "figures/comparaison_performances.png"
    plt.savefig(performance_graph_path)
    print(f"✅ Comparaison des performances enregistrée sous '{performance_graph_path}'")

    return resultats  # Retourne les scores des modèles

def main(data_path):
    """Fonction principale qui charge les données, effectue le prétraitement, entraîne et évalue les modèles."""
    try:
        # Liste d'encodages à tester pour charger le fichier CSV
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, encoding=encoding, delimiter=';')
                print(f"✅ Fichier chargé avec succès en utilisant l'encodage {encoding}")
                break
            except UnicodeDecodeError:
                print(f"❌ L'encodage {encoding} n'a pas fonctionné")
                if encoding == encodings[-1]:
                    raise Exception("Impossible de trouver un encodage compatible")
                continue
        
        # Vérification que le dataset n'est pas vide
        if df.empty:
            print("⚠ Le fichier CSV est vide. Vérifiez votre dataset.")
            return
        
        X = pretraiter_donnees(df)  # Prétraitement des données
        y = categoriser_utilisateurs(X)  # Catégorisation des utilisateurs
        
        # Séparation des données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardisation des données pour améliorer la performance des modèles
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        modeles = entrainer_modeles(X_train, y_train)  # Entraînement des modèles
        resultats = evaluer_modeles(modeles, X_test, y_test)  # Évaluation des modèles
        
        print("\nPerformances des modèles:")
        for nom, score in resultats.items():
            print(f"{nom}: {score:.2%}")  # Affichage des scores en pourcentage
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")

# Exécution du script si lancé directement
if __name__ == "__main__":
    main("bigdata_user_dataset_1200.csv")
