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
from sklearn.linear_model import LinearRegression, LogisticRegression  # Régression linéaire et logistique
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score  # Évaluation des modèles
import xgboost as xgb  # XGBoost pour classification et régression
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
    # Convertir les étiquettes textuelles en valeurs numériques pour certains modèles
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    modeles = {
        'KNN': KNeighborsClassifier(n_neighbors=3),  # Algorithme des k plus proches voisins
        'SVM': SVC(kernel='linear'),  # Support Vector Machine avec un noyau linéaire
        'Decision Tree': DecisionTreeClassifier(),  # Arbre de décision
        'Logistic Regression': LogisticRegression(max_iter=1000),  # Régression logistique 
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)))  # XGBoost pour classification
    }
    
    # Entraînement de chaque modèle sur les données d'entraînement
    for nom, modele in modeles.items():
        if nom == 'XGBoost':
            modele.fit(X_train, y_train_encoded)
        else:
            modele.fit(X_train, y_train)
    
    # Pour la régression linéaire, on va prédire un score numérique (par exemple le score)
    # puis utiliser un seuil pour classer
    return modeles, le  # Retourne les modèles entraînés et l'encodeur

def evaluer_regression_lineaire(X_train, X_test, y_train, y_test):
    """Évalue les performances de la régression linéaire et affiche les résultats."""
    # Encodage des catégories en valeurs numériques
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Modèle de régression linéaire simple (une seule variable explicative)
    regressor_simple = LinearRegression()
    regressor_simple.fit(X_train[:, 0:1], y_train_encoded)  # On utilise seulement la première feature
    
    # Modèle de régression linéaire multiple (toutes les variables explicatives)
    regressor_multiple = LinearRegression()
    regressor_multiple.fit(X_train, y_train_encoded)
    
    # Prédictions
    y_pred_simple = regressor_simple.predict(X_test[:, 0:1])
    y_pred_multiple = regressor_multiple.predict(X_test)
    
    # Arrondir les prédictions pour les transformer en classes
    y_pred_simple_class = np.round(y_pred_simple).astype(int)
    y_pred_multiple_class = np.round(y_pred_multiple).astype(int)
    
    # Corriger les valeurs hors limites
    y_pred_simple_class = np.clip(y_pred_simple_class, 0, len(le.classes_) - 1)
    y_pred_multiple_class = np.clip(y_pred_multiple_class, 0, len(le.classes_) - 1)
    
    # Calcul des métriques pour la régression
    mse_simple = mean_squared_error(y_test_encoded, y_pred_simple)
    r2_simple = r2_score(y_test_encoded, y_pred_simple)
    
    mse_multiple = mean_squared_error(y_test_encoded, y_pred_multiple)
    r2_multiple = r2_score(y_test_encoded, y_pred_multiple)
    
    # Calcul de l'accuracy pour la classification
    accuracy_simple = accuracy_score(y_test_encoded, y_pred_simple_class)
    accuracy_multiple = accuracy_score(y_test_encoded, y_pred_multiple_class)
    
    print("\nRésultats de la Régression Linéaire:")
    print(f"Régression Simple - MSE: {mse_simple:.4f}, R²: {r2_simple:.4f}, Accuracy: {accuracy_simple:.2%}")
    print(f"Régression Multiple - MSE: {mse_multiple:.4f}, R²: {r2_multiple:.4f}, Accuracy: {accuracy_multiple:.2%}")
    
    # Conversion des prédictions numériques en étiquettes d'origine
    y_pred_simple_labels = le.inverse_transform(y_pred_simple_class)
    y_pred_multiple_labels = le.inverse_transform(y_pred_multiple_class)
    
    # Création des matrices de confusion
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.heatmap(confusion_matrix(y_test, y_pred_simple_labels), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matrice - Régression Linéaire Simple')
    axes[0].set_xlabel('Prédictions')
    axes[0].set_ylabel('Réelles')
    
    sns.heatmap(confusion_matrix(y_test, y_pred_multiple_labels), annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Matrice - Régression Linéaire Multiple')
    axes[1].set_xlabel('Prédictions')
    axes[1].set_ylabel('Réelles')
    
    plt.tight_layout()
    plt.show()
    
    # Enregistrement de la figure
    regression_matrix_path = "figures/matrices_regression.png"
    fig.savefig(regression_matrix_path)
    print(f"✅ Matrices de confusion pour régression enregistrées sous '{regression_matrix_path}'")
    
    return {
        'Régression Linéaire Simple': accuracy_simple,
        'Régression Linéaire Multiple': accuracy_multiple
    }

def evaluer_modeles(modeles, X_test, y_test, label_encoder):
    """
    Évalue les performances de plusieurs modèles de classification sur des données de test.
    """
    resultats = {}  # Dictionnaire pour stocker les scores d'accuracy de chaque modèle
    accuracy_scores = {}  # Dictionnaire pour stocker les scores pour la visualisation

    # Création du dossier "figures" s'il n'existe pas pour stocker les images générées
    os.makedirs("figures", exist_ok=True)

    # Création d'une figure contenant plusieurs sous-graphiques pour afficher les matrices de confusion
    fig, axes = plt.subplots(1, len(modeles), figsize=(20, 5))  

    # Boucle sur chaque modèle pour évaluer ses performances
    for i, (nom, modele) in enumerate(modeles.items()):
        # Prédiction des étiquettes sur les données de test
        if nom == 'XGBoost':
            y_pred_encoded = modele.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
        else:
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
    plt.show()

    # Enregistrement de la figure contenant les matrices de confusion
    confusion_matrix_path = "figures/matrices_confusion.png"
    fig.savefig(confusion_matrix_path)
    print(f"✅ Matrices de confusion enregistrées sous '{confusion_matrix_path}'")

    return resultats  # Retourne les scores des modèles

def comparer_tous_les_modeles(resultats_classification, resultats_regression):
    """Compare tous les modèles de classification et de régression."""
    # Fusion des résultats
    tous_resultats = {**resultats_classification, **resultats_regression}
    
    # Génération du graphique de comparaison des scores des modèles
    plt.figure(figsize=(12, 6))
    
    # Définition des couleurs pour différencier les types de modèles
    colors = []
    models = list(tous_resultats.keys())
    for model in models:
        if model in ['KNN', 'SVM', 'Decision Tree']:
            colors.append('royalblue')  # Méthodes classiques
        elif model == 'XGBoost':
            colors.append('orangered')  # XGBoost
        else:
            colors.append('forestgreen')  # Régressions
    
    # Création du graphique à barres
    bars = plt.bar(models, list(tous_resultats.values()), color=colors)
    
    # Ajout d'une légende
    import matplotlib.patches as mpatches
    classic = mpatches.Patch(color='royalblue', label='Méthodes classiques')
    xgb = mpatches.Patch(color='orangered', label='XGBoost')
    reg = mpatches.Patch(color='forestgreen', label='Régression')
    plt.legend(handles=[classic, xgb, reg])
    
    plt.title('Comparaison des performances de tous les modèles')  # Titre du graphique
    plt.xlabel('Modèles')  # Nom de l'axe des abscisses
    plt.ylabel('Précision (Accuracy)')  # Nom de l'axe des ordonnées
    plt.ylim(0, 1)  # Définition de l'échelle de l'axe des ordonnées entre 0 et 1
    plt.xticks(rotation=45)  # Rotation des étiquettes sur l'axe des x pour une meilleure lisibilité
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', rotation=0)
    
    # Ajout d'une ligne horizontale à la valeur maximale pour référence
    max_accuracy = max(tous_resultats.values())
    plt.axhline(y=max_accuracy, color='red', linestyle='--', alpha=0.7)
    plt.text(0, max_accuracy + 0.02, f'Max: {max_accuracy:.2%}', color='red')
    
    plt.tight_layout()
    
    # Affichage du graphique de comparaison
    plt.show()

    # Enregistrement du graphique de comparaison des performances
    performance_graph_path = "figures/comparaison_tous_modeles.png"
    plt.savefig(performance_graph_path)
    print(f"✅ Comparaison de tous les modèles enregistrée sous '{performance_graph_path}'")

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
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement des modèles de classification
        modeles, label_encoder = entrainer_modeles(X_train_scaled, y_train)
        resultats_classification = evaluer_modeles(modeles, X_test_scaled, y_test, label_encoder)
        
        # Évaluation des modèles de régression linéaire
        resultats_regression = evaluer_regression_lineaire(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Comparaison de tous les modèles
        comparer_tous_les_modeles(resultats_classification, resultats_regression)
        
        print("\nPerformances des modèles de classification:")
        for nom, score in resultats_classification.items():
            print(f"{nom}: {score:.2%}")  # Affichage des scores en pourcentage
        
        print("\nPerformances des modèles de régression:")
        for nom, score in resultats_regression.items():
            print(f"{nom}: {score:.2%}")  # Affichage des scores en pourcentage
        
        # Détermination du meilleur modèle
        tous_resultats = {**resultats_classification, **resultats_regression}
        meilleur_modele = max(tous_resultats, key=tous_resultats.get)
        meilleur_score = tous_resultats[meilleur_modele]
        
        print(f"\n🏆 Le meilleur modèle est '{meilleur_modele}' avec une précision de {meilleur_score:.2%}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")

# Exécution du script si lancé directement
if __name__ == "__main__":
    main("bigdata_user_dataset_1200.csv")