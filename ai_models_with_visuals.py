
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from datetime import datetime

# Load dataset
df = pd.read_csv('bigdata_user_dataset_1200.csv')

# ---------------------------
# Preprocessing
# ---------------------------
df['Nbr_acquired_skills'] = df['Compétences_acquises'].apply(lambda x: len(str(x).split(',')))
df['Nbr_target_skills'] = df['Compétences_cibles'].apply(lambda x: len(str(x).split(',')))
df['Skill_level'] = df['Niveau_de_maîtrise'].map({'Débutant': 0, 'Intermédiaire': 1, 'Avancé': 2})

# ---------------------------
# 1. Mentor-Learner Matching
# ---------------------------
features = df[['Nbr_acquired_skills', 'Nbr_target_skills', 'Skill_level']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,5))
sns.scatterplot(x='Nbr_acquired_skills', y='Nbr_target_skills', hue='Cluster', data=df, palette='Set2')
plt.title('Mentor-Learner Clustering')
plt.savefig('mentor_matching.png')
plt.close()

# ---------------------------
# 2. Learning Time Estimation
# ---------------------------
def parse_duration(text):
    try:
        return np.mean([int(s.strip().split()[0]) for s in str(text).split(',')])
    except:
        return 0

df['Avg_learning_duration'] = df['Durée_apprentissage_par_compétence'].apply(parse_duration)

X = df[['Nbr_acquired_skills', 'Skill_level']]
y = df['Avg_learning_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Duration")
plt.ylabel("Predicted Duration")
plt.title("Learning Time Prediction")
plt.savefig('learning_time_estimation.png')
plt.close()

# ---------------------------
# 3. Engagement Optimization
# ---------------------------
df['Dernière_activité'] = pd.to_datetime(df['Dernière_activité'], format="%d/%m/%Y")
reference_date = datetime(2025, 3, 1)
df['Days_since_active'] = (reference_date - df['Dernière_activité']).dt.days
df['Active'] = (df['Days_since_active'] < 30).astype(int)

X = df[['Nbr_acquired_skills', 'Days_since_active', 'Skill_level']]
y = df['Active']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Engagement Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('engagement_optimization.png')
plt.close()
