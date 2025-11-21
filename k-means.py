import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import warnings

filename = ("data/final_nba_dataset.csv")
df = pd.read_csv(filename)
print(df.columns.tolist())
print(df.head())
df.info()
mean_salary = df["Salary"].mean()
median_salary = df["Salary"].median()
std_salary = df["Salary"].std()
print(f"Mean: {mean_salary:.2f} - Median: {median_salary:.2f} - Standar Deviaton: {std_salary:.2f}")
df["Salary_log"] = np.log(df["Salary"] + 1)
mean_salary_log = df["Salary_log"].mean()
median_salary_log = df["Salary_log"].median()
print(f"Mean: {mean_salary_log:.2f} - Median: {median_salary_log:.2f}")
ax = df["Salary_log"].plot.hist(figsize=(5,5), bins=50)
ax.set_xlabel("Salary log")
plt.show()
df.info()

# K - MEANS CLUSTERING -> PLAYER ARCHETYPE
# Feature selection
clustering_features = [
    "Games Played",
    "Minutes Played",
    "Age",
    "PTS",
    'FG%',
    '3P%',
    'FT%',
    'Offensive Production per Minute',
    'True Shooting',
    '3PAr',
    'TRB%',
    'Field Goals Assisted',
    'Possessions end by steal',
    'Two-point attempts blocked',
    'Turnovers per 100 plays',
    'Plays Used',
    'Win Shares',
    'Box Plus/Minus',
    'Value Over Replacement',
    'Win_Pct'
]
#Player, team

X_cluster = df[clustering_features].copy()

#-- Standarization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

#Find optimal K
wcss = []
k_values = range(3, 15)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

#Generate Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()

# --- 5. Fit the Final K-Means Model ---
OPTIMAL_K = 8
kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)

# Assign the cluster labels back to the main DataFrame
df['Archetype_ID'] = kmeans_final.labels_



# --- 6. Cluster Profiling (Storytelling) ---
# We use the original stats for profiling, mapping to the new names
profile_cols = [
    "Minutes Played",
    'Offensive Production per Minute',
    'Field Goals Assisted', # AST%
    'TRB%',
    "3PAr",
    "3P%",
    'True Shooting',
    'Two-point attempts blocked',
    "Possessions end by steal",
    'Plays Used', # USG%
    'Win Shares',
    "Value Over Replacement"
]

cluster_profile = (df.groupby('Archetype_ID')[profile_cols].mean().sort_values
                   (by='Value Over Replacement', ascending=False))
print("\n--- Cluster Profile (Averaged Stats) ---")
print(cluster_profile.to_markdown(floatfmt=".3f"))

# --- 7. Final Naming and Feature Creation ---
# Define the names based on the calculated profiles (derived from the profile table)
archetype_names = {
    0: "Elite Creator / Franchise Star",
    1: "Low Efficiency Creator",
    2: "Low Impact / End of Bench",
    3: "High Volume Inefficient Scorer",
    4: "Defensive Anchor / Elite Rebounder",
    5: "3&D Specialist",
    6: "Versatile Player",
    7: "High - Impact Starter / Primary Option"
}

df['Player_Archetype'] = df['Archetype_ID'].map(archetype_names)
df.info()
# Save the updated DataFrame with the new Archetype feature
df.to_csv('nba_data_with_archetypes.csv', index=False)
print(f"\nFinal dataset saved to 'nba_data_with_archetypes.csv' with Archetype feature.")




