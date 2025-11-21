import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

filename = ("data/nba_data_with_archetypes.csv")
df = pd.read_csv(filename)
df.info()
print(df.head())

#Define variables Y / X
Y = df["Salary_log"]


#Select the best features for variables X
numeric_features = [
    "Minutes Played",
    "PTS",
    "Age",
    "Games Played",
    "True Shooting",
    "Offensive Production per Minute",
    "3PAr",
    "Field Goals Assisted",
    "Plays Used",
    "Win_Pct",
    "Win Shares",
    "Value Over Replacement"

]

categorical_features = ["Player_Archetype"]

X = df[numeric_features + categorical_features]

#2. Train test split
X_train, X_test, Y_train_, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Train test split done correctly.")

#3. Build de preprocessing pipelines
#Numeric pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
#Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

#General preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)

    ],
    remainder="passthrough"
)

print("Preprocessing pipelines defined.")

#4. Training the random fores model
#Join Preprocessor -> Random forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
])


#Hyperparametres and grid
# Define the range of parameters to test
param_dist = {

    'regressor__n_estimators': randint(low=100, high=500),


    'regressor__max_depth': [10, 15, 20, 30, None],


    'regressor__min_samples_leaf': randint(1, 10),


    'regressor__max_features': [0.6, 0.8, 1.0]
}
# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)

print("Starting Randomized Search (50 iterations * 5 folds)...")

random_search.fit(X_train, Y_train_)
print("Model trained")


# 2. Print the parameters that led to the best score
print("\n--- Optimal Parameters Found ---")
print(random_search.best_params_)



#5. Model evaluation and feature importance
#Evaluate performance
# 1. Extract the best model found during the search
best_rf_model = random_search.best_estimator_

#Make the prediction
Y_pred_log = best_rf_model.predict(X_test)

#Convert the data
Y_pred_dollars = np.exp(Y_pred_log) - 1
Y_test_dollars = np.exp(Y_test) - 1

#Calculare key metrics
r2 = r2_score(Y_test_dollars, Y_pred_dollars)
mae = mean_absolute_error(Y_test_dollars, Y_pred_dollars)
print("\n--- Model Performance on Test Set ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

# --- 6. Extract Feature Importance ---

# 1. Access the trained preprocessor
preprocessor_fitted = best_rf_model.named_steps['preprocessor']

# 2. Access the fitted categorical pipeline (which is a Pipeline object)
cat_pipeline_fitted = preprocessor_fitted.named_transformers_['cat']

# 3. Access the OneHotEncoder by name within the categorical pipeline's steps
# This uses the stable .named_steps attribute
encoder = cat_pipeline_fitted.named_steps['onehot']

# 4. Extract the final feature names using the correctly retrieved encoder object
cat_feature_names = encoder.get_feature_names_out(categorical_features)


# 4. Extract the final feature names using the correctly retrieved encoder object
cat_feature_names = encoder.get_feature_names_out(categorical_features)

all_feature_names = list(numeric_features) + list(cat_feature_names)

# Extract the importances from the trained regressor
importances = best_rf_model.named_steps['regressor'].feature_importances_

# Combine names and scores into a readable DataFrame
feature_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("\n--- Top 25 Feature Importances ---")
print("This shows the model's primary drivers for salary prediction.")
print(feature_importances.head(25).to_markdown(floatfmt=".4f"))




# --- 2. Prepare Feature Matrix for Prediction ---
# This matrix (X) MUST contain the exact same columns used during training.
# We will select a simplified set for demonstration, but you must use the full set.
X_predict = df[[
    "Minutes Played",
    "PTS",
    "Age",
    "Games Played",
    "True Shooting",
    "Offensive Production per Minute",
    "3PAr",
    "Field Goals Assisted",
    "Plays Used",
    "Win_Pct",
    "Win Shares",
    "Value Over Replacement",
    "Player_Archetype",

]]

# --- 3. Get Predictions and Calculate Value Gap ---

# ACTION: Use your trained pipeline to predict salaries for every player
Y_predict_dollars = best_rf_model.predict(X_predict)
#Convert de column to dollars
df["Predicted_Salary"] = np.exp(Y_predict_dollars) - 1


# Calculate the Value Gap: The core business metric
df['Value_Gap'] = df['Salary'] - df['Predicted_Salary']

df.to_csv("data/dataset_visualizations.csv")

# --- 4. Analysis 1: Identify Bargains (The Agency Target List) ---

# Find the 20 most underpaid players (the players with the largest NEGATIVE Value_Gap)
bargain_targets = df.sort_values(by='Value_Gap', ascending=True).head(20)

print("\n--- 1. TOP 20 UNDERVALUED TARGETS (The Agency's List) ---")
print(bargain_targets[['Player',
                       'Player_Archetype',
                       'Salary',
                       'Predicted_Salary',
                       'Value_Gap']].to_markdown(floatfmt=(None, "s", "s", ",.0f", ",.0f", ",.0f")))

# Find the 20 most overpaid players (the players with the largest POSITIVE Value_Gap)
overpaid_targets = df.sort_values(by='Value_Gap', ascending=False).head(20)

print("\n--- 1. TOP 20 OVERPAID TARGETS (The Agency's List) ---")
print(overpaid_targets[['Player',
                       'Player_Archetype',
                       'Salary',
                       'Predicted_Salary',
                       'Value_Gap']].to_markdown(floatfmt=(None, "s", "s", ",.0f", ",.0f", ",.0f")))

# --- 5. Analysis 2: Strategic Insights by Archetype ---

# Group the data by the Archetype and calculate the average Value Gap
strategic_insight = df.groupby('Player_Archetype')['Value_Gap'].agg(['mean', 'count', 'median']).sort_values(by='mean'
                                                                                                             , ascending=True)

# Rename columns for clarity in the final report
strategic_insight = strategic_insight.rename(columns={'mean': 'Avg_Value_Gap', 'count': 'Player_Count', 'median': 'Median_Value_Gap'})

print("\n--- 2. STRATEGIC INSIGHTS BY ARCHETYPE (Market Mispricing) ---")
print(strategic_insight.to_markdown(floatfmt=',.0f'))

