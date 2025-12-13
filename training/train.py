import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# === Load data ===
df = pd.read_csv("Dataset/secondary_data.csv", sep=";")

X = df.drop("class", axis=1)
y = df["class"]

# === Column selection ===
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

# === Preprocessing pipelines ===
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# === Model ===
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", model)
])

# === Training ===
pipeline.fit(X, y)


# === Save model ===
joblib.dump(pipeline, "model.pkl")
print("Model saved as model.pkl")
