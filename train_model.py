import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.sklearn import save_model  # <-- new

# Load and prepare data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df[["Pclass", "Age", "SibSp", "Fare", "Survived"]].dropna()
X = df[["Pclass", "Age", "SibSp", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model locally (not using MLflow run ID)
save_model(model, "model")
print("Model saved to ./model")
