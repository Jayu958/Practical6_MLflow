
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("wine-classification-baseline")

# 2. Load Wine dataset
wine = load_wine(as_frame=True)
df = wine.frame

X = df.drop(columns=["target"])
y = df["target"]

print("Dataset shape:", df.shape)
print("Features:", X.shape[1])
print("Classes:", wine.target_names)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Define three models
models = {
    "logistic_regression": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000)
        ),
        {"model_type": "logistic_regression", "C": 1.0,
         "max_iter": 2000}
    ),

    "random_forest_shallow": (
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        {"model_type": "random_forest_shallow",
         "n_estimators": 50, "max_depth": 3}
    ),

    "random_forest_deep": (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        {"model_type": "random_forest_deep",
         "n_estimators": 200, "max_depth": -1}
    )
}

# 5. Train and track experiments
for run_name, (model, params) in models.items():

    with mlflow.start_run(run_name=run_name):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro"
        )
        recall = recall_score(
            y_test, predictions, average="macro"
        )
        f1 = f1_score(
            y_test, predictions, average="macro"
        )

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # Save confusion matrix
        cm = confusion_matrix(y_test, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=wine.target_names
        )

        display.plot()
        plt.title(run_name)
        plt.tight_layout()

        os.makedirs("artifacts", exist_ok=True)
        image_path = f"artifacts/confusion_matrix_{run_name}.png"
        plt.savefig(image_path)
        plt.close()

        mlflow.log_artifact(image_path)

        # Log trained model
        mlflow.sklearn.log_model(
           model,
           name="model",
           skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print(f"\nModel: {run_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

print("\nAll three MLflow experiments completed!")
