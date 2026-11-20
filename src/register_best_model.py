
import mlflow
from mlflow.tracking import MlflowClient

# MLflow configuration
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_ID = "1"

# Best model: Random Forest Shallow
BEST_RUN_ID = "f52d068e438b4072a5c79b97e9ac6322"

MODEL_NAME = "wine-classifier-prod"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# Find the logged model belonging to the selected run
logged_models = client.search_logged_models(
    experiment_ids=[EXPERIMENT_ID],
    max_results=100
)

matching_models = [
    model for model in logged_models
    if model.source_run_id == BEST_RUN_ID
    and model.name == "model"
]

if not matching_models:
    raise Exception(
        "No logged model found for the selected run. "
        "Check the MLflow Logged Models section."
    )

best_model = matching_models[0]
model_uri = f"models:/{best_model.model_id}"

# Register model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

# Assign Staging alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="staging",
    version=registered_model.version
)

print("Model registered successfully!")
print("Model name:", MODEL_NAME)
print("Model version:", registered_model.version)
print("Model alias: staging")
print("Model URI:", model_uri)
