import mlflow
import os
import shutil
import time

MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_NAME = os.environ.get("MODEL_NAME")
MLFLOW_HOST = os.environ.get("MLFLOW_HOST")
TEST_MODEL_PATH = "tests/model"

cwd = os.getcwd()
FULL_MODEL_PATH = os.path.join(cwd, MODEL_PATH, MODEL_NAME)

def get_latest_run_id(experiment_name, measures):
    """
    Get the ID of the latest MLflow run for a given experiment.

    :param experiment_name: The name of the experiment.
    :param measures: A tuple containing the metric to sort by (e.g. "end_time") and a boolean indicating the sort order.
    :return: The run ID of the latest run for the given experiment.
    """
    sort_values_by = measures[0]
    sort_values_order = measures[1]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    df = mlflow.search_runs(experiment_ids=experiment_id)
    last_run_id = df.sort_values(by=sort_values_by, ascending=sort_values_order).head(1).run_id.values[0]
    return last_run_id

def get_latest_model():
    """
    Download the latest model artifact from MLflow.

    :return: The run ID of the latest run that produced the downloaded model artifact.
    """
    mlflow.set_tracking_uri(MLFLOW_HOST)
    last_run_id = get_latest_run_id(MODEL_NAME, ["end_time", False])
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(last_run_id, "", FULL_MODEL_PATH)
    return last_run_id

if __name__ == "__main__":
    current_run_id = get_latest_model()
    while True:
        time.sleep(10)
        new_run_id = get_latest_run_id(MODEL_NAME, ["end_time", False])
        if current_run_id != new_run_id:
            print("New model is available")
            current_run_id = get_latest_model()
