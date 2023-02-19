import os
import shutil
import mlflow
from unittest.mock import patch, Mock
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Metric, Run, RunStatus
from src.mlflow_utils import get_latest_run_id, get_latest_model

def test_get_latest_run_id():
    experiment_name = "test_experiment"
    metric_key = "metric_key"
    experiment_id = 0
    run_id = "run_id"

    mock_experiment = Experiment(experiment_id, experiment_name, None, None, None, None)
    mock_run = Run(run_id, None, experiment_id, None, RunStatus.FINISHED, None, 1, None, None, None)

    # Test the function with a single run in the experiment
    mock_client = Mock(spec=MlflowClient)
    mock_client.get_experiment_by_name.return_value = mock_experiment
    mock_client.search_runs.return_value = [mock_run]

    result = get_latest_run_id(experiment_name, (metric_key, False), mlflow_client=mock_client)
    assert result == run_id

    # Test the function with no runs in the experiment
    mock_client.search_runs.return_value = []

    result = get_latest_run_id(experiment_name, (metric_key, False), mlflow_client=mock_client)
    assert result == None

    # Test the function with multiple runs in the experiment
    mock_client.search_runs.return_value = [mock_run, mock_run]

    result = get_latest_run_id(experiment_name, (metric_key, False), mlflow_client=mock_client)
    assert result == run_id

def test_get_latest_model(tmp_path):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    experiment_name = "test_experiment"
    metric_key = "metric_key"
    experiment_id = mlflow.create_experiment(experiment_name)
    run_id = mlflow.start_run(experiment_id=experiment_id, run_name="test_run").info.run_id

    model_dir = os.path.join(tmp_path, "model")
    os.makedirs(model_dir)

    # Test the function with a single run in the experiment
    with open(os.path.join(model_dir, "model.pkl"), "w") as f:
        f.write("test")
    mlflow.log_artifacts(model_dir, artifact_path="model")

    result = get_latest_model(model_path=tmp_path)
    assert result == run_id

    # Test the function with no runs in the experiment
    mlflow.delete_run(run_id)
    shutil.rmtree(model_dir)

    result = get_latest_model(model_path=tmp_path)
    assert result == None

    # Test the function with multiple runs in the experiment
    run_id = mlflow.start_run(experiment_id=experiment_id, run_name="test_run_2").info.run_id
    with open(os.path.join(model_dir, "model.pkl"), "w") as f:
        f.write("test")
    mlflow.log_artifacts(model_dir, artifact_path="model")

    result = get_latest_model(model_path=tmp_path)
    assert result == run_id

    mlflow.delete_run(run_id)
    mlflow.delete_experiment(experiment_id)
    shutil.rmtree(model_dir)
