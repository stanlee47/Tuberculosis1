import mlflow
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import os
import yaml

from dotenv import load_dotenv


load_dotenv()  # Loads variables from .env file

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


params=yamal.safe_load(open('params.yaml'))['train']

def evaluate_model(data_path,model_path):
    data=pd.read_csv(data_path)
    target='DEATH_EVENT'
    X=data.drop(columns=[target])
    y=data[target]
    
    # Load the model
    model=mlflow.sklearn.load_model(model_path)
    
    # Make predictions
    y_pred=model.predict(X)
    
    # Calculate metrics
    accuracy=metrics.accuracy_score(y,y_pred)
    precision=metrics.precision_score(y,y_pred)
    recall=metrics.recall_score(y,y_pred)
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })
    print(f"Evaluation is finished")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
if __name__ == "__main__":
    print("Starting evaluation")
    data_path=params['data']
    model_path=params['model_path']
    evaluate_model(data_path,model_path)

    