

import pandas as pd
import os
import yaml
import pickle as pk
import mlflow
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV

from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


def hyperparmeter_tuning(x_train,y_train):
    best_model=None
    best_score=0
    models={
        'RandomForestClassifier':{'model':RandomForestClassifier(),'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }},
        'SVC':{'model':SVC(),'params': {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf']
            }},
        'DecisionTreeClassifier':{'model':DecisionTreeClassifier(),'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }}
        }
    
    mlflow.set_tracking_uri("https://dagshub.com/stanlykurian22/Tuberculosis.mlflow")
    mlflow.set_experiment('Tuberculosis2')
    for model_name,model in models.items():
        print(f"Now Training : {model_name}")
        grid_search=GridSearchCV(estimator=model['model'],param_grid=model['params'],cv=3,verbose=2,n_jobs=-1)
        grid_search.fit(x_train,y_train)
        signature=infer_signature(x_train,grid_search.predict(x_train))
        model_uri = f"models:{model_name}/1"
        mlflow.sklearn.log_model(grid_search.best_estimator_, model_name, signature=signature)
        accuracy=grid_search.score(x_train,y_train)
        mlflow.log_metrics({"accuracy": accuracy})
        mlflow.log_artifact('params.yaml')
        
        if accuracy>best_score:
            best_model=model['model']
            best_score=accuracy
    return best_model,best_score
path=yaml.safe_load(open('params.yaml'))['train']
def train_model():
    data=pd.read_csv(path['data'])
    target='DEATH_EVENT'
    X=data.drop(columns=[target])
    y=data[target]
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
    best_model,best_score=hyperparmeter_tuning(x_train,y_train)
    print(f"Best Model: {best_model} with score: {best_score}")
    
    #Save the model in System
    model_path=path['model_path']
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    filename=path['model_path']
    pk.dump(best_model,open(filename,'wb'))
    print(f"Model saved to {filename}")


if __name__ == '__main__':
    train_model()
    print("Training Completed")
    
    
    
