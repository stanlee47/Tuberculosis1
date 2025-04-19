import pandas as pd
import sys
import yaml
import os
import pickle as pk

print("Preprocess.py")

# Load parameters from params.yaml
params= yaml.safe_load(open("params.yaml"))['preprocess']

#Preprocess the data
def preprocess_data(input_file, output_file):
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df = df.dropna()
        df = df.drop_duplicates()
        df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    else:
        print(f"Input File Does't Exist: {input_file}")
    

if __name__ == '__main__':
    preprocess_data(params['input'], params['output'])