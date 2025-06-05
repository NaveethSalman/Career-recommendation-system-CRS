import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_structured_model(data_path):
    df = pd.read_csv('data/Data_final.csv')
    X = pd.get_dummies(df.drop('Career', axis=1))
    y = df['Career']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, 'models/structured_model.pkl')

