import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from .preprocess import clean_text

def train_resume_model(data_path):
    df = pd.read_csv('data/UpdatedResumeDataSet.csv')
    df['cleaned'] = df['Resume'].apply(clean_text)

    vec = TfidfVectorizer(max_features=1000)
    X = vec.fit_transform(df['cleaned'])
    y = df['Category']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, 'models/resume_classifier.pkl')
    joblib.dump(vec, 'models/vectorizer.pkl')

