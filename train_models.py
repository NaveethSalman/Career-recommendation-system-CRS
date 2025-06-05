from src.model_text import train_resume_model
from src.model_structured import train_structured_model

train_resume_model('data/resume_dataset.csv')
train_structured_model('data/career_prediction_dataset.csv')

