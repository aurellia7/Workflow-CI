import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

dagshub.init(
    repo_owner='aurellia7',
    repo_name='Eksperimen_SML_Aurellia-Dzakiruna',
    mlflow=True
)

df = pd.read_csv('ulasan_tiktok_preprocessing.csv')
df = df.dropna()
df = df[df['teks'].str.strip() != '']

X = df['teks']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'NaiveBayes': MultinomialNB(),
    'LinearSVC': LinearSVC(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

mlflow.sklearn.autolog()

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}")

print("Training selesai!")