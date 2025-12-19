"""
model_train.py
Simple beginner-friendly training script using a Bank-like marketing dataset.
- Downloads dataset if missing
- Preprocess (simple: drop duplicates, label target, one-hot encode categoricals)
- Train DecisionTreeClassifier
- Save model bundle (joblib) and a sample CSV + confusion matrix image
- Exposes `train_from_df(df, max_depth=6)` for programmatic retrain
"""
import os
import zipfile
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
IMG_DIR = os.path.join(BASE, 'static', 'images')
for d in (DATA_DIR, MODELS_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

ZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
ZIP_PATH = os.path.join(DATA_DIR, 'bank-additional.zip')
CSV_PATH = os.path.join(DATA_DIR, 'bank-additional', 'bank-additional-full.csv')


def download_dataset():
    if not os.path.exists(CSV_PATH):
        print('Downloading dataset...')
        r = requests.get(ZIP_URL, stream=True)
        r.raise_for_status()
        with open(ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_DIR)
    return pd.read_csv(CSV_PATH, sep=';')


def preprocess_simple(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    if 'y' not in df.columns:
        raise ValueError('Dataset must contain target column "y"')
    le = LabelEncoder()
    df['y_label'] = le.fit_transform(df['y'])
    # select a few interpretable features
    features = ['age', 'job', 'marital', 'education', 'balance', 'duration', 'campaign', 'housing', 'loan']
    features = [c for c in features if c in df.columns]
    X = df[features]
    # one-hot encode categoricals simply
    X = pd.get_dummies(X, drop_first=True)
    y = df['y_label']
    return X, y, le


def train_from_df(df, max_depth=6):
    """Train on a DataFrame, save model bundle and confusion matrix image.
    Returns dict with accuracy and image path.
    """
    X, y, le = preprocess_simple(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    cm_file = os.path.join(IMG_DIR, 'confusion_matrix.png')
    fig.savefig(cm_file, bbox_inches='tight')
    plt.close(fig)

    # save model bundle
    bundle = {'model': clf, 'label_encoder': le, 'feature_columns': X.columns.tolist()}
    joblib.dump(bundle, os.path.join(MODELS_DIR, 'dt_model.pkl'))

    return {'accuracy': acc, 'confusion_matrix': os.path.join('static', 'images', 'confusion_matrix.png')}


def train_main():
    df = download_dataset()
    res = train_from_df(df)
    # save sample
    df.head(20).to_csv(os.path.join(BASE, 'data_sample.csv'), index=False)
    print('Trained. accuracy=', res['accuracy'])
    return res


if __name__ == '__main__':
    train_main()
