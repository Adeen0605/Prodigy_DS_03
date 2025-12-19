"""
Minimal Flask app for the simple Decision Tree project.
Features:
- Load trained model
- Show sample table
- Display accuracy/confusion image
- Predict from simple input form
- Upload CSV to retrain (simple)
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
from model_train import train_from_df, download_dataset

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'dt_model.pkl')
SAMPLE_CSV = os.path.join(BASE, 'data_sample.csv')
IMG_DIR = os.path.join(BASE, 'static', 'images')

app = Flask(__name__)
app.secret_key = 'dev'

# helper to load model bundle
def load_model_bundle():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@app.route('/', methods=['GET','POST'])
def index():
    bundle = load_model_bundle()
    # sample table
    sample = None
    if os.path.exists(SAMPLE_CSV):
        sample = pd.read_csv(SAMPLE_CSV).to_dict(orient='records')
    # class counts (simple from sample)
    class_counts = {'yes':0, 'no':0}
    if os.path.exists(SAMPLE_CSV):
        df = pd.read_csv(SAMPLE_CSV)
        if 'y' in df.columns:
            vc = df['y'].value_counts().to_dict()
            class_counts['yes'] = int(vc.get('yes',0))
            class_counts['no'] = int(vc.get('no',0))

    prediction = None
    probability = None
    accuracy = None
    cm_path = None
    if bundle is not None:
        # if trained, try to show last confusion image and a quick accuracy if saved
        cm_path = os.path.join('static','images','confusion_matrix.png')

    if request.method == 'POST':
        # For prediction: read form and build minimal feature row consistent with training
        bundle = load_model_bundle()
        if bundle is None:
            flash('Model not found. Train model first or upload dataset.', 'danger')
            return redirect(url_for('index'))
        # build zero row
        import numpy as np
        cols = bundle['feature_columns']
        row = {c:0 for c in cols}
        # fill the numeric fields if present
        try:
            age = int(request.form.get('age','30'))
            balance = float(request.form.get('balance','0'))
            duration = int(request.form.get('duration','100'))
            campaign = int(request.form.get('campaign','1'))
        except Exception:
            flash('Invalid numeric input', 'danger')
            return redirect(url_for('index'))
        # assign to matching columns if present
        if 'age' in row: row['age'] = age
        if 'balance' in row: row['balance'] = balance
        if 'duration' in row: row['duration'] = duration
        if 'campaign' in row: row['campaign'] = campaign
        # create DataFrame
        X = pd.DataFrame([row], columns=cols).fillna(0)
        model = bundle['model']
        try:
            probs = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            pred_label = bundle['label_encoder'].inverse_transform([pred])[0]
            prediction = pred_label
            probability = float(max(probs))
        except Exception as e:
            flash(f'Prediction error: {e}', 'danger')

    return render_template('index.html', sample_table=sample, class_counts=class_counts, prediction=prediction, probability=probability, accuracy=accuracy, cm_path=cm_path)

@app.route('/upload', methods=['POST'])
def upload():
    # Accept CSV upload and retrain
    if 'file' not in request.files:
        flash('No file uploaded', 'warning')
        return redirect(url_for('index'))
    f = request.files['file']
    if f.filename == '':
        flash('Empty filename', 'warning')
        return redirect(url_for('index'))
    # save uploaded file temporarily
    upload_path = os.path.join(BASE, 'uploaded.csv')
    f.save(upload_path)
    try:
        df = pd.read_csv(upload_path)
        res = train_from_df(df)
        # save sample
        df.head(20).to_csv(SAMPLE_CSV, index=False)
        flash(f"Retrained model. accuracy={res['accuracy']:.4f}", 'success')
    except Exception as e:
        flash(f'Retrain failed: {e}', 'danger')
    return redirect(url_for('index'))

@app.route('/train_local')
def train_local():
    # quick train using the canonical download dataset
    try:
        df = download_dataset()
        res = train_from_df(df)
        df.head(20).to_csv(SAMPLE_CSV, index=False)
        flash(f"Trained from canonical dataset. accuracy={res['accuracy']:.4f}", 'success')
    except Exception as e:
        flash(f'Training failed: {e}', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Disable the auto-reloader to avoid repeated restarts (OneDrive watchers)
    app.run(debug=True, use_reloader=False)
