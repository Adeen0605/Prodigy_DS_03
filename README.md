Simple Decision Tree Flask app

1. Create virtualenv and install deps:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train model (downloads dataset):

```powershell
python model_train.py
```

3. Run app:

```powershell
python app.py
```

Open http://127.0.0.1:5000
