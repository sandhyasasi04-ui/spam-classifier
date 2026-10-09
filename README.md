# Spam Shield - Email Spam Classifier & Threat Detection Dashboard

## Run on your computer (Windows, VS Code terminal)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements-local.txt
python train.py
uvicorn api.index:app --reload
```
Open http://127.0.0.1:8000

## Use the real dataset (recommended)
`data/spam.csv` is a small sample so the project runs immediately.
Download "SMS Spam Collection Dataset" from Kaggle, put its `spam.csv` in `data/`
(replace the sample), then run `python train.py` again. Expect about 98% accuracy.

## Push to GitHub
```
git init
git add .
git commit -m "Spam Shield"
git branch -M main
git remote add origin https://github.com/YOURNAME/spam-classifier.git
git push -u origin main
```

## Deploy on Render
New + > Web Service > pick this repo, then set:
- Build Command: `pip install -r requirements.txt && python train.py`
- Start Command: `uvicorn api.index:app --host 0.0.0.0 --port $PORT`
- Instance type: Free
