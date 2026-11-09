# Bye-Bye Lanternfly

A small YOLOv8 + Flask project split into two parts:

- `notebooks/`: Google Colab notebooks for training and evaluating the detector.
- `app.py` and `templates/index.html`: the local web interface.


## Run the website locally (Windows PowerShell)

Use Python 3.11 or 3.12 for a straightforward Ultralytics environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.
