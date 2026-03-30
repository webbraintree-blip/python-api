# Label Code Scanner

This app scans visible 2D label codes from uploaded jewelry images and shows the extracted values with an annotated preview.

## Local Run

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Render Deploy

1. Push this folder to GitHub.
2. Create a new Render Web Service from the repo.
3. Render will pick up `render.yaml`, or use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`

## Notes

- The web app accepts uploaded images instead of relying on a fixed local file path.
- Detection logic is shared with `qr_detect.py`.
