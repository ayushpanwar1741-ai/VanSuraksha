# VanSuraksha

VanSuraksha is a Flask-based web application for forest fire monitoring and prediction.

It includes:
- Real-time live monitoring with YOLOv5-based fire detection
- Fire alerts dashboard with saved snapshots
- Image prediction page for fire/smoke detection
- Wildfire forecasting page based on MODIS data

## Project Structure

- `App.py`: Flask app factory and server entry point
- `View.py`: routes, live monitoring logic, prediction, wildfire forecast
- `database.py`: SQLAlchemy models
- `Models/yolocff.pt`: YOLOv5 model used by live monitor and image prediction
- `templates/`: HTML templates
- `static/`: CSS, JS, images, generated uploads/shots/videos

## Requirements

- Python 3.10+ (project currently running on Python 3.13 in this workspace)
- pip
- Virtual environment (recommended)
- Webcam (for live monitoring)

Dependencies are listed in `requirements.txt`.

## Setup

1. Clone repository.
2. Create and activate a virtual environment.
3. Install dependencies.

Example (Windows PowerShell):

```powershell
cd Fire_Eye-A_Real-Time_Forest_Fire_and_Smoke_Detection_Website
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Project

From project root:

```powershell
python App.py
```

Open:

`http://127.0.0.1:5000`

## Key Runtime Files

Make sure these exist:

- `Models/yolocff.pt`
- `fire_alarm.ogg`

The app will auto-create folders if missing:

- `static/uploads`
- `static/shots`
- `static/video`
- `instance`

## GitHub Push Steps

After creating your GitHub repo:

```powershell
git init
git add .
git commit -m "Initial commit: VanSuraksha project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Notes

- The `.gitignore` excludes generated media, local DB files, caches, and most large checkpoints.
- It keeps `Models/yolocff.pt` tracked as the active runtime model.
