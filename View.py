from flask import Blueprint, render_template, jsonify, request, Response
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, time
import os
import io
import glob
import yolov5
from threading import Thread
import torch
import time
from pygame import mixer
import base64
from database import db, Fire_Alerts
from App import create_app
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
UPLOADS_DIR = os.path.join(STATIC_DIR, 'uploads')
SHOTS_DIR = os.path.join(STATIC_DIR, 'shots')
VIDEO_DIR = os.path.join(STATIC_DIR, 'video')
MODEL_PATH = os.path.join(MODELS_DIR, 'yolocff.pt')
ALARM_SOUND_PATH = os.path.join(BASE_DIR, 'fire_alarm.ogg')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SHOTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

model1 = yolov5.load(MODEL_PATH)
classes = model1.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

global rec_frame, switch, neg, rec, out
neg = 0
switch = 0
rec = 0
rec_frame = None
camera = None
detection_streak = 0
last_alert_time = None
alarm_channel = None
alarm_sound = None
alarm_stop_at = None

ALERT_FIRE_THRESHOLD = 0.35
ALERT_CONTROLLED_FIRE_THRESHOLD = 0.30
ALERT_STREAK_THRESHOLD = 1
ALERT_COOLDOWN_SECONDS = 8
ALARM_DURATION_SECONDS = 5
SMALL_FLAME_AREA_RATIO = 0.015
SMALL_FLAME_FIRE_THRESHOLD = 0.18
SMALL_FLAME_CONTROLLED_THRESHOLD = 0.22

alarm_ready = False

def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def score_frame(frame):
    model1.to(device)
    frame = [frame]
    results = model1(frame)
    print(results)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    return classes[int(x)]


def plot_boxes(results, frame):
    global detection_streak, last_alert_time, alarm_ready, alarm_channel, alarm_sound, alarm_stop_at

    now = datetime.now()
    if alarm_channel and alarm_stop_at and now >= alarm_stop_at:
        alarm_channel.stop()
        alarm_channel = None
        alarm_stop_at = None

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    fire_candidates = []

    for i in range(n):
        row = cord[i]
        conf = float(row[4])
        label_name = class_to_label(labels[i]).lower()
        x1, y1 = int(row[0] * x_shape), int(row[1] * y_shape)
        x2, y2 = int(row[2] * x_shape), int(row[3] * y_shape)
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, x_shape * y_shape))
        is_small_flame = box_area_ratio <= SMALL_FLAME_AREA_RATIO

        fire_threshold = SMALL_FLAME_FIRE_THRESHOLD if is_small_flame else ALERT_FIRE_THRESHOLD
        controlled_threshold = SMALL_FLAME_CONTROLLED_THRESHOLD if is_small_flame else ALERT_CONTROLLED_FIRE_THRESHOLD

        if label_name == 'fire' and conf >= fire_threshold:
            fire_candidates.append((conf, x1, y1, x2, y2, label_name))
            box_color = (0, 0, 255)
        elif label_name == 'controlled fire' and conf >= controlled_threshold:
            fire_candidates.append((conf, x1, y1, x2, y2, label_name))
            box_color = (0, 165, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame,
            f"{label_name} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
        )

    if not fire_candidates:
        detection_streak = 0
        return frame

    detection_streak += 1
    if detection_streak < ALERT_STREAK_THRESHOLD:
        return frame

    if last_alert_time and (now - last_alert_time).total_seconds() < ALERT_COOLDOWN_SECONDS:
        return frame

    if not alarm_ready:
        mixer.init()
        alarm_sound = mixer.Sound(ALARM_SOUND_PATH)
        alarm_ready = True

    if alarm_sound is not None:
        alarm_channel = alarm_sound.play(loops=-1)
        if alarm_channel is not None:
            alarm_stop_at = now + timedelta(seconds=ALARM_DURATION_SECONDS)

    print("fire is detected")

    p = os.path.join(SHOTS_DIR, 'shot_{}.png'.format(str(now).replace(":", '')))
    cv2.imwrite(p, frame)

    with open(p, 'rb') as file:
        image_data = file.read()

    encoded_image = base64.b64encode(image_data).decode('utf-8')

    new_alert = Fire_Alerts(date=str(now.date()), time=str(now.time()), image_path=encoded_image)
    db.session.add(new_alert)
    db.session.commit()
    last_alert_time = now
    detection_streak = 0
    return frame


def gen_frames(app):  # generate frame by frame from camera
 with app.app_context():
    global rec_frame, camera
    while True:
      if camera is not None:
        success, frame = camera.read()
        if success:
            if switch:
                results = score_frame(frame)
                frame = plot_boxes(results, frame)
            if neg:
                frame = cv2.bitwise_not(frame)
            if rec:
                global rec_frame
                rec_frame = frame
                frame = cv2.putText(frame, "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass
      else:
          # Camera not started — yield a placeholder frame so the browser doesn't hang
          time.sleep(0.2)
          placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
          cv2.putText(placeholder, "Camera Off  |  Click Monitor to Start",
                      (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
          cv2.putText(placeholder, "VanSuraksha Live Monitor",
                      (180, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
          _, buf = cv2.imencode('.jpg', placeholder)
          yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def predict_label(img_path):
    """Run YOLOv5 detection on an uploaded image.
    Returns (label, confidence_pct, annotated_img_path)."""
    frame = cv2.imread(img_path)
    if frame is None:
        return "Error reading image", 0.0, img_path

    model1.to(device)
    results = model1([frame])
    labels_t, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    best_label = "No Fire"
    best_conf  = 0.0
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(len(labels_t)):
        row  = cord[i]
        conf = float(row[4])
        if conf < 0.25:
            continue
        lname = class_to_label(labels_t[i])
        if conf > best_conf:
            best_conf  = conf
            best_label = lname
        x1, y1 = int(row[0] * x_shape), int(row[1] * y_shape)
        x2, y2 = int(row[2] * x_shape), int(row[3] * y_shape)
        color  = (0, 0, 255) if 'fire' in lname.lower() else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{lname} {conf:.2f}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    base, ext = os.path.splitext(img_path)
    result_path = base + '_result' + ext
    cv2.imwrite(result_path, frame)

    return best_label, round(best_conf * 100, 1), result_path


View = Blueprint(__name__, "View")


@View.route('/')
def Home():
    return render_template('index.html')


@View.route('/About')
def About():
    return render_template('About.html')


@View.route('/FireAlerts')
def FireAlerts():
    app = create_app()  # Create the Flask app object
    with app.app_context():
      alerts = Fire_Alerts.query.all()
    return render_template('FireAlerts.html', alerts=alerts)

@View.route('/data')
def get_data():
    # Calculate the date range for the last 10 days
    today = datetime.now().date()
    ten_days_ago = today - timedelta(days=9)
    
    # Retrieve the data for the last 10 days from the database
    fire_alerts = Fire_Alerts.query.filter(Fire_Alerts.date >= ten_days_ago).all()

    # Calculate the count of fire alerts per day
    data = []
    for i in range(10):
        date_i = today - timedelta(days=i)
        count = sum(datetime.strptime(alert.date, '%Y-%m-%d').date() == date_i for alert in fire_alerts)
        data.append({'date': date_i.strftime('%Y-%m-%d'), 'count': count})

    return jsonify(data)


@View.route('/ModelTesting')
def ModelTesting(): 
    history=np.load('my_history.npy',allow_pickle='TRUE').item()
    # set the font globally
    params = {'font.family':'Comic Sans MS',
              "xtick.color" : "white",
              "ytick.color" : "white"}
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(8,4)) 
    fig.patch.set_facecolor('xkcd:mahogany')

    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy', fontsize=16, color='white')
    plt.ylabel('Accuracy', fontsize=14, color='white')
    plt.xlabel('Epochs', fontsize=14, color='white')
    plt.legend()
    plt.savefig('static/graphs/Acc_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    fig2 = plt.figure(figsize=(8,4)) 
    fig2.patch.set_facecolor('xkcd:mahogany')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss', fontsize=16, color='white')
    plt.ylabel('Loss', fontsize=14, color='white')
    plt.xlabel('Epochs', fontsize=14, color='white')
    plt.legend()
    plt.savefig('static/graphs/Loss_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return render_template('ModelTesting.html', acc_plot_url="static/graphs/Acc_plot.png", loss_plot_url="static/graphs/Loss_plot.png")


@View.route('/delete_alert/<int:alert_id>', methods=['POST'])
def delete_alert(alert_id):
    alert = Fire_Alerts.query.get_or_404(alert_id)
    try:
        db.session.delete(alert)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@View.route("/Prediction", methods=['GET', 'POST'])
def Prediction():
    return render_template("Prediction.html")


@View.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        filename = secure_filename(img.filename)
        img_path = os.path.join(UPLOADS_DIR, filename)
        img.save(img_path)
        label, confidence, result_path = predict_label(img_path)
        return render_template("Prediction.html", prediction=label, img_path=result_path, confidence=confidence)
    return render_template("Prediction.html")


@View.route('/LiveMonitor')
def LiveMonitor():
    return render_template('LiveMonitor.html')


@View.route('/video_feed')
def video_feed():
    app = create_app()  # Create the Flask app object
    return Response(gen_frames(app), mimetype='multipart/x-mixed-replace; boundary=frame')


@View.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('neg') == 'NEGATIVE':
            global neg
            neg = not neg
        elif request.form.get('stop') == 'MONITOR':
            if switch == 0:
                camera = cv2.VideoCapture(0)
                switch = 1
            else:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
        elif request.form.get('rec') == 'RECORD':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join(VIDEO_DIR, 'vid_{}.avi'.format(str(now).replace(":", ''))), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif rec == False:
                out.release()
    elif request.method == 'GET':
        return render_template('LiveMonitor.html')
    return render_template('LiveMonitor.html')


# ─── Wildfire Prediction Feature ────────────────────────────────────────────

def run_wildfire_prediction_analysis(year, month=None):
    """
    Load MODIS CSV data, train a Prophet time-series model, and generate
    charts + a folium map for the Fire Eye Wildfire Prediction page.
    Returns (result_dict, error_string).
    """
    import pandas as pd
    import folium
    import branca.colormap as cm_branca
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error

    # ── 1. Load data ──────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        return None, "No CSV files found in the 'data' folder."

    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw['acq_date'] = pd.to_datetime(df_raw['acq_date'])
    df_raw.set_index('acq_date', inplace=True)
    df_raw.sort_index(inplace=True)

    # Drop unneeded columns
    drop_cols = [c for c in ['acq_time', 'satellite', 'instrument', 'version', 'daynight'] if c in df_raw.columns]
    df_raw.drop(columns=drop_cols, inplace=True)

    # India bounding box
    df_india = df_raw[
        (df_raw['latitude'] >= 6.5) & (df_raw['latitude'] <= 37.5) &
        (df_raw['longitude'] >= 68.0) & (df_raw['longitude'] <= 97.5)
    ].copy()

    total_records = len(df_india)

    # Keep lat/lon separately for mapping before aggregating
    df_map_src = df_india.copy()

    # Daily median for time-series
    df_daily = df_india.resample('D')['frp'].median().dropna().to_frame()
    if df_daily.empty:
        return None, "No fire data available after filtering for India."

    # ── 2. Monthly FRP bar chart ──────────────────────────────────────────
    year_daily = df_daily[df_daily.index.year == year]
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor('#1a0502')
    ax1.set_facecolor('#1a0502')
    if not year_daily.empty:
        monthly_frp = year_daily['frp'].resample('ME').mean()
        month_labels = monthly_frp.index.strftime('%b')
        bars = ax1.bar(month_labels, monthly_frp.values, color='#ff4500', edgecolor='#ff8c00', linewidth=0.8)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', color='white', fontsize=8)
    ax1.set_title(f'Monthly Median Fire Radiative Power (FRP) — {year}', color='white', fontsize=13, pad=10)
    ax1.set_xlabel('Month', color='#ccc')
    ax1.set_ylabel('Median FRP (MW)', color='#ccc')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', facecolor=fig1.get_facecolor(), dpi=120)
    buf1.seek(0)
    monthly_chart = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)

    # ── 3. Prophet model ──────────────────────────────────────────────────
    split_date = '2024-01-01'
    df_train = df_daily[df_daily.index < split_date].copy()
    df_test  = df_daily[df_daily.index >= split_date].copy()

    # Fallback: if no data before split use 80/20
    if df_train.empty:
        n = len(df_daily)
        df_train = df_daily.iloc[:int(n * 0.8)]
        df_test  = df_daily.iloc[int(n * 0.8):]

    df_train_prophet = df_train.reset_index().rename(columns={'acq_date': 'ds', 'frp': 'y'})
    df_test_prophet  = df_test.reset_index().rename(columns={'acq_date': 'ds', 'frp': 'y'})

    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(df_train_prophet)

    forecast_df = prophet_model.predict(df_test_prophet[['ds']])

    mae  = round(mean_absolute_error(df_test_prophet['y'], forecast_df['yhat']), 3)
    mape = round(float(np.mean(np.abs((df_test_prophet['y'].values - forecast_df['yhat'].values)
                                       / (df_test_prophet['y'].values + 1e-9))) * 100), 2)

    # ── 4. Forecast vs Actual chart ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    fig2.patch.set_facecolor('#1a0502')
    ax2.set_facecolor('#1a0502')
    ax2.plot(forecast_df['ds'], forecast_df['yhat'], label='Prophet Forecast', color='#ff8c00', linewidth=1.5)
    ax2.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                     alpha=0.25, color='#ff4500', label='Uncertainty interval')
    ax2.scatter(df_test_prophet['ds'], df_test_prophet['y'], label='Actual (Test)', color='white', s=4, alpha=0.8)
    ax2.set_title('Prophet Forecast vs Actual FRP (Test Period)', color='white', fontsize=13, pad=10)
    ax2.set_xlabel('Date', color='#ccc')
    ax2.set_ylabel('FRP (MW)', color='#ccc')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#2a1005', labelcolor='white', fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', facecolor=fig2.get_facecolor(), dpi=120)
    buf2.seek(0)
    forecast_chart = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)

    # ── 5. Rolling median chart ───────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    fig3.patch.set_facecolor('#1a0502')
    ax3.set_facecolor('#1a0502')
    ax3.plot(forecast_df['ds'], forecast_df['yhat'], label='Prophet Forecast', color='#ff8c00', linewidth=1.5)
    ax3.plot(df_test_prophet['ds'], df_test_prophet['y'], label='Actual', color='#87ceeb', linewidth=1, alpha=0.7)
    rolling = df_test_prophet.set_index('ds')['y'].rolling(window=28).median()
    ax3.plot(rolling.index, rolling.values, label='Rolling Median (28d)', color='#90ee90', linewidth=2)
    ax3.set_title('Forecast vs Actual with Rolling Median', color='white', fontsize=13, pad=10)
    ax3.set_xlabel('Date', color='#ccc')
    ax3.set_ylabel('FRP (MW)', color='#ccc')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#2a1005', labelcolor='white', fontsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', facecolor=fig3.get_facecolor(), dpi=120)
    buf3.seek(0)
    rolling_chart = base64.b64encode(buf3.read()).decode('utf-8')
    plt.close(fig3)

    # ── 6. Folium map ─────────────────────────────────────────────────────
    if month:
        map_data = df_map_src[(df_map_src.index.year == year) & (df_map_src.index.month == month)].copy()
        timeframe_label = f"{datetime(year, month, 1).strftime('%B')} {year}"
    else:
        map_data = df_map_src[df_map_src.index.year == year].copy()
        timeframe_label = str(year)

    # Sample to at most 400 points to keep map fast
    if len(map_data) > 400:
        map_data = map_data.sample(400, random_state=42)

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    colormap = cm_branca.LinearColormap(
        colors=['yellow', 'orange', 'red'],
        vmin=0, vmax=200,
        caption='Fire Radiative Power (FRP in MW)'
    )
    colormap.add_to(m)

    if not map_data.empty:
        valid = map_data.dropna(subset=['latitude', 'longitude', 'frp'])
        max_frp = valid['frp'].max() or 1
        min_frp = valid['frp'].min()
        for idx, row in valid.iterrows():
            ratio = (row['frp'] - min_frp) / (max_frp - min_frp + 0.001)
            radius = 3 + ratio * 10
            frp_clipped = min(row['frp'], 200)
            color = colormap(frp_clipped)
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5 + ratio * 0.4,
                tooltip=f"FRP: {row['frp']:.1f} MW | {idx.strftime('%Y-%m-%d')}"
            ).add_to(m)

    os.makedirs(os.path.join('static', 'graphs'), exist_ok=True)
    map_path = os.path.join('static', 'graphs', 'wildfire_map.html')
    m.save(map_path)

    return {
        'monthly_chart':  monthly_chart,
        'forecast_chart': forecast_chart,
        'rolling_chart':  rolling_chart,
        'map_url':        'static/graphs/wildfire_map.html',
        'mae':            mae,
        'mape':           mape,
        'train_size':     len(df_train),
        'test_size':      len(df_test),
        'total_records':  total_records,
        'year':           year,
        'month':          month,
        'timeframe':      timeframe_label,
        'csv_files':      [os.path.basename(f) for f in csv_files],
    }, None


@View.route('/WildfirePrediction', methods=['GET', 'POST'])
def WildfirePrediction():
    result = None
    error  = None
    if request.method == 'POST':
        try:
            year  = int(request.form.get('year', 2024))
            month = request.form.get('month', '')
            month = int(month) if month.strip() else None
            result, error = run_wildfire_prediction_analysis(year, month)
        except Exception as e:
            error = f"Prediction failed: {str(e)}"
    return render_template('WildfirePrediction.html', result=result, error=error)

