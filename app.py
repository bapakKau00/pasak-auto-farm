import requests
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
# Load environment variables from .env file
load_dotenv(override=True)


THINGSBOARD_URL = os.getenv("THINGSBOARD_URL", "https://insight.ipinfraiot.com")
KPI_THRESHOLD = float(os.getenv("KPI_THRESHOLD", "0.25"))

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

print(f"DEBUG: Loaded WEATHER_DEVICE_ID: {os.getenv('WEATHER_DEVICE_ID')}")
print(f"DEBUG: Loaded SENSOR_DEVICE_IDS: {os.getenv('SENSOR_DEVICE_IDS')}")

# Device IDs - Load as lists
def get_device_list(env_var):
    val = os.getenv(env_var, "")
    if not val:
        return []
    return [d.strip() for d in val.split(",") if d.strip()]

SENSOR_DEVICES = get_device_list("SENSOR_DEVICE_IDS")
WEATHER_DEVICE = os.getenv("WEATHER_DEVICE_ID")

MODEL_PATH = os.getenv("MODEL_PATH", "abnormal_kpi_rf.joblib")

# Botpress Config
BOTPRESS_WEBHOOK_URL = os.getenv("BOTPRESS_WEBHOOK_URL")
BOTPRESS_CONVERSATION_ID = os.getenv("BOTPRESS_CONVERSATION_ID")
BOTPRESS_EVENT_NAME = os.getenv("BOTPRESS_EVENT_NAME", "auto_farm_report")


# ======================
# LOGIN
# ======================

def tb_login():
    url = f"{THINGSBOARD_URL}/api/auth/login"
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    token = response.json()['token']
    return token


# ======================
# FETCH TELEMETRY
# ======================

def fetch_telemetry(device_id, keys, token):
    url = f"{THINGSBOARD_URL}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
    headers = {
        "X-Authorization": f"Bearer {token}"
    }
    params = {
        "keys": keys,
        "limit": 1,
        "useStrictDataTypes": "false"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching telemetry for {device_id}: Status {response.status_code}")
            print(f"Response: {response.text}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request failed for {device_id}: {e}")
        return None


# ======================
# GET BATCH SENSOR VALUES
# ======================

def get_all_sensor_values(token):
    # Fetch common weather data once
    weather_data = fetch_telemetry(WEATHER_DEVICE, "temperature", token)
    temp = float(weather_data['temperature'][0]['value']) if weather_data and 'temperature' in weather_data else 0.0

    all_features = []
    
    print(f"Fetching data from {len(SENSOR_DEVICES)} devices...")
    
    for i, device_id in enumerate(SENSOR_DEVICES):
        try:
            # Fetch all 5 sensor values (N, P, K, pH, EC) in one call
            data = fetch_telemetry(device_id, "nitrogen,phosphorus,potassium,pH,ec", token)
            
            if not data:
                continue

            N = float(data['nitrogen'][0]['value']) if 'nitrogen' in data else 0.0
            P = float(data['phosphorus'][0]['value']) if 'phosphorus' in data else 0.0
            K = float(data['potassium'][0]['value']) if 'potassium' in data else 0.0
            pH = float(data['ph'][0]['value']) if 'ph' in data else 0.0
            EC = float(data['ec'][0]['value']) if 'ec' in data else 0.0
            
            features = [N, P, K, pH, EC, temp]
            all_features.append(features)
        except Exception as e:
            print(f"Skipping device {device_id} (Set {i+1}) due to error: {e}")
            
    return all_features


# ======================
# WHATSAPP NOTIFICATION
# ======================

def send_whatsapp_notification(avg_kpi, num_devices, peak_kpi=None, peak_device=None, alert_plots=None):
    if not BOTPRESS_WEBHOOK_URL or "YOUR_BOTPRESS_WEBHOOK_URL" in BOTPRESS_WEBHOOK_URL:
        print("WhatsApp Notification skipped: BOTPRESS_WEBHOOK_URL not configured in .env")
        return

    from datetime import datetime
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    alert_count = len(alert_plots) if alert_plots else 0
    status_emoji = "⚠️" if alert_count > 0 else "✅"
    
    # Message formatting for WhatsApp
    message_text = f"📢 *Auto-Farm Report*\n\n" \
                   f"{status_emoji} Status: *{'ALERT' if alert_count > 0 else 'NORMAL'}*\n" \
                   f"📍 Processed *{num_devices}* plots.\n"
    
    if alert_count > 0:
        message_text += f"🚨 Alerts in: *{', '.join(alert_plots)}*\n"
    
    message_text += f"📊 Average Abnormal KPI: *{avg_kpi:.2f}*\n"
    
    if peak_kpi is not None:
        message_text += f"🚀 Peak KPI: *{peak_kpi:.2f}* ({peak_device})\n"
    
    message_text += f"\n🕒 Time: {time_str}"

    payload = {
        "event_name": BOTPRESS_EVENT_NAME,
        "status": "ALERT" if alert_count > 0 else "NORMAL",
        "alert_plots": alert_plots if alert_plots else [],
        "text": message_text,
        "avg_kpi": round(float(avg_kpi), 2),
        "peak_kpi": round(float(peak_kpi), 2) if peak_kpi is not None else None,
        "peak_device": peak_device,
        "num_devices": int(num_devices),
        "conversationId": BOTPRESS_CONVERSATION_ID,
        "timestamp": time_str
    }

    try:
        response = requests.post(BOTPRESS_WEBHOOK_URL, json=payload)
        if response.status_code == 200 or response.status_code == 201:
            print("WhatsApp Webhook triggered successfully.")
        else:
            print(f"Failed to trigger Webhook. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error sending Webhook notification: {e}")


# ======================
# MAIN
# ======================

model = joblib.load(MODEL_PATH)

def run_prediction():
    try:
        token = tb_login()
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    all_features = get_all_sensor_values(token)
    
    if not all_features:
        print("No data collected from sensors.")
        return

    print(f"\nProcessing {len(all_features)} plot sets...\n")
    
    X = np.array(all_features)
    predictions = model.predict(X)
    alert_plots = []

    for i, (features, kpi) in enumerate(zip(all_features, predictions)):
        plot_name = f"Plot {i+1}"
        if kpi >= KPI_THRESHOLD:
            alert_plots.append(plot_name)
            
        print(f"{plot_name} Results ({SENSOR_DEVICES[i] if i < len(SENSOR_DEVICES) else 'Unknown'}):")
        print(f"  Inputs: N={features[0]}, P={features[1]}, K={features[2]}, pH={features[3]}, EC={features[4]}, temp={features[5]}")
        print(f"  Predicted Abnormal KPI: {kpi:.2f}\n")

    if len(predictions) > 0:
        avg_kpi = np.mean(predictions)
        
        # Find peak KPI and corresponding device
        peak_idx = np.argmax(predictions)
        peak_kpi = predictions[peak_idx]
        peak_plot_name = f"Plot {peak_idx + 1}"

        print("-" * 30)
        print(f"SUMMARY REPORT")
        print(f"Total Plots:     {len(predictions)}")
        if alert_plots:
            print(f"Alerts in:       {', '.join(alert_plots)}")
        print(f"Average KPI:     {avg_kpi:.2f}")
        print(f"Peak KPI:        {peak_kpi:.2f} ({peak_plot_name})")
        print("-" * 30)
        
        # Send WhatsApp Notification ONLY if there are alerts
        if alert_plots:
            send_whatsapp_notification(avg_kpi, len(predictions), peak_kpi, peak_plot_name, alert_plots)
        else:
            print("WhatsApp Notification skipped: All plots are within normal range.")


if __name__ == "__main__":
    run_prediction()
