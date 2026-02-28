# ============================================
# CardioScan AI — ECG Heartbeat Classification
# CNN + LSTM Hybrid Model | PyTorch
# ============================================

import os
import io
import csv
import base64
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================
# Flask setup
# ============================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# Device setup
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# ============================================
# CNN + LSTM Model Definition
# (must match exactly what was trained)
# ============================================

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm     = nn.LSTM(input_size=128, hidden_size=128,
                                num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,  1),  nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.dropout1(x)
        x = self.classifier(x)
        return x

# ============================================
# Load trained model
# ============================================

model = CNN_LSTM()
model_path = "../model/best_cnn_lstm_model.pth"
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.to(device)
model.eval()
print("ECG CNN+LSTM model loaded successfully.")

# ============================================
# Signal parsing — accepts CSV row or plain values
# ============================================

def parse_signal(file_path):
    """
    Accepts:
      - A CSV file where one row contains 187 (or 188) numeric values
        (if 188 values, the last column is the label — we drop it)
      - A plain text file with comma- or newline-separated numbers
    Returns: np.array of shape (187,) dtype float32
    """
    signals = []

    with open(file_path, "r") as f:
        content = f.read().strip()

    # Try CSV parsing first
    reader = csv.reader(io.StringIO(content))
    for row in reader:
        values = []
        for v in row:
            v = v.strip()
            if v:
                try:
                    values.append(float(v))
                except ValueError:
                    pass
        if len(values) >= 187:
            signals = values[:187]   # drop label column if present
            break

    if not signals:
        # Fallback: plain whitespace/newline-separated values
        for token in content.replace(",", " ").split():
            try:
                signals.append(float(token))
            except ValueError:
                pass

    if len(signals) < 187:
        raise ValueError(
            f"Signal too short: got {len(signals)} values, expected 187."
        )

    return np.array(signals[:187], dtype=np.float32)

# ============================================
# Generate waveform chart — base64 PNG
# ============================================

def render_waveform(signal, prediction, probability):
    """
    Renders the ECG waveform with styled annotation.
    Returns base64-encoded PNG string.
    """
    is_abnormal  = prediction == 1
    wave_color   = "#ef4444" if is_abnormal else "#22c55e"
    fill_alpha   = 0.12
    bg_color     = "#0d0d0d"

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    x = np.arange(187)

    # Subtle grid
    ax.grid(True, color="#1f1f1f", linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)

    # Baseline
    ax.axhline(y=0, color="#333333", linewidth=0.8, linestyle="--")

    # Fill under curve
    ax.fill_between(x, signal, 0, alpha=fill_alpha, color=wave_color)

    # Main signal line
    ax.plot(x, signal, color=wave_color, linewidth=1.8, zorder=3)

    # Annotate the R-peak (highest point)
    peak_idx = int(np.argmax(signal))
    peak_val = signal[peak_idx]
    ax.annotate(
        f"R-peak: {peak_val:.3f}",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx + 12, peak_val + 0.05),
        color=wave_color,
        fontsize=8,
        fontfamily="monospace",
        arrowprops=dict(arrowstyle="->", color=wave_color, lw=1),
    )

    # Styling
    ax.set_xlim(0, 186)
    ax.tick_params(colors="#444444", labelsize=8)
    ax.set_xlabel("Time Steps (0 – 186)", color="#555555",
                  fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Amplitude", color="#555555",
                  fontsize=8, fontfamily="monospace")
    for spine in ax.spines.values():
        spine.set_color("#1f1f1f")

    # Title
    label_txt = "ABNORMAL HEARTBEAT" if is_abnormal else "NORMAL HEARTBEAT"
    fig.suptitle(
        f"{label_txt}   ·   Confidence: {probability:.1%}",
        color=wave_color, fontsize=10, fontfamily="monospace",
        fontweight="bold", y=1.01
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=bg_color, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ============================================
# Generate confidence gauge — base64 PNG
# ============================================

def render_confidence_chart(prob_abnormal):
    """
    Renders a horizontal probability bar for Normal vs Abnormal.
    Returns base64-encoded PNG string.
    """
    prob_normal = 1.0 - prob_abnormal
    bg_color    = "#0d0d0d"

    fig, ax = plt.subplots(figsize=(5, 2.2), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    categories = ["Normal", "Abnormal"]
    values     = [prob_normal, prob_abnormal]
    colors     = ["#22c55e", "#ef4444"]

    bars = ax.barh(categories, values, color=colors, height=0.45,
                   edgecolor="#1f1f1f", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            color="white", fontsize=9, fontfamily="monospace", fontweight="bold"
        )

    ax.axvline(x=0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", color="#555555", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#555555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1f1f1f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=bg_color, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ============================================
# Core prediction function
# ============================================

def predict_ecg(file_path, threshold=0.5):
    """
    Load a CSV signal file, run CNN+LSTM inference, return result dict.
    """
    # Parse signal
    signal = parse_signal(file_path)

    # Normalize to [0, 1] if not already (safety guard)
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max > 1.0 or sig_min < 0.0:
        signal = (signal - sig_min) / (sig_max - sig_min + 1e-8)

    # Build tensor: (1, 1, 187)
    tensor = torch.tensor(signal, dtype=torch.float32)\
                  .unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prob_abnormal = model(tensor).item()

    prediction = 1 if prob_abnormal >= threshold else 0
    prob_normal = 1.0 - prob_abnormal

    # Signal stats
    r_peak      = float(np.max(signal))
    r_peak_idx  = int(np.argmax(signal))
    mean_amp    = float(np.mean(signal))
    std_amp     = float(np.std(signal))

    # Render charts
    waveform_b64   = render_waveform(signal, prediction, prob_abnormal)
    confidence_b64 = render_confidence_chart(prob_abnormal)

    return {
        "prediction":       prediction,
        "label":            "ABNORMAL" if prediction == 1 else "NORMAL",
        "prob_abnormal":    round(prob_abnormal * 100, 2),
        "prob_normal":      round(prob_normal   * 100, 2),
        "confidence":       round(max(prob_abnormal, prob_normal) * 100, 2),
        "r_peak":           round(r_peak,     4),
        "r_peak_idx":       r_peak_idx,
        "mean_amplitude":   round(mean_amp,   4),
        "std_amplitude":    round(std_amp,    4),
        "signal_length":    len(signal),
        "waveform_image":   waveform_b64,
        "confidence_image": confidence_b64,
        "action":           "Flag for cardiologist review." if prediction == 1
                            else "No immediate action required.",
    }

# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."})

    allowed = {".csv", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format '{ext}'. Please upload a .csv or .txt file."})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result = predict_ecg(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    print("\nStarting CardioScan AI — ECG Heartbeat Classification App")
    print("Open browser at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
