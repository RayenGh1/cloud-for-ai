import os
import time
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pkl")

# Wacht tot het model bestaat (Docker volume)
while not os.path.exists(MODEL_PATH):
    print("Waiting for model...")
    time.sleep(1)

model = joblib.load(MODEL_PATH)
print("Model loaded")


# Homepage met formulier
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Predict via formulier
@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "cap-diameter": float(request.form["cap-diameter"]),
        "stem-height": float(request.form["stem-height"]),
        "stem-width": float(request.form["stem-width"]),
        "cap-shape": request.form["cap-shape"],
        "cap-surface": request.form["cap-surface"],
        "cap-color": request.form["cap-color"],
        "does-bruise-or-bleed": request.form["does-bruise-or-bleed"],
        "gill-attachment": request.form["gill-attachment"],
        "gill-spacing": request.form["gill-spacing"],
        "gill-color": request.form["gill-color"],
        "stem-root": request.form["stem-root"],
        "stem-surface": request.form["stem-surface"],
        "stem-color": request.form["stem-color"],
        "veil-type": request.form["veil-type"],
        "veil-color": request.form["veil-color"],
        "has-ring": request.form["has-ring"],
        "ring-type": request.form["ring-type"],
        "spore-print-color": request.form["spore-print-color"],
        "habitat": request.form["habitat"],
        "season": request.form["season"]
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
