import os
from flask import Flask, request, jsonify
from predictor import MultiLabelPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model_multilabel.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "label_vocab.json")
GROUPS_PATH = os.path.join(BASE_DIR, "label_groups.json")

app = Flask(__name__)

predictor = MultiLabelPredictor(MODEL_PATH, VOCAB_PATH, GROUPS_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():

    if "file" not in request.files:
        return jsonify({"error": "Missing file. Use multipart/form-data with key 'file'."}), 400

    f = request.files["file"]
    img_bytes = f.read()

    if not img_bytes:
        return jsonify({"error": "Empty file uploaded."}), 400

   
    threshold = request.form.get("threshold", "0.5")
    top_k = request.form.get("top_k", "5")

    try:
        threshold = float(threshold)
    except ValueError:
        threshold = 0.5

    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 5

    pred = predictor.predict(img_bytes, threshold=threshold, top_k=top_k)

    return jsonify({
        "filename": f.filename,
        "threshold": threshold,
        "top_k": top_k,
        **pred
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6060, debug=False)
