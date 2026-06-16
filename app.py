from flask import Flask, render_template, request
from predict import predict_fruit
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file selected"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    fruit, confidence = predict_fruit(filepath)

    return render_template(
        "index.html",
        prediction=fruit,
        confidence=round(confidence,2),
        image=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)