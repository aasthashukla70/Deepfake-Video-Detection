from flask import Flask, render_template, request
import os

from src.video_predict_final import predict_video


app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)



@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route("/predict", methods=["POST"])
def predict():

    video = request.files["video"]


    if video.filename == "":

        return "No video selected."


    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        video.filename
    )


    video.save(save_path)



    prediction, confidence = predict_video(
        save_path
    )


    return render_template(
        "result.html",
        prediction=prediction,
        confidence=f"{confidence:.2f}"
    )



if __name__ == "__main__":

    app.run(
        debug=True
    )