import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_foot_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (512, 512))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    foot_contour = contours[0]
    mask = np.zeros_like(resized)
    cv2.drawContours(mask, [foot_contour], -1, 255, -1)
    foot_only = cv2.bitwise_and(resized, resized, mask=mask)
    rect = cv2.minAreaRect(foot_contour)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    center = (resized.shape[1] // 2, resized.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(foot_only, M, (resized.shape[1], resized.shape[0]))
    x, y, w, h = cv2.boundingRect(foot_contour)
    cropped = rotated[y:y+h, x:x+w]
    final = cv2.resize(cropped, (256, 256))
    cv2.imwrite(save_path, final)
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    processed_image = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], "input_" + filename)
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + filename)
            file.save(input_path)
            success = preprocess_foot_image(input_path, output_path)
            if success:
                processed_image = "processed/" + "processed_" + filename
    return render_template("index.html", processed_image=processed_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
