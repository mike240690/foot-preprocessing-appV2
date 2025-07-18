from flask import Flask, request, render_template, url_for
import os
import cv2
import uuid

app = Flask(__name__)

# Ensure 'static/' folder exists
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save original image to static/original_
        original_filename = f"original_{uuid.uuid4().hex}.jpg"
        original_path = os.path.join(STATIC_FOLDER, original_filename)
        file.save(original_path)

        # Read and process the image (example: grayscale)
        img = cv2.imread(original_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save processed image
        processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
        processed_path = os.path.join(STATIC_FOLDER, processed_filename)
        cv2.imwrite(processed_path, gray)

        # Render result with image URL using Flask's static route
        return render_template("result.html", processed_image=processed_filename)

    return render_template("index.html")
