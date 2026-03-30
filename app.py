import os
import base64
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from qr_detect import analyze_image, resize_for_display


app = Flask(__name__)

# 🔥 CORS FIX
CORS(app)


# ==============================
# HTML UI (optional - your old UI)
# ==============================
PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>QR Scanner</title>
</head>
<body>
    <h2>Upload Image</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Scan</button>
    </form>

    {% if result %}
        <h3>Results:</h3>
        <pre>{{ result }}</pre>
        <img src="data:image/jpeg;base64,{{ image }}">
    {% endif %}
</body>
</html>
"""


def image_to_base64(image):
    ok, encoded = cv2.imencode(".jpg", resize_for_display(image, max_width=1200))
    if not ok:
        raise ValueError("Image encode failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


# ==============================
# UI ROUTE (browser use)
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is not None:
                analysis = analyze_image(image)

                result = analysis["results"]
                image_b64 = image_to_base64(analysis["annotated_image"])

    return render_template_string(PAGE_TEMPLATE, result=result, image=image_b64)


# ==============================
# 🔥 API ROUTE (MAIN USE)
# ==============================
@app.route("/api/detect", methods=["POST"])
def api_detect():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # 🔥 FAST decode
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        # 🔥 PERFORMANCE BOOST (resize)
        h, w = image.shape[:2]
        if w > 1000:
            scale = 1000 / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        analysis = analyze_image(image)

        return jsonify({
            "success": True,
            "formats": analysis.get("formats", []),
            "results": analysis.get("results", [])
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)