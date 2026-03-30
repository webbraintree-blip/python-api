import os
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from qr_detect import analyze_image

app = Flask(__name__)

# 🔥 CORS ENABLE
CORS(app)


@app.route("/api/detect", methods=["POST"])
def api_detect():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # 🔥 READ IMAGE
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if image is None or image.size == 0:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        # 🔥 HANDLE CHANNELS SAFELY
        if len(image.shape) == 2:
            # already grayscale → convert to BGR for consistency
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image

        # 🔥 RESIZE (balanced for speed + accuracy)
        h, w = image_color.shape[:2]
        if w > 1400:
            scale = 1400 / w
            image_color = cv2.resize(image_color, (int(w * scale), int(h * scale)))

        # 🔥 FIRST TRY (COLOR)
        analysis = analyze_image(image_color)

        # 🔥 SECOND TRY (GRAYSCALE fallback)
        if not any(r.get("value") for r in analysis.get("results", [])):

            if len(image_color.shape) == 3:
                gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_color

            analysis = analyze_image(gray)

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


# 🔥 HEALTH CHECK (important for Render)
@app.route("/", methods=["GET"])
def health():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)