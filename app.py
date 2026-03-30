import os
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from qr_detect import analyze_image

app = Flask(__name__)
CORS(app)


@app.route("/api/detect", methods=["POST"])
def api_detect():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # 🔥 decode image
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        # 🔥 RESIZE (balanced)
        h, w = image.shape[:2]
        if w > 1400:
            scale = 1400 / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # 🔥 TRY BOTH COLOR + GRAYSCALE (BEST)
        analysis = analyze_image(image)

        if not any(r.get("value") for r in analysis.get("results", [])):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)