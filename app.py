from flask import jsonify, request
import numpy as np
import cv2

@app.route("/api/detect", methods=["POST"])
def api_detect():

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    # 🔥 FAST decode (no unnecessary copies)
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)

    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    # 🔥 BIG PERFORMANCE BOOST (resize before processing)
    h, w = image.shape[:2]
    if w > 1000:
        scale = 1000 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    try:
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