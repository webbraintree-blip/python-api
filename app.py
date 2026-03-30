
from flask import jsonify

@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]

    file_bytes = file.read()
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    analysis = analyze_image(image)

    return jsonify({
        "success": True,
        "formats": analysis["formats"],
        "results": analysis["results"]
    })