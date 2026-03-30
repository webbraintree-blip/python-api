import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template_string, request

from qr_detect import analyze_image, resize_for_display


app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Label Code Scanner</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-2: #1f2937;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #22c55e;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Segoe UI", sans-serif;
            background: linear-gradient(160deg, #0f172a, #111827 55%, #1e293b);
            color: var(--text);
        }

        .wrap {
            max-width: 1100px;
            margin: 0 auto;
            padding: 32px 20px 48px;
        }

        .card {
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
        }

        h1 {
            margin: 0 0 10px;
            font-size: 32px;
        }

        p {
            color: var(--muted);
            line-height: 1.5;
        }

        form {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        input[type="file"] {
            flex: 1 1 300px;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: var(--panel-2);
            color: var(--text);
        }

        button {
            border: 0;
            border-radius: 12px;
            background: var(--accent);
            color: #052e16;
            padding: 12px 18px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
        }

        .error {
            margin-top: 16px;
            color: #fecaca;
            background: rgba(127, 29, 29, 0.35);
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-radius: 12px;
            padding: 12px 14px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 20px;
            margin-top: 24px;
        }

        .panel {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 18px;
        }

        img {
            width: 100%;
            display: block;
            border-radius: 12px;
        }

        .pill {
            display: inline-block;
            margin-right: 8px;
            margin-bottom: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.16);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.25);
        }

        .result-list {
            display: grid;
            gap: 10px;
            margin-top: 14px;
        }

        .result-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px 14px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .result-item strong {
            color: #86efac;
        }

        .muted {
            color: var(--muted);
        }

        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h1>Label Code Scanner</h1>
            <p>Upload a jewelry label image and the app will scan visible 2D codes from the labels, then show the extracted values and an annotated preview.</p>

            <form method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Scan Image</button>
            </form>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            {% if result %}
                <div class="grid">
                    <div class="panel">
                        <h2>Annotated Preview</h2>
                        <img src="data:image/jpeg;base64,{{ result.annotated_image }}" alt="Annotated detection output">
                    </div>

                    <div class="panel">
                        <h2>Detected Values</h2>
                        {% if result.formats %}
                            <div>
                                {% for fmt in result.formats %}
                                    <span class="pill">{{ fmt }}</span>
                                {% endfor %}
                            </div>
                        {% endif %}

                        <div class="result-list">
                            {% for item in result.results %}
                                <div class="result-item">
                                    <div><strong>Code {{ item.index }}</strong></div>
                                    <div>{{ item.value if item.value else "Not detected" }}</div>
                                    {% if item.format %}
                                        <div class="muted">{{ item.format }}</div>
                                    {% endif %}
                                </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


def image_to_base64(image):
    ok, encoded = cv2.imencode(".jpg", resize_for_display(image, max_width=1500))
    if not ok:
        raise ValueError("Could not encode annotated image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            error = "Please choose an image file first."
        else:
            file_bytes = upload.read()
            image_array = np.frombuffer(file_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                error = "Uploaded file could not be read as an image."
            else:
                analysis = analyze_image(image)
                result = {
                    "results": analysis["results"],
                    "formats": analysis["formats"],
                    "annotated_image": image_to_base64(analysis["annotated_image"]),
                }

    return render_template_string(PAGE_TEMPLATE, result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
