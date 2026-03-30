try:
    import cv2
except ImportError:
    print("Missing dependency: opencv-python")
    print("Install it with: python -m pip install opencv-python")
    raise SystemExit(1)

import numpy as np

try:
    from pylibdmtx.pylibdmtx import decode as decode_datamatrix
    DMTX_IMPORT_ERROR = None
except ImportError:
    decode_datamatrix = None
    DMTX_IMPORT_ERROR = "ImportError"
except Exception as exc:
    decode_datamatrix = None
    DMTX_IMPORT_ERROR = str(exc)

try:
    import zxingcpp
    ZXING_IMPORT_ERROR = None
except ImportError:
    zxingcpp = None
    ZXING_IMPORT_ERROR = "ImportError"
except Exception as exc:
    zxingcpp = None
    ZXING_IMPORT_ERROR = str(exc)


IMAGE_PATH = "working_0007.jpeg"
DISPLAY_MAX_WIDTH = 1400


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image '{path}'")
    return image


def resize_for_display(image, max_width=DISPLAY_MAX_WIDTH):
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()

    scale = max_width / float(width)
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def detect_rings(image):
    """
    Detect ring circles.
    The ring count is only used to associate nearby tags; decoding depends on tags.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    min_dim = min(image.shape[:2])
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(45, int(min_dim * 0.05)),
        param1=110,
        param2=28,
        minRadius=max(20, int(min_dim * 0.03)),
        maxRadius=max(50, int(min_dim * 0.085)),
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles.tolist(), key=lambda item: (item[1], item[0]))

    deduped = []
    for x, y, r in circles:
        duplicate = False
        for px, py, pr in deduped:
            if np.hypot(x - px, y - py) < max(r, pr) * 0.8:
                duplicate = True
                break
        if not duplicate:
            deduped.append((x, y, r))

    return deduped


def detect_tags(image):
    """
    Detect rectangular jewelry tags.
    Tags are bright, compact rectangles and are easier to localize than the code itself.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    tags = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), _ = rect
        if min(w, h) <= 0:
            continue

        area = w * h
        if area < image_area * 0.002 or area > image_area * 0.03:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / float(short_side)
        if not 1.8 <= aspect <= 4.8:
            continue

        points = cv2.boxPoints(rect).astype(np.float32)
        x, y, bw, bh = cv2.boundingRect(points.astype(np.int32))
        tags.append(
            {
                "rect": rect,
                "points": points,
                "bbox": (x, y, x + bw, y + bh),
                "center": (cx, cy),
                "area": area,
            }
        )

    tags.sort(key=lambda item: (item["center"][1], item["center"][0]))

    deduped = []
    for tag in tags:
        keep = True
        cx, cy = tag["center"]
        for existing in deduped:
            ex, ey = existing["center"]
            if np.hypot(cx - ex, cy - ey) < 20:
                keep = False
                break
        if keep:
            deduped.append(tag)

    return deduped


def associate_rings_and_tags(rings, tags):
    assignments = []
    used = set()

    for x, y, r in rings:
        best_index = None
        best_score = float("inf")
        for idx, tag in enumerate(tags):
            if idx in used:
                continue

            tx, ty = tag["center"]
            distance = np.hypot(tx - x, ty - y)
            vertical_gap = abs(ty - y)
            if distance > r * 6.0 or vertical_gap > r * 2.2:
                continue

            score = distance - 0.2 * abs(tx - x)
            if score < best_score:
                best_score = score
                best_index = idx

        assignments.append(best_index)
        if best_index is not None:
            used.add(best_index)

    return assignments


def order_quad(points):
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def warp_tag(image, tag):
    rect = order_quad(tag["points"])
    top_width = np.linalg.norm(rect[1] - rect[0])
    bottom_width = np.linalg.norm(rect[2] - rect[3])
    left_height = np.linalg.norm(rect[3] - rect[0])
    right_height = np.linalg.norm(rect[2] - rect[1])

    width = max(110, int(max(top_width, bottom_width)))
    height = max(40, int(max(left_height, right_height)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped


def extract_symbol_patch(tag_image):
    """
    The 2D code sits on the left side of the tag.
    We crop that area so the decoder sees mostly the symbol, not the whole label text.
    """
    h, w = tag_image.shape[:2]
    x1 = int(w * 0.04)
    x2 = int(w * 0.34)
    y1 = int(h * 0.08)
    y2 = int(h * 0.82)
    return tag_image[y1:y2, x1:x2].copy()


def generate_decoder_inputs(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)

    outputs = []
    for scale in (4.0, 6.0):
        enlarged = cv2.resize(
            denoised,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        sharpen = cv2.addWeighted(enlarged, 1.8, cv2.GaussianBlur(enlarged, (3, 3), 0), -0.8, 0)
        binary = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary_inv = cv2.bitwise_not(binary)

        outputs.extend(
            [
                sharpen,
                binary,
                binary_inv,
            ]
        )

    return outputs


def decode_symbol(crop):
    for prepared in generate_decoder_inputs(crop):
        for angle in (0, 90, 180, 270):
            rotated = prepared
            if angle == 90:
                rotated = cv2.rotate(prepared, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(prepared, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(prepared, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if decode_datamatrix is not None:
                results = decode_datamatrix(rotated)
                if results:
                    value = results[0].data.decode("utf-8", errors="replace")
                    rect = results[0].rect
                    return value, rect

            if zxingcpp is not None:
                results = zxingcpp.read_barcodes(rotated)
                if results:
                    return results[0].text, results[0].position

    return None


def position_to_points(position):
    return np.array(
        [
            [position.top_left.x, position.top_left.y],
            [position.top_right.x, position.top_right.y],
            [position.bottom_right.x, position.bottom_right.y],
            [position.bottom_left.x, position.bottom_left.y],
        ],
        dtype=np.int32,
    )


def detect_visible_codes(image):
    if zxingcpp is None:
        return []

    detections = []
    for result in zxingcpp.read_barcodes(image):
        points = position_to_points(result.position)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        detections.append(
            {
                "text": result.text,
                "format": str(result.format),
                "points": points,
                "center": (center_x, center_y),
            }
        )

    detections.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return detections


def draw_tag_label(image, tag, text, color):
    points = np.round(tag["points"]).astype(np.int32)
    cv2.polylines(image, [points], True, color, 2)

    x, y, _, _ = tag["bbox"]
    cv2.putText(
        image,
        text,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_code_label(image, detection, index, color):
    points = detection["points"]
    cv2.polylines(image, [points], True, color, 2)

    x = int(points[:, 0].min())
    y = int(points[:, 1].min())
    cv2.putText(
        image,
        f"Code {index}: {detection['text']}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def analyze_image(image):
    annotated = image.copy()
    detections = detect_visible_codes(image)
    results = []
    formats = []
    used_label_fallback = False

    if detections:
        formats = sorted({item["format"] for item in detections})
        for code_index, detection in enumerate(detections, start=1):
            draw_code_label(annotated, detection, code_index, (0, 255, 0))
            results.append(
                {
                    "index": code_index,
                    "value": detection["text"],
                    "format": detection["format"],
                }
            )
    else:
        used_label_fallback = True
        tags = detect_tags(image)

        for tag_index, tag in enumerate(tags, start=1):
            warped_tag = warp_tag(image, tag)
            symbol_patch = extract_symbol_patch(warped_tag)
            decoded = decode_symbol(symbol_patch)

            if decoded is None:
                draw_tag_label(annotated, tag, f"Label {tag_index}: Not detected", (0, 0, 255))
                results.append(
                    {
                        "index": tag_index,
                        "value": None,
                        "format": None,
                    }
                )
                continue

            value, _ = decoded
            draw_tag_label(annotated, tag, f"Label {tag_index}: {value}", (0, 255, 0))
            results.append(
                {
                    "index": tag_index,
                    "value": value,
                    "format": "Fallback decode",
                }
            )

    return {
        "annotated_image": annotated,
        "results": results,
        "formats": formats,
        "used_label_fallback": used_label_fallback,
    }


def main():
    if decode_datamatrix is None and zxingcpp is None:
        print("Missing decoder dependency.")
        print("Install one of these:")
        print("1. python -m pip install zxing-cpp")
        print("2. python -m pip install pylibdmtx")
        if DMTX_IMPORT_ERROR:
            print(f"pylibdmtx import detail: {DMTX_IMPORT_ERROR}")
        if ZXING_IMPORT_ERROR:
            print(f"zxing-cpp import detail: {ZXING_IMPORT_ERROR}")
        return

    if decode_datamatrix is None and DMTX_IMPORT_ERROR:
        print("pylibdmtx is unavailable, using zxing-cpp fallback.")
        print(f"pylibdmtx import detail: {DMTX_IMPORT_ERROR}")

    image = load_image(IMAGE_PATH)
    analysis = analyze_image(image)
    annotated = analysis["annotated_image"]
    results = analysis["results"]

    if analysis["formats"]:
        print(f"Detected visible codes: {len(results)}")
        print("Detected code type(s):", ", ".join(analysis["formats"]))
        for item in results:
            print(f"Code {item['index']} -> Value: {item['value']}")
    elif analysis["used_label_fallback"]:
        print("Full-image code detection found nothing. Used label fallback.")
        for item in results:
            if item["value"] is None:
                print(f"Label {item['index']} -> Code: Not detected")
            else:
                print(f"Label {item['index']} -> Value: {item['value']}")

    if not any(item["value"] for item in results):
        print("No codes decoded.")
        print("If this still happens, send one close-up crop of a single label.")

    cv2.imshow("Code Detection", resize_for_display(annotated))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
