import cv2
import numpy as np

# Load image
image = cv2.imread("rings.jpg")

# Resize (optional)
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blur = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=20,
    maxRadius=100
)

# Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # circle
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # center

    print(f"Detected rings: {len(circles)}")
else:
    print("No rings detected")

# Show image
cv2.imshow("Detected Rings", image)
cv2.waitKey(0)
cv2.destroyAllWindows()