import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

# Define the midpoint function
def mid_point(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Create output directory if it doesn't exist
if not os.path.exists('predict'):
    os.makedirs('predict')

# Width of reference leftmost object (in inches)
leftmostWidth = 0.5

# Process all images in the dataset folder
for filename in os.listdir('dataset/images'):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image_path = os.path.join('dataset/images', filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Perform edge detection, dilation, and erosion
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # Find and sort contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        # Find the leftmost contour for calibration
        leftmostX = None
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (x, y, w, h) = cv2.boundingRect(box.astype("int"))
            if leftmostX is None or x < leftmostX:
                leftmostX = x
                leftmostBox = box

        # Calculate the pixelsPerMetric
        midpoint = ((leftmostBox[0][0] + leftmostBox[1][0]) // 2, (leftmostBox[0][1] + leftmostBox[1][1]) // 2)
        pixelsPerMetric = np.abs(midpoint[0] - leftmostX) / leftmostWidth

        # Process each contour
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # Compute and draw midpoints
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = mid_point(tl, tr)
            (blbrX, blbrY) = mid_point(bl, br)
            (tlblX, tlblY) = mid_point(tl, bl)
            (trbrX, trbrY) = mid_point(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            # Compute object dimensions
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            # Draw object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Save the output image
            output_path = os.path.join('predict', f'predicted_{filename}')
            cv2.imwrite(output_path, orig)

        print(f"Processed and saved: {output_path}")

print("All images processed and saved in the 'predict' folder.")
