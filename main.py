import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression

src_img = cv.imread("./imgs/ind_coins101.jpg")  # main img
template = cv.imread("./imgs/ind_coins100.jpg")  # temp img
img_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

(tH, tW) = template.shape[:2]
print("src_img shape: ", src_img.shape[:2])
(tH_src, tW_src) = template.shape[:2]


result = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)

(yCoords, xCoords) = np.where(result >= 0.5)
clone = src_img.copy()

print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
    # draw the bounding box on the image
    cv.rectangle(clone, (x, y), (x + tW, y + tH), (255, 0, 0), 3)
# show our output image *before* applying non-maxima suppression
cv.imwrite("./imgs/Before_NMS.png", clone)

rects = []
# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
    # update our list of rectangles
    rects.append((x, y, x + tW, y + tH))
# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))

print(f"Total count = {len(pick)}")

print("[INFO] {} matched locations *after* NMS".format(len(pick)))
# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
    # draw the bounding box on the image
    cv.rectangle(src_img, (startX, startY), (endX, endY), (255, 0, 0), 3)

# font
font = cv.FONT_HERSHEY_SIMPLEX

# org
org = (tH_src - 100, tW_src - 100)
# bottom_right = (tH_src-10, tW_src-10)

# fontScale
fontScale = 1

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

cv.putText(
    src_img, f"counts={len(pick)}", org, font, fontScale, color, thickness, cv.LINE_AA
)
# show the output image
cv.imwrite("./imgs/ind_coins100_final.png", src_img)
