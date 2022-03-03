import cv2
import numpy as np

img = cv2.imread('test_l3.jpg', cv2.IMREAD_COLOR)


pts1 = np.float32([[990, 185], [1855, 135],
                  [600, 845], [1750, 908]])

offset = (3500, 1500)

pts2 = np.float32([offset, [offset[0] + 1152, offset[1]],
                  [offset[0], offset[1] + 1337], [offset[0] + 1152, offset[1] + 1337]])

# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)


result = cv2.warpPerspective(img, matrix, (2000 + offset[0], 2500 + offset[1]))


SCALE_FACTOR = 0.3
gray_blurred = cv2.blur(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), (2, 2))
bestImage = cv2.resize(gray_blurred, (int(result.shape[1] * SCALE_FACTOR), int(result.shape[0] * SCALE_FACTOR)))

detected_circles = cv2.HoughCircles(bestImage,
                                   cv2.HOUGH_GRADIENT, 1, 100000, param1=60,
                                   param2=20, minRadius=int(517 * SCALE_FACTOR), maxRadius=int(518 * SCALE_FACTOR))

c1 = (600, 845)
c2 = (c1[0] + 1150, c1[1] + 63)
c3 = (c1[0] + 1255, c1[1] - 710)
c4 = (c1[0] + 390, c1[1] - 660)
cv2.line(img, c1, c2, (255, 255, 255), 2)
cv2.line(img, c2, c3, (255, 255, 255), 2)
cv2.line(img, c3, c4, (255, 255, 255), 2)
cv2.line(img, c1, c4, (255, 255, 255), 2)

cv2.imshow('Transform quadrangle', img)
cv2.waitKey(0)


circle = (np.uint16(np.around(detected_circles[0])))[0]

a, b, r = int(circle[0] / SCALE_FACTOR), int(circle[1] / SCALE_FACTOR), int(circle[2] / SCALE_FACTOR)

cv2.circle(result, (a, b), r, (0, 0, 0), 12)
cv2.circle(result, (a, b), r, (0, 165, 255), 6)

cv2.circle(result, (a, b), 15, (0, 0, 0), -1)
cv2.circle(result, (a, b), 10, (0, 255, 255), -1)


inv_result = cv2.warpPerspective(result, np.linalg.pinv(matrix), (img.shape[1], img.shape[0]))
cv2.imshow('Output image', inv_result)
cv2.imwrite("Output Film.jpg", inv_result)
cv2.waitKey(0)

cv2.destroyAllWindows()

