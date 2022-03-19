import cv2
import numpy as np


def area_filter(min_area, input_image):
    # Perform an area filter on the binary blobs:
    components_number, labeled_image, component_stats, _ = \
        cv2.connectedComponentsWithStats(input_image, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remaining_component_labels = [i for i in range(1, components_number) if component_stats[i][4] >= min_area]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filtered_image = np.where(np.isin(labeled_image, remaining_component_labels), 255, 0).astype('uint8')

    return filtered_image


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


for index in range(3):

    img = cv2.imread(f'Test{index}.jpg')
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # Convert to float and divide by 255:
    imgFloat = image_sharp.astype(float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold image:
    _, binaryImage = cv2.threshold(kChannel, 190, 255, cv2.THRESH_BINARY)

    # Filter small blobs:
    binaryImage = area_filter(900, binaryImage)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)
    edges = auto_canny(binaryImage)
    minLineLength = 600
    maxLineGap = 100
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    leftMostLine = ((9999, 9999), (9999, 9999))

    rightMostLine = ((0, 0), (0, 0))

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) * 0.53 < abs(y1 - y2):
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if (leftMostLine[0][0] + leftMostLine[1][0]) > (x1 + x2):
                leftMostLine = [(x1, y1), (x2, y2)]
            if (rightMostLine[0][0] + rightMostLine[1][0]) < (x1 + x2):
                rightMostLine = [(x1, y1), (x2, y2)]

    leftMostLine.sort(key=lambda x: x[1])
    rightMostLine.sort(key=lambda x: x[1])

    cv2.arrowedLine(img, leftMostLine[1], leftMostLine[0], (0, 0, 255), 3)
    cv2.arrowedLine(img, rightMostLine[1], rightMostLine[0], (0, 0, 255), 3)

    xPoints = (int((leftMostLine[0][0] + rightMostLine[0][0]) / 2), int((leftMostLine[1][0] + rightMostLine[1][0]) / 2))

    yPoints = (int((leftMostLine[0][1] + rightMostLine[0][1]) / 2), int((leftMostLine[1][1] + rightMostLine[1][1]) / 2))
    yPoints = [int(x) for x in yPoints]

    cv2.arrowedLine(img, (xPoints[1], yPoints[1]), (xPoints[0], yPoints[0]), (0, 0, 255), 50)
    cv2.imwrite(f'DirectionOfTravel{index}.jpg', img)
