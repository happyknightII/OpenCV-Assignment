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


def fill(src, direction, resolution=20, thickness=None):
    if not thickness:
        thickness = resolution
    if direction == 0:
        for x in range(border_size * 10, src.shape[1] - border_size * 10, resolution):
            startPoint = None
            endPoint = None
            for y in range(border_size * 10, src.shape[0] - border_size * 10):
                if src[y][x] == 0 and src[y - 1][x] == 255:
                    startPoint = (x, y)
                    break
            if startPoint:
                for y in reversed(range(border_size * 10, src.shape[0] - border_size * 10)):
                    if src[y][x] == 0 and src[y + 1][x] == 255:
                        endPoint = (x, y)
                        break
                if endPoint:
                    cv2.line(src, startPoint, endPoint, 255, thickness=thickness)
    else:

        for y in range(border_size * 10, src.shape[0] - border_size * 10, resolution):
            startPoint = None
            endPoint = None
            for x in range(border_size * 10, src.shape[1] - border_size * 10):
                if src[y][x] == 0 and src[y][x - 1] == 255:
                    startPoint = (x, y)
                    break
            if startPoint:
                for x in reversed(range(border_size * 10, src.shape[1] - border_size * 10)):
                    if src[y][x] == 0 and src[y][x + 1] == 255:
                        endPoint = (x, y)
                        break

                if endPoint:
                    cv2.line(src, startPoint, endPoint, 255, thickness=thickness)


def morph(src, kernelSize=3, opIterations=3, ):
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                            cv2.BORDER_REFLECT101)


def clean(src, kernelSize=3, opIterations=3, ):
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    return cv2.morphologyEx(src, cv2.MORPH_OPEN, morphKernel, None, None, opIterations,
                            cv2.BORDER_REFLECT101)


def skeletonize(src, kernelSize=3):
    src = src.copy()  # don't clobber original
    skeleton = src.copy()

    skeleton[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernelSize, kernelSize))

    while True:

        eroded = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(src, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        src[:, :] = eroded[:, :]
        if cv2.countNonZero(src) == 0:
            break

    return skeleton


def smooth(src, kernelSize=61, threshold=10):
    src = cv2.GaussianBlur(src, (kernelSize, kernelSize), 0)
    _, src = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)
    return src


for index in range(3):
    print(f"processing Test{index}.jpg")
    img = cv2.imread(f'Test{index}.jpg')
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    border_size = int(img.shape[0] * 0.015)
    cropped_img = cv2.copyMakeBorder(
        img[border_size: img.shape[0] - border_size, border_size: img.shape[1] - border_size],
        border_size * 10,
        border_size * 10,
        border_size * 10,
        border_size * 10,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    image_sharp = cv2.filter2D(src=cropped_img, ddepth=-1, kernel=kernel)
    # Convert to float and divide by 255:
    imgFloat = image_sharp.astype(float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold image:
    _, binaryImage = cv2.threshold(kChannel, 205, 255, cv2.THRESH_BINARY)

    # Filter small blobs:
    binaryImage = area_filter(900, binaryImage)

    # Perform closing:
    binaryImage = morph(binaryImage, 3, 3)
    print("finished mask")
    fill(binaryImage, 1, resolution=50, thickness=1)
    print("finished vertical fill")
    fill(binaryImage, 0, resolution=50, thickness=1)
    print("finished horizontal fill")
    fill(binaryImage, 1, resolution=5, thickness=8)
    print("finished vertical fill")
    ogShape = binaryImage.shape
    binaryImage = cv2.resize(binaryImage, (int(binaryImage.shape[1] * 0.2), int(binaryImage.shape[0] * 0.2)))
    binaryImage = smooth(binaryImage)
    binaryImage = skeletonize(binaryImage)
    print("finished skeletonize")
    # Trying to remove Y-tail
    binaryImage = area_filter(2, binaryImage)
    binaryImage = smooth(binaryImage, 9)
    binaryImage = area_filter(500, binaryImage)
    binaryImage = smooth(binaryImage, 9)
    print("finished processing")

    pointY = 0
    sumsY = binaryImage.sum(axis=1)
    while True:
        pointY += 1
        if sumsY[pointY] != 0:
            break

    pointX = 0
    right = False
    while True:
        pointX += 1
        if binaryImage[pointY + 10, pointX] != 0:
            break
        elif binaryImage[pointY + 10, binaryImage.shape[1] - pointX] != 0:
            pointX = binaryImage.shape[1] - pointX
            right = True
            break

    if right:
        offset = 1
    else:
        offset = -1

    cv2.arrowedLine(binaryImage, (pointX - 10 * offset, pointY), (pointX + 70 * offset, pointY), (255, 255, 255), 10)
    binaryImage = binaryImage[
                  int(border_size * 11 * 0.2): int((ogShape[0] - border_size * 11) * 0.2),
                  int(border_size * 11 * 0.2): int((ogShape[1] - border_size * 11) * 0.2)]
    binaryImage = cv2.resize(binaryImage, (int(img.shape[1]), int(img.shape[0])))
    backtorgb = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(img, 0.7, backtorgb, 0.5, 0)
    dst = cv2.resize(dst, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
    cv2.imshow("Mask", dst)
    cv2.waitKey(0)
    cv2.imwrite(f"IdentifiedTravel{index}.jpg", dst)
    cv2.destroyAllWindows()
