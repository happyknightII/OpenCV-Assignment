import cv2

IMAGE_PATH = "watch.jpg"
image = cv2.imread(IMAGE_PATH)

rows, columns, _ = image.shape
try:
    print("running vision")
    for y in range(rows):
        for x in range(columns):
            blue, green, red = image[y, x]
            if 160 < red < 190 and 130 < green < 140 and 60 < blue < 70:
                print(red)
                target = (x, y)
                raise StopIteration
    print("Failed to find circle")

except StopIteration:
    print(target)
    print(image[target[1], target[0]])
    # black dot with white outline
    cv2.circle(image, target, 20, (0, 0, 0), -1)
    cv2.circle(image, target, 20, (255, 255, 255), 2)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    cv2.imshow("Detecting circle edge", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
