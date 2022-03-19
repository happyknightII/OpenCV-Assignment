import cv2
import numpy as np

SCALE_FACTOR = 2

cap = cv2.VideoCapture('video2.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

if not cap.isOpened():
    print("Error opening video stream or file")
    cap.release()
else:
    output = cv2.VideoWriter('output.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             30, (width, height))
    frameNum = 0
    frameTotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    totalCircles = 0
    lastFrame = None
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            gray_blurred = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (2, 2))
            resizedImage = cv2.resize(gray_blurred, (int(width/SCALE_FACTOR), int(height/SCALE_FACTOR)))
            detected_circles = cv2.HoughCircles(resizedImage,
                                                cv2.HOUGH_GRADIENT, 1, 10000, param1=170,
                                                param2=35, minRadius=83, maxRadius=90)
            if detected_circles is not None:
                x, y, r = np.uint16(np.around(detected_circles[0]))[0]
                x *= SCALE_FACTOR
                y *= SCALE_FACTOR
                r *= SCALE_FACTOR
                cv2.circle(frame, (x, y), r, (0, 0, 0), 9)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 6)

                cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
                totalCircles += 1
        else:
            break

        cv2.putText(frame, f"{totalCircles} / {frameTotal}", (10, int(height / 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, f"{totalCircles} / {frameTotal}", (10, int(height / 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        output.write(frame)
        lastFrame = frame

        if cv2.waitKey(10) == ord('q'):
            break
        frameNum += 1

    cap.release()
    cv2.putText(lastFrame, f"Accuracy: {int(totalCircles * 100 / frameTotal)}%", (int(width / 5), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.putText(lastFrame, f"Accuracy: {int(totalCircles * 100 / frameTotal)}%", (int(width / 5), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 7, cv2.LINE_AA)
    cv2.putText(lastFrame, f"Accuracy: {int(totalCircles * 100 / frameTotal)}%", (int(width / 5), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.imshow('Frame', lastFrame)
    output.write(lastFrame)
    output.release()
    cv2.waitKey()
    cv2.destroyAllWindows()
