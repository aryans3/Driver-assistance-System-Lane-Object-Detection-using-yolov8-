from ultralytics import YOLO
import cv2 as cv
import math
import cvzone
import statistics
import numpy as np

cap = cv.VideoCapture("../testcase/demo.mp4")
alert = cv.imread("../signs/alert.png", cv.IMREAD_UNCHANGED)
warn = cv.imread("../signs/warn.png", cv.IMREAD_UNCHANGED)
cross = cv.imread("../signs/cross.png", cv.IMREAD_UNCHANGED)
stop = cv.imread("../signs/stop.png", cv.IMREAD_UNCHANGED)
signal = cv.imread("../signs/signal.png", cv.IMREAD_UNCHANGED)
go = cv.imread("../signs/go.png", cv.IMREAD_UNCHANGED)
steer = cv.imread("../signs/steer.png", cv.IMREAD_UNCHANGED)

model = YOLO('../yolo-weights/yolov8s.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

lower_valG = np.array([75, 65, 75])
upper_valG = np.array([84, 255, 255])
lower_valR = np.array([0, 70, 100])
upper_valR = np.array([0, 255, 255])

while True:
    success, img = cap.read()
    height, width = img.shape[:2]

    left_x = (width / 2) - (width / 5)
    right_x = (width / 2) + (width / 5)
    results = model(img, stream=True)
    # object
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            if cls == 0:
                if int(math.dist((x1, y1), (x2, y2))) > 220:
                    img = cvzone.overlayPNG(img,cross, [0, 0])
                elif x1>left_x and x1<right_x and x2>left_x and x2<right_x and int(math.dist((x1, y1), (x2, y2))) > 80:
                    img = cvzone.overlayPNG(img, cross, [0, 0])

            if (cls == 1 or cls == 2 or cls == 3 or cls == 5 or cls == 7) and (x1>(width/4) and x1<width and x2>(width/4) and x2<width and y1>height/2 and y2>height/2):
                if int(math.dist((x1, y1), (x2, y2))) > 275 and int(math.dist((x1, y1), (x2, y2))) < 350:
                    img = cvzone.overlayPNG(img, alert, [x1, y1])
                elif int(math.dist((x1, y1), (x2, y2))) > 350:
                    img = cvzone.overlayPNG(img, warn, [x1, y1])

            # cv.putText(img, classNames[cls], (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 1,color, 2)

            if cls == 9 and x1>left_x and x1<width and x2>left_x and x2<width:
                crop_img = img[y1:y2, x1:x2]
                hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)

                maskG = cv.inRange(hsv, lower_valG, upper_valG)
                hasGreen = np.sum(maskG)

                maskR = cv.inRange(hsv, lower_valR, upper_valR)
                hasRed = np.sum(maskR)

                if hasGreen > 0:
                    if int(math.dist((x1, y1), (x2, y2))) > 30:
                        img = cvzone.overlayPNG(img, go, [0, 300])
                elif hasRed > 0:
                    if int(math.dist((x1, y1), (x2, y2))) < 50:
                        img = cvzone.overlayPNG(img, signal, [0, 150])
                    if int(math.dist((x1, y1), (x2, y2))) > 30:
                        img = cvzone.overlayPNG(img, stop, [0, 300])

    # lane
    tl = (625, 500)
    bl = (400, height)
    tr = (775, 500)
    br = (1050, height)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv.warpPerspective(img, matrix, (640, 480))
    hsv_transformed_frame = cv.cvtColor(transformed_frame, cv.COLOR_BGR2HSV)

    lower = np.array([0, 0, 200])
    upper = np.array([255, 150, 255])
    mask = cv.inRange(hsv_transformed_frame, lower, upper)

    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        avglx = []
        avgrx = []
        imgl = mask[y - 40:y, left_base - 50:left_base + 50]
        contours, _ = cv.findContours(imgl, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        imgr = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv.findContours(imgr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40
        avglx.append(left_base + 50)
        avgrx.append(right_base + 50)

    left = int(statistics.mean(avglx))
    right = int(statistics.mean(avgrx))
    (h, w) = steer.shape[:2]
    angle = -((((left+right)/2)-320)/320)*60
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(steer, M, (w, h))
    cv.imshow("Bird's Eye View", transformed_frame)
    cv.imshow("Lane Detection - Sliding Windows", msk)
    img = cvzone.overlayPNG(img, rotated, [int(width / 2), 600])
    cv.imshow("DashCam", img)
    cv.waitKey(1)