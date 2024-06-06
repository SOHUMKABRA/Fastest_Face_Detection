import cv2
from utils import FaceDetector

fd = FaceDetector()
cap = cv2.VideoCapture(0)  # capture from camera
# cap = cv2.VideoCapture('rtsp://admin:Iam_1234@192.168.1.180:554/')  # capture from camera
#ret, frame = cap.read()
#frame = cv2.resize(frame, (500, 500), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
#cv2.imshow("Camera Preview", frame)
threshold = 0.5

sum = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("no img")
        break
    boxes, probs = fd.maininfer(orig_image)
    cv2.imshow("Camera Preview", orig_image)

    print(boxes, probs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
