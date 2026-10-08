from ultralytics import YOLO
import cv2
import numpy as np

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")
face_model = YOLO("models/yolov11m-face.pt").to("cuda")

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

while cv2.waitKey(1) != ord("x"):
    _, frame = cap.read()
    result = face_model.track(frame, persist=True, verbose=False)
    processed_image = result[0].plot()
    boxes = result[0].boxes.xyxy
    track_ids = result[0].boxes.id
    if track_ids is None:
        continue
    for box, track_id in zip(boxes, track_ids):
        if box.index == 0:
            continue
        left, top, right, bottom = box.int()
        face = frame[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (200, 200))
        label, distance = face_recognizer.predict(face)
        cv2.putText(
            processed_image,
            str(label) + " | " + str(int(distance)) + " | track_id=" + str(int(track_id)),
            (int(left), int(bottom) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.imshow("Image", processed_image)

cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()