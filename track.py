from ultralytics import YOLO
import cv2
import numpy as np
from anti_spoof import is_face_real

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")
face_model = YOLO("models/yolov11m-face.pt").to("cuda")

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

while cv2.waitKey(1) != ord("x"):
    _, frame = cap.read()
    result = face_model.track(frame, persist=True, verbose=False)
    processed_image = frame.copy()
    boxes = result[0].boxes.xyxy
    track_ids = result[0].boxes.id
    if track_ids is None:
        cv2.imshow("Image", frame)
        continue
    for box, track_id in zip(boxes, track_ids):
        if box.index == 0:
            continue
        left, top, right, bottom = map(int, box)

        spoof_label, spoof_confidence = is_face_real(frame, box)
        if not spoof_label == "Real":
            cv2.rectangle(
                processed_image,
                (left, top),
                (right, bottom),
                (0, 0, 255),  # BGR = Green
                2
            )
            cv2.putText(
                processed_image,
                f"{spoof_label} {spoof_confidence * 100:.1f}%",
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow("Image", processed_image)
            continue


        face = frame[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (200, 200))
        label, distance = face_recognizer.predict(face)
        cv2.rectangle(
            processed_image,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )
        cv2.putText(
            processed_image,
            str(label) + " | " + str(int(distance)) + " | track_id=" + str(int(track_id)),
            (left, bottom + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.imshow("Image", processed_image)

cap.release()
cv2.destroyAllWindows()