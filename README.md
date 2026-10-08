# Face Tracking & Recognition

A real-time face tracking and recognition system built with **Python, YOLO, OpenCV, and LBPH**.

The system uses a camera to detect faces, track them across video frames, and recognize detected faces using a trained LBPH face-recognition model.

## Features

- Real-time face detection using YOLO
- Real-time face tracking using the Ultralytics YOLO tracker
- Persistent tracking IDs for detected faces
- Face recognition using OpenCV's LBPH Face Recognizer
- Face image preprocessing before recognition
- GPU acceleration using CUDA
- Displays recognition label, recognition distance, and tracking ID

## Technologies

- Python
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- OpenCV Contrib
- NumPy
- PyTorch
- CUDA

## How It Works

The system processes each camera frame through the following pipeline:

```text
Camera
   ↓
YOLO Face Detection
   ↓
Object Tracking
   ↓
Tracking ID
   ↓
Face Cropping
   ↓
Grayscale Conversion
   ↓
Resize to 200 × 200
   ↓
LBPH Face Recognition
   ↓
Display:
Label | Distance | Track ID
```

For example:

```text
1 | 42 | track_id=1
```

Where:

- `1` → recognized person's LBPH label
- `42` → LBPH recognition distance
- `1` → YOLO tracking ID

The **tracking ID** identifies the same detected face across consecutive frames, while the **LBPH label** identifies who the person is.

## Project Structure

```text
face-tracking/
│
├── track.py
├── face_recognizer.yml
├── models/
│   └── yolov11m-face.pt
│
└── README.md
```

### Files

**`track.py`**

Main application responsible for:

- Capturing camera frames
- Running YOLO detection and tracking
- Extracting face bounding boxes
- Cropping and preprocessing faces
- Running LBPH recognition
- Displaying tracking and recognition information

**`face_recognizer.yml`**

The trained OpenCV LBPH face-recognition model.

**`models/yolov11m-face.pt`**

YOLO face-detection model used for detecting and tracking faces.

## Requirements

Install the required Python packages:

```bash
pip install ultralytics opencv-contrib-python numpy
```

For GPU acceleration, install a compatible PyTorch version with CUDA support.

The YOLO tracker may also require:

```bash
pip install "lap>=0.5.12"
```

## Running the Project

Make sure the following files exist:

```text
face_recognizer.yml
models/yolov11m-face.pt
```

Then run:

```bash
python track.py
```

The application will open the camera and begin detecting, tracking, and recognizing faces.

Press:

```text
X
```

to exit.

## GPU Acceleration

The YOLO model is explicitly moved to the CUDA GPU:

```python
face_model = YOLO("models/yolov11m-face.pt").to("cuda")
```

This allows YOLO detection and tracking to run using the GPU when a compatible CUDA-enabled NVIDIA GPU and PyTorch installation are available.

## Tracking

Tracking is enabled with:

```python
result = face_model.track(
    frame,
    persist=True,
    verbose=False
)
```

`persist=True` tells the tracker to maintain object identities between consecutive frames.

The tracking IDs are retrieved with:

```python
track_ids = result[0].boxes.id
```

For example:

```text
Face A → track_id=1
Face B → track_id=2
```

The tracking ID is different from the face-recognition label.

## Face Recognition

The project uses OpenCV's LBPH Face Recognizer:

```python
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
```

The trained model is loaded from:

```python
face_recognizer.read("face_recognizer.yml")
```

Each detected face is:

1. Cropped from the camera frame
2. Converted to grayscale
3. Resized to `200 × 200`
4. Passed to the LBPH recognizer

```python
face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
face = cv2.resize(face, (200, 200))

label, distance = face_recognizer.predict(face)
```

## Recognition Distance

LBPH returns both a label and a distance:

```python
label, distance = face_recognizer.predict(face)
```

The label represents the predicted identity, while the distance represents how closely the detected face matches the trained data.

Generally, a **lower distance indicates a closer match**, although the appropriate threshold depends on the trained model and dataset.

## Current Limitations

- Recognition is performed on detected faces for each processed frame.
- The system currently uses a single camera.
- The camera index is hardcoded to `0`.
- Recognition labels are numeric and require a mapping to actual names.
- The YOLO face model and LBPH model must already be available locally.
- CUDA is currently assumed for YOLO inference.

## Future Improvements

Possible future improvements include:

- Map LBPH labels to people's names
- Improve recognition accuracy
- Add recognition confidence/threshold handling
- Reduce unnecessary recognition calls using tracking
- Re-identify faces after temporary disappearance
- Store recognized people and timestamps
- Add attendance functionality
- Add a backend API
- Add a database for recognized users
- Add a web or desktop interface
- Improve multi-person tracking
- Experiment with ByteTrack, BoT-SORT, or other tracking algorithms

## License

This project is for learning and experimentation with computer vision, face recognition, and object tracking.