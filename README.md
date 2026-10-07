# FRCS – Face Recognition CCTV Attendance System

Real-time attendance tracking from a live CCTV (RTSP) feed. The system detects and tracks every person in the frame, finds their face, recognises them against a database of registered employees, and writes morning/afternoon attendance into SQLite — fully automatically, with no check-in device required.

Built for CPU-only deployment on standard office hardware.

## How it works

```
RTSP camera stream
      ↓
YOLO26x               person detection (COCO class 0)
      ↓
TrackTrack            multi-object tracking → stable Track IDs
      ↓
SCRFD-34G (ONNX)      face detection inside each person box
      ↓
InsightFace buffalo_l ArcFace 512-d face embedding
      ↓
L2 normalisation + cosine similarity against registered embeddings
      ↓
Face ↔ Track association + identity memory (survives brief occlusions)
      ↓
Attendance logic (morning / afternoon sessions)
      ↓
SQLite  (database/attendance.db)
```

Key design points:

- **Track-level identity** – recognition runs every few frames per Track ID instead of every frame, so a person keeps their name even when the face is briefly lost. This keeps CPU load manageable.
- **Face-in-box association** – a face is only assigned to a track if its centre lies inside that person's bounding box, preventing identities from jumping between neighbouring people.
- **Recognition margin** – a match must beat the second-best candidate by a margin, which suppresses ambiguous matches between similar-looking people.
- **Session-based attendance** – one row per person per day with separate morning and afternoon detection times and a computed attendance value.

## Repository layout

| File | Purpose |
|------|---------|
| `final.py` | **Main application.** Full pipeline: RTSP capture, detection, tracking, recognition and SQLite attendance. |
| `T1.py` | Performance variant of `final.py`: SCRFD runs in a background thread on every second frame. Same recognition and attendance logic. |
| `register.py` | Registers a new person: reads a folder of photos, extracts ArcFace embeddings and saves a `<name>.npy` to `database/`. |
| `requirements.txt` | Pinned Python dependencies. |


## Setup

### 1. Python environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 2. Model weights (not included in the repo)

| Model | Where to get it | Put it at |
|-------|-----------------|-----------|
| `yolo26x.pt` | Downloaded automatically by Ultralytics on first run, or from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) | project root |
| `scrfd_34g.onnx` | Export from the [InsightFace SCRFD model zoo](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) | `models/scrfd_34g.onnx` |
| `buffalo_l` | Downloaded automatically by InsightFace on first run | `~/.insightface/models/` |

### 3. Camera URL

Set the RTSP URL through an environment variable (never commit credentials):

```bash
# Windows PowerShell
$env:RTSP_URL = "rtsp://user:password@192.168.x.x:554/cam/realmonitor?channel=1&subtype=0"

# Linux / macOS
export RTSP_URL="rtsp://user:password@192.168.x.x:554/cam/realmonitor?channel=1&subtype=0"
```

If the variable is not set, the scripts fall back to a placeholder URL that you must edit.

## Usage

### Register people

Put 3–7 clear, front-facing photos of each person in a folder, then run:

```bash
python register.py
# Enter person's name: alice
# Enter image folder: images/alice
```

This writes `database/alice.npy` (the averaged, L2-normalised embedding). Repeat for each person.

### Run attendance

```bash
python final.py
```

A window shows the live feed with person boxes, Track IDs and recognised names. Attendance is written to `database/attendance.db`:

```sql
CREATE TABLE attendance (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name             TEXT NOT NULL,
    attendance_date         TEXT NOT NULL,
    morning_detected_time   TEXT,
    afternoon_detected_time TEXT,
    attendance_value        REAL NOT NULL DEFAULT 0.0,
    UNIQUE(person_name, attendance_date)
);
```

Press `q` to quit.

## Tuning

All thresholds live at the top of `final.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `PERSON_CONFIDENCE` | 0.15 | Lower = fewer missed people, more false boxes |
| `YOLO_IMAGE_SIZE` | 1280 | Higher = better small-person detection, slower |
| `FACE_CONFIDENCE` | 0.50 | SCRFD face detection threshold |
| `RECOGNITION_THRESHOLD` | 0.45 | Min cosine similarity to accept a match |
| `RECOGNITION_INTERVAL` | 5 | Re-recognise each track every N frames |
| `IDENTITY_MEMORY_FRAMES` | 60 | Frames to keep a name after the face disappears |

## Privacy

This repository contains code only. The `images/` and `database/` folders are empty placeholders: registration photos, face embeddings (`database/*.npy`), the attendance database and any output frames are git-ignored and never committed. Deploy only with the informed consent of the people being recognised and in accordance with local biometric-data regulations.

## Tech stack

Python · OpenCV · Ultralytics YOLO26 · TrackTrack · InsightFace (SCRFD, ArcFace / buffalo_l) · ONNX Runtime · PyTorch · SQLite
