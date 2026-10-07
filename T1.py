from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import sqlite3
import threading

from insightface.app import FaceAnalysis


# ============================================================
# FRCS - FINAL REAL-TIME CCTV FACE RECOGNITION ATTENDANCE
# ============================================================
#
# PIPELINE:
#
# RTSP
#   ↓
# YOLO26x
#   ↓
# TrackTrack
#   ↓
# SCRFD-34G
#   ↓
# Buffalo_l
#   ↓
# ArcFace embedding
#   ↓
# L2 normalization
#   ↓
# Cosine similarity
#   ↓
# Face ↔ Track association
#   ↓
# Track identity memory
#   ↓
# Attendance
#   ↓
# SQLite
#
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0"
)


WINDOW_NAME = "FRCS - CCTV Attendance"


# ============================================================
# YOLO / P1-P2 SETTINGS
# DO NOT CHANGE
# ============================================================

MODEL_NAME = "yolo26x.pt"

PERSON_CLASS_ID = 0

PERSON_CONFIDENCE = 0.15

PERSON_IOU = 0.35

YOLO_IMAGE_SIZE = 1280

MAX_DETECTIONS = 100

TRACKER_CONFIG = "tracktrack.yaml"


# ============================================================
# SCRFD / P3 SETTINGS
# DO NOT CHANGE
# ============================================================

SCRFD_MODEL_PATH = (
    os.path.join("models", "scrfd_34g.onnx")
)

SCRFD_INPUT_SIZE = (640, 640)

FACE_CONFIDENCE = 0.50

NMS_THRESHOLD = 0.40

# Run SCRFD on every second CCTV frame.
# The last completed SCRFD result is reused between detections.
# This reduces CPU contention without changing the SCRFD model
# or any of its detection settings.
SCRFD_FRAME_INTERVAL = 2


# ============================================================
# BUFFALO_L / P4 SETTINGS
# ============================================================

FACE_CROP_PADDING = 0.40

CROP_DETECTION_SIZE = (256, 256)

CROP_DETECTION_THRESHOLD = 0.30


# ============================================================
# RECOGNITION SETTINGS
# ============================================================

RECOGNITION_THRESHOLD = 0.45

# Recognition is expensive on CPU.
# A Track ID is re-recognized periodically instead of
# performing recognition on every frame.
RECOGNITION_INTERVAL = 5


# ============================================================
# FACE ↔ TRACK ASSOCIATION
# ============================================================

# Face center must normally be inside the person box.
# This prevents a face belonging to a nearby person from
# being assigned to the wrong Track ID.

FACE_CENTER_REQUIRED = True


# ============================================================
# IDENTITY STABILIZATION
# ============================================================

# If a known person's face disappears temporarily, retain
# their identity for this many frames while TrackTrack still
# maintains the same person Track ID.

IDENTITY_MEMORY_FRAMES = 60


# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = "database"

ATTENDANCE_DB = os.path.join(
    DATABASE_PATH,
    "attendance.db"
)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

if not RTSP_URL:
    print("ERROR: RTSP_URL is empty.")
    print("Set your CCTV RTSP URL in cctv.py.")
    raise SystemExit


# ============================================================
# 3. LOAD YOLO26x
# ============================================================

print("Loading YOLO26x...")

yolo_model = YOLO(
    MODEL_NAME
)


# ============================================================
# 4. LOAD SCRFD-34G
# ============================================================

print("Loading SCRFD-34G...")

scrfd_session = ort.InferenceSession(
    SCRFD_MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

scrfd_input_name = (
    scrfd_session
    .get_inputs()[0]
    .name
)


# ============================================================
# 5. LOAD BUFFALO_L
# ============================================================

print("Loading Buffalo_l...")

face_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=[
        "detection",
        "recognition"
    ],
    providers=[
        "CPUExecutionProvider"
    ]
)

face_app.prepare(
    ctx_id=0,
    det_size=CROP_DETECTION_SIZE,
    det_thresh=CROP_DETECTION_THRESHOLD
)


# ============================================================
# 6. LOAD REGISTERED FACE DATABASE
# ============================================================

print("Loading face database...")

gallery = {}


if not os.path.isdir(DATABASE_PATH):

    print(
        "ERROR: database folder not found."
    )

    raise SystemExit


for filename in os.listdir(
    DATABASE_PATH
):

    if not filename.lower().endswith(
        ".npy"
    ):
        continue


    file_path = os.path.join(
        DATABASE_PATH,
        filename
    )


    try:

        embeddings = np.load(
            file_path
        )


        if embeddings.ndim == 1:

            embeddings = embeddings[
                np.newaxis,
                :
            ]


        normalized_embeddings = []


        for embedding in embeddings:

            embedding = (
                embedding.astype(
                    np.float32
                )
            )


            norm = np.linalg.norm(
                embedding
            )


            if norm == 0:

                continue


            embedding = (
                embedding / norm
            )


            normalized_embeddings.append(
                embedding
            )


        if len(
            normalized_embeddings
        ) == 0:

            continue


        person_name = os.path.splitext(
            filename
        )[0]


        person_name = (
            person_name
            .replace("_", " ")
            .title()
        )


        gallery[
            person_name
        ] = np.asarray(
            normalized_embeddings,
            dtype=np.float32
        )


    except Exception as e:

        print(
            f"WARNING: Could not load "
            f"{filename}: {e}"
        )


if len(gallery) == 0:

    print(
        "ERROR: No registered face "
        "embeddings found."
    )

    raise SystemExit


print(
    f"Loaded {len(gallery)} registered person(s)."
)


# ============================================================
# 7. ATTENDANCE DATABASE
# ============================================================

os.makedirs(
    DATABASE_PATH,
    exist_ok=True
)


db = sqlite3.connect(
    ATTENDANCE_DB,
    timeout=10,
    check_same_thread=False
)


db.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_name TEXT NOT NULL,
        attendance_date TEXT NOT NULL,
        morning_detected_time TEXT,
        afternoon_detected_time TEXT,
        attendance_value REAL NOT NULL DEFAULT 0.0,
        UNIQUE(person_name, attendance_date)
    )
    """
)


db.commit()


# ============================================================
# 8. SCRFD PRECOMPUTED ANCHOR CENTERS
# ============================================================

SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHOR_CENTERS = {}

for stride in SCRFD_STRIDES:

    feature_height = SCRFD_INPUT_SIZE[1] // stride
    feature_width = SCRFD_INPUT_SIZE[0] // stride

    anchor_centers = np.stack(
        np.mgrid[
            :feature_height,
            :feature_width
        ][::-1],
        axis=-1
    ).astype(np.float32)

    anchor_centers *= stride
    anchor_centers = anchor_centers.reshape(-1, 2)

    SCRFD_ANCHOR_CENTERS[stride] = np.stack(
        [
            anchor_centers,
            anchor_centers
        ],
        axis=1
    ).reshape(-1, 2)


# ============================================================
# 9. SCRFD BBOX DECODER
# ============================================================

def distance2bbox(
    points,
    distance
):

    x1 = (
        points[:, 0]
        -
        distance[:, 0]
    )

    y1 = (
        points[:, 1]
        -
        distance[:, 1]
    )

    x2 = (
        points[:, 0]
        +
        distance[:, 2]
    )

    y2 = (
        points[:, 1]
        +
        distance[:, 3]
    )


    return np.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        axis=-1
    )


# ============================================================
# 9. SCRFD NMS
# ============================================================

def nms(
    detections,
    threshold
):

    if len(detections) == 0:

        return []


    x1 = detections[:, 0]
    y1 = detections[:, 1]

    x2 = detections[:, 2]
    y2 = detections[:, 3]

    scores = detections[:, 4]


    areas = (
        (x2 - x1 + 1)
        *
        (y2 - y1 + 1)
    )


    order = scores.argsort()[::-1]

    keep = []


    while order.size > 0:

        i = order[0]

        keep.append(i)


        xx1 = np.maximum(
            x1[i],
            x1[order[1:]]
        )

        yy1 = np.maximum(
            y1[i],
            y1[order[1:]]
        )

        xx2 = np.minimum(
            x2[i],
            x2[order[1:]]
        )

        yy2 = np.minimum(
            y2[i],
            y2[order[1:]]
        )


        width = np.maximum(
            0.0,
            xx2 - xx1 + 1
        )

        height = np.maximum(
            0.0,
            yy2 - yy1 + 1
        )


        intersection = (
            width * height
        )


        overlap = (
            intersection
            /
            (
                areas[i]
                +
                areas[order[1:]]
                -
                intersection
            )
        )


        remaining = np.where(
            overlap <= threshold
        )[0]


        order = order[
            remaining + 1
        ]


    return keep


# ============================================================
# 10. SCRFD FACE DETECTION
# ============================================================

def detect_faces(frame):

    original_height, original_width = (
        frame.shape[:2]
    )


    image_ratio = (
        float(original_height)
        /
        float(original_width)
    )


    model_ratio = (
        float(SCRFD_INPUT_SIZE[1])
        /
        float(SCRFD_INPUT_SIZE[0])
    )


    if image_ratio > model_ratio:

        new_height = (
            SCRFD_INPUT_SIZE[1]
        )

        new_width = int(
            new_height
            /
            image_ratio
        )

    else:

        new_width = (
            SCRFD_INPUT_SIZE[0]
        )

        new_height = int(
            new_width
            *
            image_ratio
        )


    scale = (
        float(new_height)
        /
        float(original_height)
    )


    resized = cv2.resize(
        frame,
        (
            new_width,
            new_height
        )
    )


    padded = np.zeros(
        (
            SCRFD_INPUT_SIZE[1],
            SCRFD_INPUT_SIZE[0],
            3
        ),
        dtype=np.uint8
    )


    padded[
        :new_height,
        :new_width
    ] = resized


    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 128.0,
        size=SCRFD_INPUT_SIZE,
        mean=(
            127.5,
            127.5,
            127.5
        ),
        swapRB=True
    )


    outputs = scrfd_session.run(
        None,
        {
            scrfd_input_name: blob
        }
    )


    all_boxes = []

    all_scores = []


    for level in range(3):

        score_output = outputs[
            level
        ]

        bbox_output = outputs[
            level + 3
        ]


        scores = (
            score_output[0]
            .reshape(-1)
        )


        bbox_predictions = (
            bbox_output[0]
            .reshape(-1, 4)
        )


        stride = SCRFD_STRIDES[level]

        anchor_centers = SCRFD_ANCHOR_CENTERS[stride]


        bbox_predictions = (
            bbox_predictions
            *
            stride
        )


        positive_indices = np.where(
            scores >= FACE_CONFIDENCE
        )[0]


        if len(
            positive_indices
        ) == 0:

            continue


        decoded_boxes = distance2bbox(
            anchor_centers,
            bbox_predictions
        )


        selected_boxes = (
            decoded_boxes[
                positive_indices
            ]
        )


        selected_scores = (
            scores[
                positive_indices
            ]
        )


        selected_boxes = (
            selected_boxes
            /
            scale
        )


        all_boxes.append(
            selected_boxes
        )


        all_scores.append(
            selected_scores
        )


    if len(all_boxes) == 0:

        return []


    all_boxes = np.vstack(
        all_boxes
    )


    all_scores = np.concatenate(
        all_scores
    )


    all_boxes[:, 0] = np.clip(
        all_boxes[:, 0],
        0,
        original_width - 1
    )

    all_boxes[:, 1] = np.clip(
        all_boxes[:, 1],
        0,
        original_height - 1
    )

    all_boxes[:, 2] = np.clip(
        all_boxes[:, 2],
        0,
        original_width - 1
    )

    all_boxes[:, 3] = np.clip(
        all_boxes[:, 3],
        0,
        original_height - 1
    )


    detections = np.hstack(
        (
            all_boxes,
            all_scores.reshape(-1, 1)
        )
    ).astype(
        np.float32
    )


    keep_indices = nms(
        detections,
        NMS_THRESHOLD
    )


    return detections[
        keep_indices
    ]


# ============================================================
# 11. BUFFALO_L EMBEDDING EXTRACTION
# ============================================================

def get_embedding(
    frame,
    bbox
):

    x1, y1, x2, y2 = bbox


    box_width = x2 - x1

    box_height = y2 - y1


    if (
        box_width <= 0
        or
        box_height <= 0
    ):

        return None


    pad_x = int(
        box_width
        *
        FACE_CROP_PADDING
    )


    pad_y = int(
        box_height
        *
        FACE_CROP_PADDING
    )


    frame_height, frame_width = (
        frame.shape[:2]
    )


    crop_x1 = max(
        0,
        x1 - pad_x
    )


    crop_y1 = max(
        0,
        y1 - pad_y
    )


    crop_x2 = min(
        frame_width,
        x2 + pad_x
    )


    crop_y2 = min(
        frame_height,
        y2 + pad_y
    )


    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    if crop.size == 0:

        return None


    faces = face_app.get(
        crop
    )


    if len(faces) == 0:

        return None


    crop_height, crop_width = (
        crop.shape[:2]
    )


    crop_center_x = (
        crop_width / 2.0
    )


    crop_center_y = (
        crop_height / 2.0
    )


    best_face = None

    best_distance = None


    for face in faces:

        fx1, fy1, fx2, fy2 = (
            face.bbox
        )


        face_center_x = (
            fx1 + fx2
        ) / 2.0


        face_center_y = (
            fy1 + fy2
        ) / 2.0


        distance = (
            (face_center_x - crop_center_x) ** 2
            +
            (face_center_y - crop_center_y) ** 2
        )


        if (
            best_distance is None
            or
            distance < best_distance
        ):

            best_distance = distance

            best_face = face


    if best_face is None:

        return None


    embedding = (
        best_face.embedding
        .astype(np.float32)
    )


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        return None


    return (
        embedding / norm
    )


# ============================================================
# 12. FACE RECOGNITION
# ============================================================

def recognize_face(
    embedding
):

    if embedding is None:

        return (
            "UNKNOWN",
            0.0
        )


    best_person = "UNKNOWN"

    best_similarity = -1.0


    for (
        person_name,
        person_embeddings
    ) in gallery.items():

        similarities = np.dot(
            person_embeddings,
            embedding
        )


        person_best = float(
            np.max(
                similarities
            )
        )


        if (
            person_best
            >
            best_similarity
        ):

            best_similarity = (
                person_best
            )

            best_person = (
                person_name
            )


    if (
        best_similarity
        <
        RECOGNITION_THRESHOLD
    ):

        return (
            "UNKNOWN",
            best_similarity
        )


    return (
        best_person,
        best_similarity
    )


# ============================================================
# 13. IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0,
        ix2 - ix1
    )


    ih = max(
        0,
        iy2 - iy1
    )


    intersection = (
        iw * ih
    )


    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )


    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return (
        intersection
        /
        union
    )


# ============================================================
# 14. FACE CENTER
# ============================================================

def get_box_center(
    bbox
):

    x1, y1, x2, y2 = bbox


    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# 15. CHECK WHETHER FACE BELONGS TO PERSON
# ============================================================

def face_belongs_to_person(
    face_bbox,
    person_bbox
):

    fx, fy = get_box_center(
        face_bbox
    )


    px1, py1, px2, py2 = (
        person_bbox
    )


    if (
        fx >= px1
        and
        fx <= px2
        and
        fy >= py1
        and
        fy <= py2
    ):

        return True


    return False


# ============================================================
# 16. ASSOCIATE FACES WITH TRACKS
# ============================================================

def associate_faces_to_tracks(
    face_detections,
    tracks
):

    associations = {}


    for track in tracks:

        track_id = track["track_id"]

        person_bbox = track["bbox"]


        best_face_index = None

        best_score = -1.0


        for face_index, face in enumerate(
            face_detections
        ):

            face_bbox = face["bbox"]


            if FACE_CENTER_REQUIRED:

                if not face_belongs_to_person(
                    face_bbox,
                    person_bbox
                ):

                    continue


            iou = calculate_iou(
                face_bbox,
                person_bbox
            )


            fx, fy = get_box_center(
                face_bbox
            )


            px, py = get_box_center(
                person_bbox
            )


            person_width = (
                person_bbox[2]
                -
                person_bbox[0]
            )


            person_height = (
                person_bbox[3]
                -
                person_bbox[1]
            )


            if (
                person_width <= 0
                or
                person_height <= 0
            ):

                continue


            normalized_distance = (
                (
                    (fx - px) / person_width
                ) ** 2
                +
                (
                    (fy - py) / person_height
                ) ** 2
            )


            center_score = (
                1.0
                /
                (
                    1.0
                    +
                    normalized_distance
                )
            )


            score = (
                iou * 0.5
                +
                center_score * 0.5
            )


            if score > best_score:

                best_score = score

                best_face_index = (
                    face_index
                )


        if best_face_index is not None:

            associations[
                track_id
            ] = best_face_index


    return associations


# ============================================================
# 17. TRACK ID IDENTITY MEMORY
# ============================================================
#
# Structure:
#
# identity_memory[track_id] = {
#     "name": "Nishanth",
#     "similarity": 0.57,
#     "last_face_frame": 100,
#     "last_recognition_frame": 100
# }
#
# ============================================================

identity_memory = {}


# ============================================================
# 18. ATTENDANCE FUNCTIONS
# ============================================================

def get_today():

    return time.strftime(
        "%Y-%m-%d"
    )


def get_current_time():

    return time.strftime(
        "%H:%M:%S"
    )


def get_attendance_record(
    person_name,
    attendance_date
):

    cursor = db.execute(
        """
        SELECT
            id,
            person_name,
            attendance_date,
            morning_detected_time,
            afternoon_detected_time,
            attendance_value
        FROM attendance
        WHERE person_name = ?
        AND attendance_date = ?
        """,
        (
            person_name,
            attendance_date
        )
    )


    return cursor.fetchone()


def record_attendance(
    person_name
):

    if (
        not person_name
        or
        person_name == "UNKNOWN"
    ):

        return


    attendance_date = get_today()

    current_time = get_current_time()


    current_hour = int(
        time.strftime("%H")
    )


    current_minute = int(
        time.strftime("%M")
    )


    is_afternoon = (
        current_hour > 15
        or
        (
            current_hour == 15
            and
            current_minute >= 0
        )
    )


    record = get_attendance_record(
        person_name,
        attendance_date
    )


    # ========================================================
    # NO RECORD YET
    # ========================================================

    if record is None:

        if not is_afternoon:

            db.execute(
                """
                INSERT INTO attendance (
                    person_name,
                    attendance_date,
                    morning_detected_time,
                    afternoon_detected_time,
                    attendance_value
                )
                VALUES (?, ?, ?, NULL, ?)
                """,
                (
                    person_name,
                    attendance_date,
                    current_time,
                    0.5
                )
            )


            db.commit()


            print(
                f"{person_name} - "
                f"{attendance_date} - "
                f"{current_time} - "
                f"0.5 - HALF DAY"
            )


        else:

            db.execute(
                """
                INSERT INTO attendance (
                    person_name,
                    attendance_date,
                    morning_detected_time,
                    afternoon_detected_time,
                    attendance_value
                )
                VALUES (?, ?, NULL, ?, ?)
                """,
                (
                    person_name,
                    attendance_date,
                    current_time,
                    0.5
                )
            )


            db.commit()


            print(
                f"{person_name} - "
                f"{attendance_date} - "
                f"{current_time} - "
                f"0.5 - HALF DAY"
            )


        return


    # ========================================================
    # RECORD ALREADY EXISTS
    # ========================================================

    (
        record_id,
        stored_name,
        stored_date,
        morning_time,
        afternoon_time,
        attendance_value
    ) = record


    # ========================================================
    # BEFORE 3 PM
    # ========================================================

    if not is_afternoon:

        # Morning already recorded.
        # Do nothing.

        if morning_time is not None:

            return


        db.execute(
            """
            UPDATE attendance
            SET
                morning_detected_time = ?,
                attendance_value = ?
            WHERE id = ?
            """,
            (
                current_time,
                0.5,
                record_id
            )
        )


        db.commit()


        print(
            f"{person_name} - "
            f"{attendance_date} - "
            f"{current_time} - "
            f"0.5 - HALF DAY"
        )


        return


    # ========================================================
    # AT / AFTER 3 PM
    # ========================================================

    # Afternoon already recorded.
    # Do nothing.

    if afternoon_time is not None:

        return


    # --------------------------------------------------------
    # Morning exists
    # --------------------------------------------------------

    if morning_time is not None:

        db.execute(
            """
            UPDATE attendance
            SET
                afternoon_detected_time = ?,
                attendance_value = ?
            WHERE id = ?
            """,
            (
                current_time,
                1.0,
                record_id
            )
        )


        db.commit()


        print(
            f"{person_name} - "
            f"{attendance_date} - "
            f"{morning_time} - "
            f"0.5 - HALF DAY"
        )


        print(
            f"{person_name} - "
            f"{attendance_date} - "
            f"{current_time} - "
            f"1 - FULL DAY"
        )


        return


    # --------------------------------------------------------
    # No morning exists
    # --------------------------------------------------------

    db.execute(
        """
        UPDATE attendance
        SET
            afternoon_detected_time = ?,
            attendance_value = ?
        WHERE id = ?
        """,
        (
            current_time,
            0.5,
            record_id
        )
    )


    db.commit()


    print(
        f"{person_name} - "
        f"{attendance_date} - "
        f"{current_time} - "
        f"0.5 - HALF DAY"
    )


# ============================================================
# 19. ASYNCHRONOUS SCRFD WORKER
# ============================================================

stop_event = threading.Event()
face_request_event = threading.Event()
face_lock = threading.Lock()

latest_face_frame = None
latest_face_detections = []
latest_face_frame_number = -1
face_request_frame_number = -1


def scrfd_worker():

    global latest_face_frame
    global latest_face_detections
    global latest_face_frame_number
    global face_request_frame_number

    while not stop_event.is_set():

        face_request_event.wait(0.1)

        if stop_event.is_set():
            break

        if not face_request_event.is_set():
            continue

        with face_lock:
            worker_frame = latest_face_frame
            worker_frame_number = face_request_frame_number
            face_request_event.clear()

        if worker_frame is None:
            continue

        raw_detections = detect_faces(worker_frame)

        detections = []

        for detection in raw_detections:

            x1, y1, x2, y2, confidence = detection

            detections.append(
                {
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": float(
                        confidence
                    )
                }
            )

        with face_lock:

            # Keep the last valid face result for one missed SCRFD
            # cycle. This avoids unnecessary face flicker from a
            # single missed detection while keeping the stream path
            # completely non-blocking.
            if detections:
                latest_face_detections = detections
                latest_face_frame_number = worker_frame_number

            elif latest_face_detections:
                latest_face_frame_number = worker_frame_number


scrfd_thread = threading.Thread(
    target=scrfd_worker,
    daemon=True
)

scrfd_thread.start()


# ============================================================
# 20. OPEN CCTV
# ============================================================

cap = cv2.VideoCapture(
    RTSP_URL
)


cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    WINDOW_NAME,
    1400,
    800
)


# ============================================================
# 21. MAIN LOOP
# ============================================================

frame_number = 0


while True:

    # ========================================================
    # READ CCTV
    # ========================================================

    ret, frame = cap.read()


    if (
        not ret
        or
        frame is None
    ):

        cap.release()

        time.sleep(2)


        cap = cv2.VideoCapture(
            RTSP_URL
        )


        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        continue


    frame_number += 1


    original_frame = frame


    # ========================================================
    # SUBMIT NEWEST FRAME TO SCRFD THREAD
    # ========================================================
    #
    # Only the newest frame is kept. If SCRFD is still working,
    # older pending frames are replaced instead of queued.
    # This prevents SCRFD from blocking the CCTV display and
    # prevents latency from continuously growing.
    #
    # ========================================================

    if frame_number % SCRFD_FRAME_INTERVAL == 0:

        with face_lock:
            latest_face_frame = original_frame.copy()
            face_request_frame_number = frame_number
            face_request_event.set()


    # ========================================================
    # P1 + P2
    #
    # YOLO26x + TrackTrack
    # ========================================================

    results = yolo_model.track(
        source=original_frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=PERSON_CONFIDENCE,
        iou=PERSON_IOU,
        classes=[
            PERSON_CLASS_ID
        ],
        max_det=MAX_DETECTIONS,
        tracker=TRACKER_CONFIG,
        persist=True,
        verbose=False
    )


    result = results[0]


    # ========================================================
    # CREATE TRACK LIST
    # ========================================================

    tracks = []


    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )


            if box.id is None:

                continue


            track_id = int(
                box.id[0]
                .cpu()
                .item()
            )


            tracks.append(
                {
                    "track_id": track_id,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "confidence": confidence
                }
            )


    # ========================================================
    # GET LATEST COMPLETED SCRFD RESULT
    # ========================================================

    with face_lock:
        face_detections = list(
            latest_face_detections
        )
        face_frame_number = latest_face_frame_number


    # ========================================================
    # FACE ↔ TRACK ASSOCIATION
    # ========================================================

    associations = (
        associate_faces_to_tracks(
            face_detections,
            tracks
        )
    )


    # ========================================================
    # TRACK IDS THAT ARE CURRENTLY VISIBLE
    # ========================================================

    visible_track_ids = set()


    for track in tracks:

        visible_track_ids.add(
            track["track_id"]
        )


    # ========================================================
    # PROCESS EACH TRACK
    # ========================================================

    for track in tracks:

        track_id = track[
            "track_id"
        ]


        person_bbox = track[
            "bbox"
        ]


        identity = identity_memory.get(
            track_id
        )


        # ====================================================
        # DOES THIS TRACK HAVE A FACE?
        # ====================================================

        face_index = associations.get(
            track_id
        )


        if face_index is not None:

            face_info = face_detections[
                face_index
            ]


            face_bbox = face_info[
                "bbox"
            ]


            # =================================================
            # CHECK WHETHER RECOGNITION IS DUE
            # =================================================

            should_recognize = False


            if identity is None:

                should_recognize = True


            else:

                frames_since_recognition = (
                    frame_number
                    -
                    identity[
                        "last_recognition_frame"
                    ]
                )


                if (
                    frames_since_recognition
                    >=
                    RECOGNITION_INTERVAL
                ):

                    should_recognize = True


            # =================================================
            # RECOGNIZE FACE
            # =================================================

            if should_recognize:

                embedding = get_embedding(
                    original_frame,
                    face_bbox
                )


                name, similarity = (
                    recognize_face(
                        embedding
                    )
                )


                # ---------------------------------------------
                # IMPORTANT IDENTITY RULE
                #
                # If recognition returns UNKNOWN, do not
                # immediately destroy an already established
                # identity.
                # ---------------------------------------------

                if (
                    name == "UNKNOWN"
                    and
                    identity is not None
                    and
                    identity["name"] != "UNKNOWN"
                ):

                    identity[
                        "last_face_frame"
                    ] = frame_number

                    identity[
                        "last_recognition_frame"
                    ] = frame_number

                else:

                    identity = {
                        "name": name,
                        "similarity": similarity,
                        "last_face_frame": frame_number,
                        "last_recognition_frame": frame_number
                    }


                identity_memory[
                    track_id
                ] = identity


            else:

                identity[
                    "last_face_frame"
                ] = frame_number


            # =================================================
            # ATTENDANCE
            # =================================================

            if (
                identity is not None
                and
                identity["name"] != "UNKNOWN"
            ):

                record_attendance(
                    identity["name"]
                )


        else:

            # =================================================
            # NO FACE CURRENTLY VISIBLE
            # =================================================

            if identity is not None:

                frames_since_face = (
                    frame_number
                    -
                    identity[
                        "last_face_frame"
                    ]
                )


                if (
                    frames_since_face
                    >
                    IDENTITY_MEMORY_FRAMES
                ):

                    identity_memory.pop(
                        track_id,
                        None
                    )


    # ========================================================
    # REMOVE OLD TRACK IDENTITIES
    # ========================================================

    tracks_to_remove = []


    for track_id, identity in (
        identity_memory.items()
    ):

        if track_id in visible_track_ids:

            continue


        last_seen_frame = max(
            identity.get(
                "last_face_frame",
                frame_number
            ),
            identity.get(
                "last_recognition_frame",
                frame_number
            )
        )


        if (
            frame_number
            -
            last_seen_frame
            >
            IDENTITY_MEMORY_FRAMES
        ):

            tracks_to_remove.append(
                track_id
            )


    for track_id in tracks_to_remove:

        identity_memory.pop(
            track_id,
            None
        )


    # ========================================================
    # DISPLAY COPY
    # ========================================================

    display_frame = (
        original_frame.copy()
    )


    # ========================================================
    # DRAW PERSON TRACKS + IDENTITY
    # ========================================================

    for track in tracks:

        track_id = track[
            "track_id"
        ]


        x1, y1, x2, y2 = (
            track["bbox"]
        )


        confidence = track[
            "confidence"
        ]


        identity = identity_memory.get(
            track_id
        )


        if identity is None:

            identity_name = "UNKNOWN"

            similarity = 0.0

        else:

            identity_name = identity[
                "name"
            ]

            similarity = identity[
                "similarity"
            ]


        cv2.rectangle(
            display_frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (0, 255, 0),
            2
        )


        label = (
            f"ID {track_id} "
            f"{identity_name} "
            f"{similarity:.2f}"
        )


        cv2.putText(
            display_frame,
            label,
            (
                x1,
                max(
                    y1 - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # DRAW FACE BOXES
    # ========================================================

    for face_index, face in enumerate(
        face_detections
    ):

        x1, y1, x2, y2 = (
            face["bbox"]
        )


        associated_track_id = None


        for track_id, index in (
            associations.items()
        ):

            if index == face_index:

                associated_track_id = (
                    track_id
                )

                break


        if associated_track_id is not None:

            identity = identity_memory.get(
                associated_track_id
            )


            if identity is not None:

                face_name = identity[
                    "name"
                ]

                similarity = identity[
                    "similarity"
                ]

            else:

                face_name = "UNKNOWN"

                similarity = 0.0

        else:

            face_name = "UNKNOWN"

            similarity = 0.0


        cv2.rectangle(
            display_frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (255, 0, 0),
            2
        )


        face_label = (
            f"{face_name} "
            f"{similarity:.2f}"
        )


        cv2.putText(
            display_frame,
            face_label,
            (
                x1,
                min(
                    y2 + 22,
                    display_frame.shape[0] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if (
        key == ord("q")
        or
        key == 27
    ):

        break


# ============================================================
# 22. CLEANUP
# ============================================================

stop_event.set()
face_request_event.set()

if scrfd_thread.is_alive():
    scrfd_thread.join(
        timeout=1.0
    )

cap.release()

cv2.destroyAllWindows()

db.close()
