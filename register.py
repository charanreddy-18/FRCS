from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os


# ============================================================
# 1. SETTINGS
# ============================================================

DATABASE_PATH = "database"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

DETECTION_THRESHOLD = 0.10
DETECTION_SIZE = (160, 160)


# ============================================================
# 2. LOAD INSIGHTFACE MODEL
# ============================================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_thresh=DETECTION_THRESHOLD,
    det_size=DETECTION_SIZE
)


# ============================================================
# 3. CREATE DATABASE FOLDER
# ============================================================

os.makedirs(
    DATABASE_PATH,
    exist_ok=True
)


# ============================================================
# 4. GET PERSON NAME
# ============================================================

person_name = input(
    "Enter person's name: "
).strip()

if not person_name:
    print("ERROR: Person name cannot be empty.")
    exit()


# ============================================================
# 5. GET IMAGE FOLDER
# ============================================================

image_folder = input(
    "Enter image folder: "
).strip()

if not os.path.isdir(image_folder):
    print(
        f"ERROR: Folder not found: {image_folder}"
    )
    exit()


# ============================================================
# 6. GET ALL IMAGES FROM FOLDER
# ============================================================

image_files = []

for filename in os.listdir(image_folder):

    if filename.lower().endswith(IMAGE_EXTENSIONS):

        image_files.append(
            os.path.join(
                image_folder,
                filename
            )
        )


if len(image_files) == 0:

    print(
        "ERROR: No images found in the folder."
    )

    exit()


# ============================================================
# 7. PROCESS ALL IMAGES
# ============================================================

embeddings = []

print()
print(
    f"Found {len(image_files)} image(s)."
)
print()
print("Processing images...")
print()


for image_path in image_files:

    print(
        f"Processing: "
        f"{os.path.basename(image_path)}"
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print(
            "  ERROR: Could not read image."
        )

        continue


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = app.get(image)


    if len(faces) == 0:

        print(
            "  WARNING: No face detected."
        )

        continue


    # --------------------------------------------------------
    # Make sure only one face is present
    # --------------------------------------------------------

    if len(faces) > 1:

        print(
            "  WARNING: Multiple faces detected. "
            "Skipping image."
        )

        continue


    # --------------------------------------------------------
    # Get face embedding
    # --------------------------------------------------------

    embedding = faces[0].embedding

    embeddings.append(
        embedding
    )

    print(
        "  Face detected successfully."
    )


# ============================================================
# 8. CHECK RESULTS
# ============================================================

if len(embeddings) == 0:

    print()
    print(
        "ERROR: No valid face embeddings were created."
    )
    print(
        "Registration failed."
    )

    exit()


# ============================================================
# 9. CONVERT TO NUMPY ARRAY
# ============================================================

embeddings = np.array(
    embeddings
)


# ============================================================
# 10. SAVE TO DATABASE
# ============================================================

safe_name = (
    person_name
    .lower()
    .replace(" ", "_")
)

database_file = os.path.join(
    DATABASE_PATH,
    safe_name + ".npy"
)

np.save(
    database_file,
    embeddings
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print()
print(
    "=========================================="
)
print(
    "REGISTRATION SUCCESSFUL"
)
print(
    "=========================================="
)

print(
    f"Person       : {person_name}"
)

print(
    f"Images found : {len(image_files)}"
)

print(
    f"Faces stored : {len(embeddings)}"
)

print(
    f"Database file: {database_file}"
)

print(
    "=========================================="
)