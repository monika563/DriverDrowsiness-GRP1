"""
drowsiness_hybrid.py

Hybrid driver drowsiness detection:
 - EAR (eye aspect ratio) from MediaPipe face mesh (fast, classic)
 - CNN classifier for eye images (deep learning hybrid)
 - Plays alert sound and shows text overlay on video
Usage:
  python drowsiness_hybrid.py          # run detector (will try to load eye_cnn.h5 if present)
  python drowsiness_hybrid.py train    # train CNN on dataset/open and dataset/closed folders
Dataset structure for training:
  dataset/
    open/
      img1.jpg
      ...
    closed/
      img1.jpg
      ...
"""
import os
import sys
import cv2
import time
import numpy as np
from threading import Thread

# --------- try imports (informative errors) ----------
try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Install: pip install mediapipe") from e

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    raise ImportError("tensorflow (and keras) required. Install: pip install tensorflow") from e

# simple cross-platform audio
# --- Cross-platform simple alarm sound using playsound ---
from threading import Thread
from playsound import playsound

def play_alarm_sound_nonblocking(_):
    Thread(target=lambda: playsound("alarm.wav"), daemon=True).start()


# --------- constants / thresholds ----------
EAR_THRESHOLD = 0.23           # below this => eye considered 'closed' by EAR
CONSECUTIVE_FRAMES_THRESHOLD = 15  # number of consecutive frames to declare drowsiness
CNN_CONFIDENCE_THRESHOLD = 0.65  # CNN must be confidently closed to count
MODEL_PATH = "eye_cnn.h5"

# MediaPipe and eye indices
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for left/right eye (commonly used)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# --------- utility functions ----------
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    """
    landmarks: list of mediapipe normalized landmarks
    eye_indices: 6 indices (p1..p6)
    returns EAR scalar
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        pts.append((x, y))
    # p1..p6
    (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = pts
    # vertical distances
    def dist(a,b):
        return np.linalg.norm(np.array(a)-np.array(b))
    A = dist((x2,y2),(x6,y6))
    B = dist((x3,y3),(x5,y5))
    C = dist((x1,y1),(x4,y4))
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear, pts

def extract_eye_patch(frame, pts, expand_ratio=0.25, size=(34,26)):
    # pts are 6 points; compute bounding rect
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    # expand
    ex = int(w * expand_ratio)
    ey = int(h * expand_ratio)
    x1 = max(0, x_min - ex)
    y1 = max(0, y_min - ey)
    x2 = min(frame.shape[1], x_max + ex)
    y2 = min(frame.shape[0], y_max + ey)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch_resized = cv2.resize(patch_gray, size)
    patch_norm = patch_resized.astype("float32") / 255.0
    patch_norm = np.expand_dims(patch_norm, axis=-1)  # channel
    return patch_norm

def build_eye_cnn(input_shape=(26,34,1)):  # (h,w,c) note: small net
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # closed probability
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_or_build_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = load_model(MODEL_PATH)
            print("[INFO] Loaded CNN model from", MODEL_PATH)
            return m
        except Exception as e:
            print("[WARN] Failed to load existing model:", e)
    print("[INFO] Building new CNN (untrained). To enable DL detections, run 'python drowsiness_hybrid.py train' with dataset.")
    return build_eye_cnn(input_shape=(26,34,1))

# Create a short beep WAV byte sequence for simpleaudio fallback if desired
# Here we won't create raw bytes; simpleaudio usage in code expects WAV bytes; for most setups user will use installed wav file if they want.
ALARM_WAV_BYTES = None

# ---------- training function ----------
def train_cnn_from_dataset(dataset_dir="dataset", epochs=8, batch_size=16, target_size=(34,26)):
    open_dir = os.path.join(dataset_dir, "open")
    closed_dir = os.path.join(dataset_dir, "closed")
    if not (os.path.isdir(open_dir) and os.path.isdir(closed_dir)):
        print("[ERROR] Dataset not found. Expected structure:")
        print("  dataset/open/*.jpg")
        print("  dataset/closed/*.jpg")
        return

    # Keras ImageDataGenerator expects shape (height,width)
    # We'll use a generator that reads grayscale images and resizes to target_size (width,height) careful: Keras uses (height,width)
    img_h, img_w = target_size[1], target_size[0]  # target_size was (w,h) in earlier funcs; adapt
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                 rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.05)
    # create combined folder with subfolders 'open' and 'closed' so flow_from_directory works on dataset_dir
    train_gen = datagen.flow_from_directory(dataset_dir,
                                            target_size=(img_h, img_w),
                                            color_mode='grayscale',
                                            class_mode='binary',
                                            batch_size=batch_size,
                                            subset='training',
                                            shuffle=True)
    val_gen = datagen.flow_from_directory(dataset_dir,
                                          target_size=(img_h, img_w),
                                          color_mode='grayscale',
                                          class_mode='binary',
                                          batch_size=batch_size,
                                          subset='validation',
                                          shuffle=True)
    model = build_eye_cnn(input_shape=(img_h, img_w, 1))
    print("[INFO] Starting training...")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(MODEL_PATH)
    print("[INFO] Training finished. Model saved to", MODEL_PATH)

# ---------- main detection / loop ----------
def run_detector(use_cnn=True):
    model = None
    if use_cnn:
        model = load_or_build_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    frame_counter = 0
    consec_drowsy = 0
    alarm_on = False
    # For FPS smoothing
    prev_time = time.time()
    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            ear_avg = None
            cnn_closed_prob = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # left eye
                try:
                    ear_l, pts_l = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                    ear_r, pts_r = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
                    ear_avg = (ear_l + ear_r) / 2.0
                    # draw eye contours
                    for p in pts_l + pts_r:
                        cv2.circle(frame, p, 1, (0,255,0), -1)
                    # CNN prediction on concatenated both eyes (or just check both)
                    if use_cnn and model is not None:
                        left_patch = extract_eye_patch(frame, pts_l, expand_ratio=0.3, size=(34,26))
                        right_patch = extract_eye_patch(frame, pts_r, expand_ratio=0.3, size=(34,26))
                        probs = []
                        for pp in [left_patch, right_patch]:
                            if pp is None:
                                continue
                            X = np.expand_dims(pp, axis=0)  # (1,h,w,1)
                            pred = model.predict(X, verbose=0)[0][0]
                            probs.append(pred)
                        if len(probs) > 0:
                            # closed probability is mean
                            cnn_closed_prob = float(np.mean(probs))
                        else:
                            cnn_closed_prob = None
                except Exception as e:
                    # sometimes landmarks mapping fails; keep going
                    pass

            # decision: if ear_avg below threshold or cnn says closed => increment
            is_drowsy_frame = False
            reasons = []
            if ear_avg is not None:
                if ear_avg < EAR_THRESHOLD:
                    is_drowsy_frame = True
                    reasons.append(f"EAR {ear_avg:.3f}")
            if cnn_closed_prob is not None:
                if cnn_closed_prob >= CNN_CONFIDENCE_THRESHOLD:
                    is_drowsy_frame = True
                    reasons.append(f"CNN {cnn_closed_prob:.2f}")
            # update counters
            if is_drowsy_frame:
                consec_drowsy += 1
            else:
                consec_drowsy = max(0, consec_drowsy - 1)  # soften recovery
            # draw info
            status_text = "SAFE DRIVE"
            color = (0,255,0)
            if consec_drowsy >= CONSECUTIVE_FRAMES_THRESHOLD:
                status_text = "DROWSINESS DETECTED - WAKE UP!"
                color = (0,0,255)
                if not alarm_on:
                    alarm_on = True
                    # play alarm sound (non-blocking)
                    play_alarm_sound_nonblocking(ALARM_WAV_BYTES)
            else:
                alarm_on = False

            # draw overlay
            cv2.putText(frame, status_text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            # debug details
            y0 = 80
            if ear_avg is not None:
                cv2.putText(frame, f"EAR: {ear_avg:.3f}", (30,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                y0 += 25
            if cnn_closed_prob is not None:
                cv2.putText(frame, f"CNN_closed_prob: {cnn_closed_prob:.2f}", (30,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                y0 += 25
            cv2.putText(frame, f"Consec: {consec_drowsy}", (30,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            cv2.imshow("Driver Drowsiness Detection (Hybrid)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ---------- entry point ----------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        # Train mode
        epochs = 8
        if len(sys.argv) > 2:
            try:
                epochs = int(sys.argv[2])
            except:
                pass
        print("[INFO] Training CNN with dataset; epochs =", epochs)
        train_cnn_from_dataset(dataset_dir="dataset", epochs=epochs)
        sys.exit(0)
    else:
        # Run detector. If model file does not exist we still run with EAR-only
        use_cnn_flag = True
        if not os.path.exists(MODEL_PATH):
            print("[WARN] Model file not found; running with EAR-only hybrid (CNN inactive). To enable CNN, place eye_cnn.h5 or run training mode.")
            use_cnn_flag = False
        run_detector(use_cnn=use_cnn_flag)
