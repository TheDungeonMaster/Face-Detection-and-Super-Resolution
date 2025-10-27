import os, cv2, json, uuid
from tqdm import tqdm
from retinaface import RetinaFace

def detect_faces_retina(img):
    try:
        results = RetinaFace.detect_faces(img)
        if isinstance(results, dict):
            detections = []
            for key, det in results.items():
                x1, y1, x2, y2 = det["facial_area"]
                conf = det["score"]
                landmarks = det["landmarks"]
                detections.append({
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": conf,
                    "landmarks": landmarks
                })
            return detections
        return []
    except Exception as e:
        print(f"[WARN] Detection failed: {e}")
        return []
