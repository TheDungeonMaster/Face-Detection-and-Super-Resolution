import os, cv2, json, uuid
from tqdm import tqdm
from retinaface import RetinaFace


def process_image(image_path, output_dir, metadata):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return
    detections = detect_faces_retina(img)
    for det in detections:
        x, y, w, h = map(int, det["box"])
        conf = det["confidence"]
        if conf < 0.9:
            continue
        face_id = str(uuid.uuid4())[:8]
        face_crop = img[y:y+h, x:x+w]
        crop_path = os.path.join(output_dir, f"{face_id}.png")
        cv2.imwrite(crop_path, face_crop)
        metadata.append({
            "id": face_id,
            "source": os.path.basename(image_path),
            "bbox": [x, y, w, h],
            "score": float(conf),
            "detector": "RetinaFace"
        })

def process_video(video_path, output_dir, metadata, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            detections = detect_faces_retina(frame)
            for det in detections:
                x, y, w, h = map(int, det["box"])
                conf = det["confidence"]
                if conf < 0.9:
                    continue
                face_id = str(uuid.uuid4())[:8]
                face_crop = frame[y:y+h, x:x+w]
                crop_path = os.path.join(output_dir, f"{face_id}.png")
                cv2.imwrite(crop_path, face_crop)
                metadata.append({
                    "id": face_id,
                    "source": f"{os.path.basename(video_path)}:frame_{frame_idx}",
                    "bbox": [x, y, w, h],
                    "score": float(conf),
                    "detector": "RetinaFace"
                })
        frame_idx += 1
    cap.release()

def main(input_dir="input", output_dir="output/faces_retina"):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for fname in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, fname)
        if fname.lower().endswith((".jpg", ".png")):
            process_image(path, output_dir, metadata)
        elif fname.lower().endswith((".mp4", ".avi", ".mov")):
            process_video(path, output_dir, metadata)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Done. Crops saved to {output_dir}, metadata written to metadata.json")

if __name__ == "__main__":
    main()

