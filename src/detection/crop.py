from __future__ import annotations
import json, os, io, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Union, Tuple

import numpy as np
from PIL import Image
import cv2


@dataclass
class FaceBox:
    """Bounding box in absolute pixels."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: Optional[float] = None


def _to_numpy_rgb(image: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, Optional[str]]:
    """
    Accepts: file path, bytes, numpy array (HWC), or PIL Image.
    Returns: RGB uint8 numpy array and best-effort image_id (may be None).
    """
    image_id = None
    if isinstance(image, str):  # path
        p = Path(image)
        if p.is_dir():
            raise ValueError(f"Expected an image file, got directory: {p}")
        image_id = p.stem
        img = Image.open(p).convert("RGB")
        return np.array(img), image_id
    if isinstance(image, bytes):
        image_id = hashlib.sha1(image).hexdigest()[:16]
        file_bytes = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, image_id
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB")), image_id
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image array.")
        return arr.copy(), image_id
    raise TypeError("Unsupported image type; pass path, bytes, numpy array, or PIL.Image.")


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x, y, x + w, y + h


def _clip_box(box: FaceBox, H: int, W: int) -> FaceBox:
    return FaceBox(
        x1=max(0, min(W - 1, box.x1)),
        y1=max(0, min(H - 1, box.y1)),
        x2=max(0, min(W - 1, box.x2)),
        y2=max(0, min(H - 1, box.y2)),
        score=box.score,
    )


def _expand_box(box: FaceBox, H: int, W: int, pad: float) -> FaceBox:
    """Expand box by pad fraction (e.g., 0.2 adds 20% margin on each side)."""
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    bw = (box.x2 - box.x1)
    bh = (box.y2 - box.y1)
    bw2 = bw * (1 + pad * 2)
    bh2 = bh * (1 + pad * 2)
    x1 = cx - bw2 / 2.0
    y1 = cy - bh2 / 2.0
    x2 = cx + bw2 / 2.0
    y2 = cy + bh2 / 2.0
    return _clip_box(FaceBox(x1, y1, x2, y2, score=box.score), H, W)


def _align_by_5pts(img_rgb: np.ndarray, landmarks: np.ndarray, out_size: int) -> np.ndarray:
    """
    landmarks: np.ndarray shape (5,2) in (x,y) order for [left_eye, right_eye, nose, left_mouth, right_mouth]
    Uses a canonical ArcFace template for similarity transform.
    """
    ref_112 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    if out_size != 112:
        ref = ref_112 * (out_size / 112.0)
    else:
        ref = ref_112

    src = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
    if M is None:
        h, w, _ = img_rgb.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_rgb[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(img_rgb, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FaceCropSaver:
    """
    Prepare face crops (aligned or simple crops) for Super-Resolution training/inference.
    """
    def __init__(
        self,
        output_dir: str,
        crop_size: int = 128,
        padding: float = 0.25,
        bbox_mode: str = "xyxy",   # or "xywh"
        align: bool = True,
        image_format: str = "png", # "png" or "jpg"
        manifest_name: str = "manifest.jsonl",
    ):
        # Make absolute and create
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.crop_size = int(crop_size)
        self.padding = float(padding)
        assert bbox_mode in ("xyxy", "xywh")
        self.bbox_mode = bbox_mode
        self.align = bool(align)
        self.image_format = image_format.lower()
        assert self.image_format in ("png", "jpg", "jpeg")
        self.manifest_path = (self.output_dir / manifest_name)

        # open manifest in append mode
        self._manifest_f = open(self.manifest_path, "a", encoding="utf-8")

    def close(self):
        try:
            self._manifest_f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _normalize_boxes(self, boxes: Sequence[Union[FaceBox, Sequence[float], Dict]]) -> List[FaceBox]:
        norm: List[FaceBox] = []
        for b in boxes:
            if isinstance(b, FaceBox):
                norm.append(b)
            elif isinstance(b, dict):
                if self.bbox_mode == "xyxy":
                    norm.append(FaceBox(b["x1"], b["y1"], b["x2"], b["y2"], b.get("score")))
                else:
                    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                    x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
                    norm.append(FaceBox(x1, y1, x2, y2, b.get("score")))
            else:
                # assume sequence
                if self.bbox_mode == "xyxy":
                    x1, y1, x2, y2 = b[:4]
                    score = b[4] if len(b) > 4 else None
                    norm.append(FaceBox(x1, y1, x2, y2, score))
                else:
                    x, y, w, h = b[:4]
                    score = b[4] if len(b) > 4 else None
                    x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
                    norm.append(FaceBox(x1, y1, x2, y2, score))
        return norm

    def process_image(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        boxes: Sequence[Union[FaceBox, Sequence[float], Dict]],
        landmarks_5pts: Optional[Sequence[np.ndarray]] = None,
        image_id: Optional[str] = None,
        subdir_by_image: bool = True,
        extra_meta: Optional[Dict] = None,
    ) -> List[str]:
        """
        Returns list of saved crop paths.
        - landmarks_5pts: list with shape (num_faces, 5, 2) arrays if align=True
        """
        rgb, auto_id = _to_numpy_rgb(image)
        if image_id is None:
            image_id = auto_id or hashlib.sha1(rgb.tobytes()).hexdigest()[:16]

        H, W, _ = rgb.shape
        boxes_norm = self._normalize_boxes(boxes)

        if self.padding > 0:
            boxes_norm = [_expand_box(b, H, W, self.padding) for b in boxes_norm]
        else:
            boxes_norm = [_clip_box(b, H, W) for b in boxes_norm]

        # choose save directory
        save_dir = self.output_dir / image_id if subdir_by_image else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []
        for idx, box in enumerate(boxes_norm):
            x1, y1, x2, y2 = [int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2)]
            crop = rgb[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if self.align and landmarks_5pts is not None and idx < len(landmarks_5pts) and landmarks_5pts[idx] is not None:
                out_img = _align_by_5pts(rgb, landmarks_5pts[idx], out_size=self.crop_size)
                aligned_flag = True
            else:
                out_img = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                aligned_flag = False

            filename = f"{image_id}_face{idx:03d}.{self.image_format}"
            out_path = save_dir / filename

            if self.image_format in ("jpg", "jpeg"):
                bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                Image.fromarray(out_img).save(out_path)

            saved_paths.append(str(out_path))

            row = {
                "image_id": image_id,
                "source_path": str(image) if isinstance(image, (str, Path)) else None,
                "saved_path": str(out_path),
                "bbox": {"x1": float(box.x1), "y1": float(box.y1), "x2": float(box.x2), "y2": float(box.y2)},
                "score": box.score,
                "aligned": aligned_flag,
                "crop_size": self.crop_size,
                "padding": self.padding,
                "bbox_mode_input": self.bbox_mode,
            }
            if landmarks_5pts is not None and idx < len(landmarks_5pts) and landmarks_5pts[idx] is not None:
                row["landmarks_5pts"] = np.asarray(landmarks_5pts[idx], dtype=float).tolist()
            if extra_meta:
                row["meta"] = extra_meta

            self._manifest_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._manifest_f.flush()

        return saved_paths


if __name__ == "__main__":
    # Build paths relative to the project root (.. from src/detection → src → project)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # NOTE: put your real image under the project data folder (writable)
    image_path = PROJECT_ROOT / "data" / "raw" / "cctv" / "cctv1.png"

    bboxes_xyxy = [
        (400, 500, 450, 650, 0.98)  # x1,y1,x2,y2,score
    ]

    # WRITE TO A WRITABLE PATH (no leading slash)
    saver = FaceCropSaver(
        output_dir=PROJECT_ROOT / "data" / "interim" / "face_crops",
        crop_size=128,
        padding=0.25,
        bbox_mode="xyxy",
        align=False,
        image_format="png",
    )

    out_paths = saver.process_image(
        image=str(image_path),
        boxes=bboxes_xyxy,
        image_id=None,
        subdir_by_image=True,
        extra_meta={"camera": "A1", "ts": "2025-10-27T13:04:00Z"},
    )
    print("Saved:", out_paths)
    saver.close()
