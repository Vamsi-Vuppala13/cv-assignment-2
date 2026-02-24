import os, glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

FRAMES_DIR = "frames"
OUT_PARQUET = "outputs/detections.parquet"
VIDEO_ID = "YcvECxtXoxQ"
SAMPLE_EVERY_SEC = 5

IMG_SIZE = 960   # speeds up 4K frames a lot
CONF_TH = 0.25

def main():
    model = YOLO("yolov8n.pt")  # baseline (swap to car-parts model if you have one)
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))

    rows = []
    for i, fp in enumerate(tqdm(frame_paths, desc="Detecting frames")):
        timestamp_sec = i * SAMPLE_EVERY_SEC
        res = model(fp, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            continue

        for b in res.boxes:
            cls_id = int(b.cls.item())
            label = model.names[cls_id]
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            rows.append({
                "video_id": VIDEO_ID,
                "frame_index": i,
                "timestamp_sec": timestamp_sec,
                "class_label": label,
                "bounding_box": (x1, y1, x2, y2),
                "confidence_score": conf,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"Wrote {len(df)} detections -> {OUT_PARQUET}")
    print("Columns:", df.columns.tolist())
    print(df.head(3))

if __name__ == "__main__":
    main()
