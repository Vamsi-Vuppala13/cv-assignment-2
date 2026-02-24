import os
import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO

DETECTIONS_PARQUET = "outputs/detections.parquet"
OUT_CSV = "outputs/retrieval_results.csv"

MODEL_PATH = "yolov8n.pt"   # same detector as video step
CONF_TH = 0.25
GAP = 5  # seconds; frames are spaced 5 seconds apart

def merge_intervals(times, gap=5):
    if not times:
        return []
    times = sorted(set(int(t) for t in times))
    out = []
    s = p = times[0]
    k = 1
    for t in times[1:]:
        if t - p <= gap:
            p = t
            k += 1
        else:
            out.append((s, p, k))
            s = p = t
            k = 1
    out.append((s, p, k))
    return out

def main():
    det = pd.read_parquet(DETECTIONS_PARQUET)
    model = YOLO(MODEL_PATH)

    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")

    rows = []
    for i in range(len(ds)):
        img = ds[i]["image"]
        qts = int(ds[i].get("timestamp_sec", -1))

        r = model(img, conf=CONF_TH, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        labels = {model.names[int(b.cls.item())] for b in r.boxes}
        for lbl in sorted(labels):
            hits = det[(det["class_label"] == lbl) & (det["confidence_score"] >= CONF_TH)]
            for s,e,k in merge_intervals(hits["timestamp_sec"].tolist(), gap=GAP):
                rows.append({
                    "query_index": i,
                    "query_timestamp_sec": qts,
                    "class_label": lbl,
                    "start_timestamp": int(s),
                    "end_timestamp": int(e),
                    "number_of_supporting_detections": int(k),
                })

    out = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("wrote:", OUT_CSV, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
