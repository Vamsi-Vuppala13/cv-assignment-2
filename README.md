# CV Spring 2026 — Assignment 2 (Image-to-Video Retrieval)

## Hugging Face (detections parquet)
https://huggingface.co/datasets/VamsiVuppala/cv-a2-detections

## What’s in this repo
- `scripts/detect_to_parquet.py` — detect objects in sampled video frames -> `detections.parquet`
- `scripts/retrieve_queries.py` — detect objects in query images -> retrieve time intervals -> `retrieval_results.csv`

## How to run
1) Download video:
   `yt-dlp -o input_video.mp4 https://www.youtube.com/watch?v=YcvECxtXoxQ`
2) Extract frames (1 per 5 sec):
   `ffmpeg -i input_video.mp4 -vf "fps=1/5" frames/frame_%06d.jpg`
3) Detect + write parquet:
   `python scripts/detect_to_parquet.py`
4) Retrieve:
   `python scripts/retrieve_queries.py`
