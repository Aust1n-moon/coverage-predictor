from ultralytics import YOLO

model = YOLO("Yolo/yolo-training/runs/detect/runs/nfl-player-tracker-3/weights/best.pt")
results = model.track(source="Yolo/Training videos/clip-2.mp4", show=True, conf=0.5, tracker="bytetrack.yaml")

locked_ids = set()

for r in results:
    boxes = r.boxes
