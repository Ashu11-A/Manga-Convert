from ultralytics import YOLO, checks

checks()

async def yoloFindBetter (size: int | list[int] = 1280):
  model = YOLO("yolov8n-seg", task='segment')
  
  # https://docs.ultralytics.com/pt/guides/hyperparameter-tuning
  model.tune(
    data="/home/ashu/Documents/GitHub/Manga-Convert/coco-seg/data.yaml",
    imgsz=size,
    epochs=100,
    iterations=1000,
    batch=-1,
    plots=False,
    save=False,
    val=False
  )