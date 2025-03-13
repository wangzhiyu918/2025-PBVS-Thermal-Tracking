from ultralytics import YOLO
import os
import cv2

def val():
    model = YOLO(f"runs/detect/train/weights/epoch75.pt")
    metrics = model.val(imgsz=1280, save_json=True, batch=1, workers=4, save=True)

if __name__ == '__main__':
    val()
