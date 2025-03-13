from ultralytics import YOLO
import os
import cv2

def train():
    model = YOLO('yolov8s.pt')
    model.train(data='ultralytics/cfg/datasets/pbvs.yaml', epochs=90, batch=16,
                imgsz=1280, device=[7], save_period=5, workers=32)

if __name__ == '__main__':
    train()
