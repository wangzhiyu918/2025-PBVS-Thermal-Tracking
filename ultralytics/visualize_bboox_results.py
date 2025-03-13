import os
import cv2
import json

with open("./runs/detect/val/predictions.json", "r") as f:
    results = json.load(f)

image_to_results = {}
for result in results:
    image_name = result["image_id"]
    if image_name not in image_to_results:
        image_to_results[image_name] = [result]
    else:
        image_to_results[image_name].append(result)

output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)
image_dir = "../data/tmot_dataset_challenge/local_test/images/"
for image_name, results in image_to_results.items():
    image_path = os.path.join(image_dir, image_name + ".png")
    frame = cv2.imread(image_path)
    for result in results:
        bbox = result['bbox']
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    frame_name = os.path.basename(image_path)[:-4] + ".jpg"
    output_frame_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(output_frame_path, frame)
