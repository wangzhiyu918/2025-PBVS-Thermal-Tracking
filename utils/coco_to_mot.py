
import os
import json

def coco_to_mot(coco_file, mot_file):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    mot_data = []
    for image in coco_data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                object_id = annotation['track_id']
                bbox = annotation['bbox']
                object_class = annotation['category_id']
                # You may need to adjust the coordinates depending on the format used in your dataset
                x, y, w, h = bbox
                #x_center = x + w / 2
                #y_center = y + h / 2
                
                mot_data.append([image_id, object_id, x, y, w, h, object_class, -1, -1, -1])

    with open(mot_file, 'w') as f:
        for item in mot_data:
            f.write(','.join(str(i) for i in item) + '\n')

# Example usage

annotations_folder = "/media/wassimea/Storage/datasets/infolks_dataset/annotations/val/"

out_folder = "/home/wassimea/Desktop/gt/"

sequences = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]

#sequences = ["seq47"]

target = "rgb"

for sequence in sequences:
    coco_file = os.path.join(annotations_folder, sequence, target + "/COCO/annotations.json")
    mot_file = os.path.join(out_folder, sequence + "_" + target + ".txt")
    coco_to_mot(coco_file, mot_file)
