import os
import json

def merge_coco_annotations(directory):
    merged_data = {
        "info": [],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Update image IDs
                for image in data["images"]:
                    image["id"] += image_id_offset
                    merged_data["images"].append(image)

                # Update annotation IDs
                for annotation in data["annotations"]:
                    annotation["id"] += annotation_id_offset
                    annotation["image_id"] += image_id_offset
                    merged_data["annotations"].append(annotation)

                # Merge "categories" key
                merged_data["categories"].extend(data["categories"])

                # Merge "info" key
                if "info" in data:
                    if not merged_data["info"]:
                        merged_data["info"].append(data["info"])
                    else:
                        merged_data["info"] = data["info"]

                # Merge "licenses" key
                if "licenses" in data:
                    merged_data["licenses"].extend(data["licenses"])

                # Update ID offsets
                image_id_offset += len(data["images"])
                annotation_id_offset += len(data["annotations"])

    return merged_data

# Example usage
targets = ["rgb", "thermal"]
directory_path = "/media/wassimea/Storage/datasets/infolks_dataset/annotations/val/"

for target in targets:
    merged_annotations = merge_coco_annotations(os.path.join(directory_path, target))

    # Save merged annotations to a new file
    output_file = os.path.join(directory_path, target + "/annotations.json")
    with open(output_file, "w") as f:
        json.dump(merged_annotations, f, indent=4)