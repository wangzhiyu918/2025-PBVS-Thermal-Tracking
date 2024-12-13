import os
import shutil

def process(targets, extracted_folder):
    folders = next(os.walk(extracted_folder))[1]
    for target in targets:
        target_folder = os.path.join(extracted_folder, target)
        os.makedirs(target_folder, exist_ok=True)

        for folder in folders:
            src_folder = os.path.join(extracted_folder, folder, target,"COCO")
            destination_folder = os.path.join(target_folder,folder)
            shutil.copytree(src_folder, destination_folder)



if __name__ == "__main__":
    targets = ["thermal", "rgb"]
    extracted_folder = "/media/wassimea/Storage/datasets/infolks_dataset/annotations/val/"
    process(targets, extracted_folder)