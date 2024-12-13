import os
import shutil

if __name__ == "__main__":
    mode = "rgb"
    annotated_sequences_dir = "/media/wassimea/Storage/datasets/infolks_dataset/annotations/train/rgb/"
    image_dataset_dir = "/media/wassimea/Storage/datasets/infolks_dataset/images_unprocessed/train/"
    out_dir = "/media/wassimea/Storage/datasets/infolks_dataset/images/rgb/train/"


    annotated_sequences = next(os.walk(annotated_sequences_dir))[1]
    for annotated_sequence in annotated_sequences:
        src_folder = os.path.join(image_dataset_dir, annotated_sequence, mode)
        png_files = [f for f in os.listdir(src_folder) if f.endswith('.png')]
        for png_file in png_files:
            shutil.copy(os.path.join(src_folder, png_file), os.path.join(out_dir, png_file))

        v = 1
    