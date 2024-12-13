import mmcv
from mmdet.apis import init_detector, inference_detector
import os
import cv2

def run_mmdetection_inference(config_file, checkpoint_file, input_images):
    # Load the configuration file
    # Build the detector model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    threshold = 0.4

    for image_path in input_images:
        # Read input image
        image = cv2.imread(image_path)

        # Run inference on the input image
        result = inference_detector(model, image).cpu().numpy()

        boxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            if scores[i] > threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)

        # Wait for key press to proceed to the next image
        cv2.imshow("im", image)
        cv2.waitKey(1)

# Example usage
config_file = '/home/wassimea/Desktop/tmot/mmdetection/work_dirs/tood_thermal/tood_thermal.py'
checkpoint_file = '/home/wassimea/Desktop/tmot/mmdetection/work_dirs/tood_thermal/epoch_10.pth'
input_image_dir = '/media/wassimea/5CDE5210DE51E336/Users/welah/Desktop/infolks_annotate/seq17/thermal/'

# Get the paths of all image files in the input directory
input_images = [os.path.join(input_image_dir, img) for img in os.listdir(input_image_dir) if img.endswith('.png')]

# Run inference on multiple images and display the results
run_mmdetection_inference(config_file, checkpoint_file, input_images)
