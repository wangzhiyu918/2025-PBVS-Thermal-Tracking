import os
import cv2
from glob import glob
import numpy as np

def visualize_tracking_results(results_file, frames_dir, output_video):
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = [line.strip().split(',') for line in f.readlines()]
    
    # 获取帧信息
    frame_data = {}
    for result in results:
        frame_id = int(result[0])
        track_id = int(result[1])
        x1, y1, w, h = map(float, result[2:6])
        score = float(result[6])
        
        if frame_id not in frame_data:
            frame_data[frame_id] = []
        
        frame_data[frame_id].append({
            'track_id': track_id,
            'bbox': (x1, y1, w, h),
            'score': score
        })

    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = glob(f"{frames_dir}/*.png")
    for i, frame_path in enumerate(frame_paths):
        frame_id = i + 1
        frame = cv2.imread(frame_path)

        if frame_id in frame_data:
            # 绘制每一帧中的目标框和信息
            for obj in frame_data[frame_id]:
                x1, y1, w, h = obj['bbox']
                track_id = obj['track_id']
                score = obj['score']
                
                # 绘制目标框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
                
                # 显示 ID 和置信度
                label = f"ID:{track_id} Score:{score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame_name = os.path.basename(frame_path)[:-4] + ".jpg"
        output_frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(output_frame_path, frame)

# 使用示例
results_file = './seq2_thermal.txt'  # 结果文件路径
frames_dir = '../data/tmot_dataset_challenge/images/val/seq2/thermal/'  # 视频帧目录
output_video = 'output_tracking_video.avi'  # 输出视频路径

visualize_tracking_results(results_file, frames_dir, output_video)
