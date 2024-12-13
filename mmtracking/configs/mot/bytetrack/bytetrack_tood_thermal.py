_base_ = [
    '../../_base_/models/tood.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]


img_scale = (800, 1440)
samples_per_gpu = 4

model = dict(
    type='ByteTrack',
    detector=dict(
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/wassimea/Desktop/tmot/mmdetection/work_dirs/tood_thermal/epoch_12.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))