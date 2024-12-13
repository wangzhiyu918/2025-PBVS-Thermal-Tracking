_base_ = [
    '../../_base_/models/tood.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]


img_scale = (800, 1440)
samples_per_gpu = 4

model = dict(
    type='OCSORT',
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
        type='OCSORTTracker',
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))