# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_mot, init_model

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, track_classes in results:
            for tlwh, track_id, score, track_class in zip(tlwhs, track_ids, scores, track_classes):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = tlwh
                w = x2-x1
                h = y2-y1
                label = track_class
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                            h=round(h, 1), s=round(score, 2), label=label)
                f.write(line)
    #logger.info('save results to {}'.format(filename))




def process(folder, conf, outfile):
    results = []

    imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(folder)),
            key=lambda x: int(x.split('.')[0]))
    
    model = init_model(conf, None, device="cuda:0")

    prog_bar = mmcv.ProgressBar(len(imgs))

    for frame_id, img in enumerate(imgs):
        img = osp.join(folder, img)
        result = inference_mot(model, img, frame_id=frame_id)

        #model.show_result(
        #    img,
        #    result,
        #    score_thr=0.8,
        #    show=True,
        #    wait_time=int(1000. / 5),
        #    out_file=None,
        #    thickness=3)


        online_x1y1x2y2 = []
        online_ids = []
        online_scores = []
        online_cls = []

        for online_track in result["track_bboxes"][0]:
            #online_track = online_track[0]
            track_id = online_track[0]
            x1 = online_track[1]
            y1 = online_track[2]
            x2 = online_track[3]
            y2 = online_track[4]
            score = online_track[5]
            online_x1y1x2y2.append([x1,y1,x2,y2])
            online_ids.append(track_id)
            online_scores.append(score)
            online_cls.append(1)

        results.append((frame_id+1, online_x1y1x2y2, online_ids, online_scores, online_cls))

        prog_bar.update()

    write_results(outfile, results)



if __name__ == "__main__":
    mmtrack_config = "/home/wassimea/Desktop/tmot/mmtracking/configs/mot/ocsort/ocsort_tood_rgb.py"
    mmtrack_config = "./mmtracking/configs/mot/bytetrack/bytetrack_tood_thermal.py"

    data_folder = "/media/wassimea/Storage/datasets/infolks_dataset/images_unprocessed/val/"
    data_folder = "../data/tmot_dataset_challenge/images/val/"

    out_folder = "/home/wassimea/Desktop/preds/"
    out_folder = "./preds/"

    sequences = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]
    #sequences = ["seq47"]

    target = "rgb"
    target = "thermal"

    for sequence in sequences:
        process(os.path.join(data_folder, sequence, target), mmtrack_config, os.path.join(out_folder, sequence + "_" + target + ".txt"))
