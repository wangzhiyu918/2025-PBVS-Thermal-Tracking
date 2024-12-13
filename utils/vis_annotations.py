import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import json

import sys
sys.path.append('/home/stronguser/Desktop/cocoapi/PythonAPI')

from pycocotools.coco import COCO
import pycocotools.mask as mask
import cv2

#def process():
#    tif_dir = "/home/stronguser/projects/median-vision/data/raw/msdis/relevant_tiles_tif/"
#    tifs = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
#    random.shuffle(tifs)
#
#    tifs = tifs[:1]
#
#    sids = []
#    for tif in tifs:
#        sid = tif.split("_", 1)[1].replace(".tif","")
#        sids.append(sid)
#    print(sids)

def read_ans():
    coco_annotation_file_path='/home/wassimea/Desktop/tmot/ann_results/thermal/annotations.json'
    img_dir = '/home/wassimea/Desktop/tmot/images/thermal/'
    coco = COCO(annotation_file=coco_annotation_file_path)

    for i in range(1,500):
        img = coco.imgs[i]

        cv_img = cv2.imread(os.path.join(img_dir, img['file_name']))
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        for ann in anns:
            x1, y1, w, h = ann["bbox"]

            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)

            x2 = x1 + w
            y2 = y1 + h

            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0,0,255), 2)


        cv2.imshow("im", cv_img)
        cv2.waitKey()


if __name__ == "__main__":
    read_ans()