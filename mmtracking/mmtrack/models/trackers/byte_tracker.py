# Copyright (c) OpenMMLab. All rights reserved.
import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker

import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.transform import resize

@TRACKERS.register_module()
class ByteTracker(BaseTracker):
    """Tracker for ByteTrack.
    Args:
        obj_score_thrs (dict): Detection score threshold for matching objects.
            - high (float): Threshold of the first matching. Defaults to 0.6.
            - low (float): Threshold of the second matching. Defaults to 0.1.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thrs (dict): IOU distance threshold for matching between two
            frames.
            - high (float): Threshold of the first matching. Defaults to 0.1.
            - low (float): Threshold of the second matching. Defaults to 0.5.
            - tentative (float): Threshold of the matching for tentative
                tracklets. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 obj_score_thrs=dict(high=0.6, low=0.1),
                 init_track_thr=0.7,
                 weight_iou_with_det_scores=True,
                 match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives
        self.cv_gray_image = None

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids
    
    def calculate_current_hist(self, bb):
        bb_np = bb.cpu().numpy()[0]
        x1,y1,x2,y2,score = bb_np

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        roi = self.cv_gray_image[y1:y2, x1:x2]

        current_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

        #cv2.imshow("roi", roi)
        #cv2.waitKey()

        return current_hist

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True

        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)
        
        current_hist = self.calculate_current_hist(self.tracks[id].bboxes[-1])
        self.tracks[id].current_hist = current_hist
        self.tracks[id].hist_sim = 0

    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)
        
        current_hist = self.calculate_current_hist(self.tracks[id].bboxes[-1])
        hs = cv2.compareHist(cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX), cv2.normalize(self.tracks[id].current_hist, self.tracks[id].current_hist, 0, 1, cv2.NORM_MINMAX), cv2.HISTCMP_CORREL)
        self.tracks[id].hist_sim = hs
        self.tracks[id].current_hist = current_hist

    def pop_invalid_tracks(self, frame_id):
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id

            #case3 = v["hist_sim"] > 0.3 and frame_id > 5
            
            if case1 or case2:# or case3:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)
    
    def get_thermal_dists_hist(self, track_bboxes, det_bboxes, cv_img):
        track_bboxes = track_bboxes.cpu().numpy()
        det_bboxes = det_bboxes.cpu().numpy()

        # Initialize an empty similarity matrix
        dists = np.zeros((len(track_bboxes), len(det_bboxes)))

        # Calculate pairwise histogram similarity
        for i, track_bbox in enumerate(track_bboxes):
            for j, det_bbox in enumerate(det_bboxes):
                # Extract regions of interest from the grayscale image
                track_roi = cv_img[int(track_bbox[1]):int(track_bbox[3]), int(track_bbox[0]):int(track_bbox[2])]
                det_roi = cv_img[int(det_bbox[1]):int(det_bbox[3]), int(det_bbox[0]):int(det_bbox[2])]

                # Calculate histograms for the regions of interest
                track_hist = cv2.calcHist([track_roi], [0], None, [256], [0, 256])
                det_hist = cv2.calcHist([det_roi], [0], None, [256], [0, 256])

                # Normalize the histograms
                track_hist = cv2.normalize(track_hist, track_hist, 0, 1, cv2.NORM_MINMAX)
                det_hist = cv2.normalize(det_hist, det_hist, 0, 1, cv2.NORM_MINMAX)

                # Calculate histogram similarity using Bhattacharyya coefficient
                similarity = cv2.compareHist(track_hist, det_hist, cv2.HISTCMP_CORREL)
                similarity = (similarity + 1) / 2  # Normalize the similarity value
                similarity = 1-similarity

                # Update the similarity matrix
                dists[i, j] = similarity

        return dists

    def combine_with_multi_threshold(self, dists_iou, dists_thermal, threshold1, threshold2):
        # Create an array full of high values (e.g., 1.0) as initial combined distance
        dists_combined = np.ones_like(dists_iou)

        # Apply different strategies based on different thresholds
        mask1 = dists_iou <= threshold1
        mask2 = (dists_iou > threshold1) & (dists_iou <= threshold2)
        mask3 = dists_iou > threshold2

        # Set the values of the combined array based on the threshold
        # These rules can be changed according to specific requirements
        dists_combined[mask1] = dists_iou[mask1]  # trust IOU more when IOU is very low
        dists_combined[mask2] = np.fmin(dists_iou[mask2], dists_thermal[mask2])  # take the minimum value when in mid-range
        dists_combined[mask3] = np.fmin(dists_iou[mask3], dists_thermal[mask3])  # take the minimum value when IOU is high

        return dists_combined
    

    def adjust_dists(self, dists):
        rows, cols = dists.shape
        for i in range(rows):
            overlaps_idxs = []
            corresponding_thermal_signature = []
            for j in range(cols):
                if dists[i, j] < 1 and dists[i, j] >= 0.5:
                    overlaps_idxs.append(j)
                    corresponding_thermal_signature.append(dists[i,j])

            if len(overlaps_idxs) > 1:
                #for p in range(len(overlaps_idxs)):
                #    print("Row", i, "column ", overlaps_idxs[p], "value ", dists_iou[i,overlaps_idxs[p]],"thermal",corresponding_thermal_signature[p])

                min_thermal_sig_idx = corresponding_thermal_signature.index(min(corresponding_thermal_signature))

                for p in range(len(overlaps_idxs)):
                    if p != min_thermal_sig_idx:
                        dists[i, overlaps_idxs[p]] = 1

                v = 1
        return dists

    
    def assign_ids(self,
                   ids,
                   det_bboxes,
                   det_labels,
                   cv_gray_img,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5,
                   apply_thermal_iou_adjustment=False,
                   apply_weighted_matching=False):
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 5)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.
        Returns:
            tuple(int): The assigning ids.
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4])

        if apply_thermal_iou_adjustment:
            ious_np = ious.cpu().numpy()
            ious_np = self.adjust_dists(ious_np)
            ious = torch.from_numpy(ious_np).to(det_bboxes)

        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]

        dists_iou = 1 - ious.cpu().numpy()
        dists_thermal = self.get_thermal_dists_hist(track_bboxes, det_bboxes, cv_gray_img)

        #if not dists_iou.size == 0:
        #    dists_iou = (dists_iou - dists_iou.min()) / (dists_iou.max() - dists_iou.min())
        if not dists_thermal.size == 0:
            dists_thermal = (dists_thermal - dists_thermal.min()) / (dists_thermal.max() - dists_thermal.min())

        dists = dists_iou
        if apply_weighted_matching:
            alpha = 0.8
            dists = (alpha * dists_iou) + ((1-alpha) * dists_thermal)
            match_iou_thr = match_iou_thr + 0.3
            #dists = dists_thermal
        #dists = np.maximum(dists_iou, dists_thermal)

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    @force_fp32(apply_to=('img', 'bboxes'))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.
        Returns:
            tuple: Tracking results.
        """
        cv_gray_image = img_metas[0]["cv_gray_image"]
        self.cv_gray_image = cv_gray_image

        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            # get the detection bboxes for the first association
            first_det_inds = bboxes[:, -1] > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_ids = ids[first_det_inds]

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                bboxes[:, -1] > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_ids = ids[second_det_inds]

            # 1. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes, first_det_labels,cv_gray_image,
                self.weight_iou_with_det_scores, self.match_iou_thrs['high'], apply_thermal_iou_adjustment=False, apply_weighted_matching=False)
            

            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 first_unmatch_det_labels, cv_gray_image, self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'], apply_thermal_iou_adjustment=False, apply_weighted_matching=False)
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, second_det_labels,cv_gray_image,
                False, self.match_iou_thrs['low'],apply_thermal_iou_adjustment=False, apply_weighted_matching=False)
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        if -1 in self.ids:
            v = 1
        return bboxes, labels, ids
