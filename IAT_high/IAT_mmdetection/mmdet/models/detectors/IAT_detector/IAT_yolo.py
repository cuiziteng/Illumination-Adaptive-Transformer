# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
from ...builder import DETECTORS, build_backbone, build_head, build_neck
from ..single_stage import SingleStageDetector

@DETECTORS.register_module()
class IAT_YOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 pre_encoder,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(IAT_YOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)

        self.pre_encoder = build_backbone(pre_encoder)

        def extract_feat(self, img):
            """Directly extract features from the backbone+neck."""
            _, _, x = self.pre_encoder(img)
            x = self.backbone(x)
            if self.with_neck:
                x = self.neck(x)
            return x

        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None):
            """
            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                    Typically these should be mean centered and std scaled.
                img_metas (list[dict]): A List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    :class:`mmdet.datasets.pipelines.Collect`.
                gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                    image in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): Class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                    boxes can be ignored when computing the loss.

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            super(SingleStageDetector, self).forward_train(img, img_metas)
            # print(img_metas)
            # print(img.shape)
            x = self.extract_feat(img)
            # check_locations(img, img_metas, gt_bboxes, gt_labels, '/home/czt/mmdetection/SHOW/KITTI')
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
            return losses


