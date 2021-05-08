import pytorch_lightning as pl
import torch
from torch import Tensor
import torchvision
from typing import List, Dict, Optional

from .fasterrcnn import fasterrcnn_resnet50_fpn
from .helpers.metrics import (
    mean_average_precision,
    transform_detections,
    transform_true_boxes,
    label_set_accuracy,
    label_set_recall
    )


class BaseModel(pl.LightningModule):

    def __init__(self,
                 optimizer,
                 learning_rate,
                 weight_decay,
                 pretrained_backbone,
                 num_classes
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.num_classes = num_classes
        # self.model = torchvision.models.detection.retinanet_resnet50_fpn(
        #                     num_classes=num_classes,
        #                     pretrained_backbone=pretrained_backbone
        #                     )
        self.model = fasterrcnn_resnet50_fpn(
                            num_classes=num_classes,
                            pretrained_backbone=pretrained_backbone
                            )

    def forward(self,
                image: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ):

        losses, detections = self.model(image, targets)

        return (losses, detections)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):

        images, targets = train_batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict, detections = self.model(images, targets)
        classification_loss = loss_dict['loss_classifier']
        bbox_loss = loss_dict['loss_box_reg']
        objectness_loss = loss_dict['loss_objectness']
        rpn_box_loss = loss_dict['loss_rpn_box_reg']

        loss = sum(loss for loss in loss_dict.values())

        self.log("train_classification_loss",
                 classification_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("train_bbox_regression_loss",
                 bbox_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("train_objectness_loss",
                 objectness_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("train_rpn_box_reg_loss",
                 rpn_box_loss,
                 on_epoch=True,
                 on_step=False
                 )
        self.log('train_loss', loss,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True,
                 logger=True
                 )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict, detections = self.model(images, targets)
        classification_loss = loss_dict['loss_classifier']
        bbox_loss = loss_dict['loss_box_reg']
        objectness_loss = loss_dict['loss_objectness']
        rpn_box_loss = loss_dict['loss_rpn_box_reg']

        loss = sum(loss for loss in loss_dict.values())

        pred_boxes = transform_detections(detections)
        true_boxes = transform_true_boxes(targets)

        mAP0dot5 = mean_average_precision(pred_boxes,
                                          true_boxes,
                                          iou_threshold=0.5,
                                          num_classes=self.num_classes
                                          )

        mAP0dot7 = mean_average_precision(pred_boxes,
                                          true_boxes,
                                          iou_threshold=0.7,
                                          num_classes=self.num_classes
                                          )

        mAP0dot05 = mean_average_precision(pred_boxes,
                                           true_boxes,
                                           iou_threshold=0.05,
                                           num_classes=self.num_classes
                                           )
        mAP0dot95 = mean_average_precision(pred_boxes,
                                           true_boxes,
                                           iou_threshold=0.95,
                                           num_classes=self.num_classes
                                           )

        # label_acc = label_set_accuracy(detections, targets)
        # label_rec = label_set_recall(detections, targets)

        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True
                 )

        self.log("val_classification_loss",
                 classification_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_bbox_regression_loss",
                 bbox_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_objectness_loss",
                 objectness_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_rpn_box_reg_loss",
                 rpn_box_loss,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_mAP@0.5",
                 mAP0dot5,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True
                 )

        self.log("val_mAP@0.7",
                 mAP0dot7,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_mAP@0.05",
                 mAP0dot05,
                 on_epoch=True,
                 on_step=False
                 )

        self.log("val_mAP@0.95",
                 mAP0dot95,
                 on_epoch=True,
                 on_step=False
                 )

        # self.log("val_label_set_accuracy",
        #          label_acc,
        #          on_epoch=True,
        #          on_step=False
        #          )

        # self.log("val_label_set_recall",
        #          label_rec,
        #          on_epoch=True,
        #          on_step=False
        #          )
