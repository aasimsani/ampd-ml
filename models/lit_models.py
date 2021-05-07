import pytorch_lightning as pl
import torch
from torch import Tensor
import torchvision
from typing import List, Dict, Optional


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
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
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

        loss_dict = self.model(images, targets)
        # classification_loss = loss_dict['classification']
        # bbox_loss = loss_dict['bbox_regression']
        loss = sum(loss for loss in loss_dict.values())

        self.log('train_loss', loss,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True,
                 logger=True
                 )

        # self.log("train_classification_loss",
        #          classification_loss,
        #          on_epoch=True,
        #          on_step=False
        #          )

        # self.log("train_bbox_regression_loss",
        #          bbox_loss,
        #          on_epoch=True,
        #          on_step=False
        #          )

        return {"loss": loss}
