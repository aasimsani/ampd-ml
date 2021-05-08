from data.pytorch_dataset import MangaPanelDataset
from models.lit_models import BaseModel
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def train_model(args):

    wandb_logger = WandbLogger(project='ampd-speech-bubble-detection',
                               entity='aasimsani'
                               )
    # Loading dataset
    labels_file = "data/datasets/speech_bubble_locations.csv"
    image_folder = "data/datasets/page_images/"
    train_dataset = MangaPanelDataset(
                                labels_file=labels_file,
                                image_folder=image_folder,
                                split=[0.0, 0.8]
                                )
    val_dataset = MangaPanelDataset(
                                labels_file=labels_file,
                                image_folder=image_folder,
                                split=[0.8, 1.0]
                                )

    collate_fn = train_dataset.collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=6,
                                               collate_fn=collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=6,
                                             collate_fn=collate_fn
                                             )

    optimizer = torch.optim.Adam
    weight_decay = 0.0
    learning_rate = 3e-5

    trainer = pl.Trainer.from_argparse_args(args)
    if args.log:
        trainer.logger = wandb_logger

    model = BaseModel(optimizer,
                      learning_rate,
                      weight_decay,
                      pretrained_backbone=True,
                      num_classes=2
                      )

    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader
                )
