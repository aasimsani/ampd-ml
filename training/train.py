from data.pytorch_dataset import MangaPanelDataset
from models.lit_models import BaseModel
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import calendar

import wandb


def train_model(args):

    wandb_logger = WandbLogger(project='ampd-speech-bubble-detection',
                               entity='aasimsani',
                               log_model=False,
                               save_code=False
                               )
    # Loading dataset
    labels_file = "data/datasets/speech_bubble_locations.csv"
    image_folder = "data/datasets/page_images_shrunk/"

    train_split = [0.0, 0.8]
    val_split = [0.8, 1.0]
    # train_split = [0.0, 0.1]
    # val_split = [0.1, 0.15]
    train_dataset = MangaPanelDataset(
                                labels_file=labels_file,
                                image_folder=image_folder,
                                split=train_split
                                )

    val_dataset = MangaPanelDataset(
                                labels_file=labels_file,
                                image_folder=image_folder,
                                split=val_split
                                )

    collate_fn = train_dataset.collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=6,
                                               prefetch_factor=4,
                                               collate_fn=collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=6,
                                             collate_fn=collate_fn
                                             )

    optimizer = torch.optim.AdamW
    weight_decay = 0.0
    learning_rate = 6.918309709189363e-05

    trainer = pl.Trainer.from_argparse_args(args)
    if args.log:
        trainer.logger = wandb_logger

    model = BaseModel(optimizer,
                      learning_rate,
                      weight_decay,
                      pretrained_backbone=False,
                      num_classes=2,
                      )

    # trainer.tune(model, train_dataloader=train_loader)
    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader,
                )

    if args.log:
        d = datetime.utcnow()
        unixtime = str(calendar.timegm(d.utctimetuple()))
        checkpoint_name = "fastercnn"+unixtime+".ckpt"
        checkpoint_path = "artifacts/weights/" + checkpoint_name
        trainer.save_checkpoint(checkpoint_path)

        print("Registering and logging artifact")
        artifact = wandb.Artifact("trained-fasterrcnn",
                                  type="checkpoint"
                                  )
        artifact.add_file(checkpoint_path)
        wandb_logger._experiment.log_artifact(artifact)
        print("Finished logging artifact")
