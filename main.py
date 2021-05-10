import pytorch_lightning as pl
from argparse import ArgumentParser
from data.preprocessing.create_tags import create_speech_bubble_data
from data.preprocessing.shrink_images import shrink_images
from training.train import train_model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--create_tags", action="store_true")
    parser.add_argument("--shrink_images", action="store_true")
    parser.add_argument("--log", action="store_true")

    args = parser.parse_args()

    if args.create_tags:
        create_speech_bubble_data()

    if args.shrink_images:
        shrink_images()

    if args.train_model:
        train_model(args)
        # --train --gpus -1 --max_epochs 30 --precision 16 --benchmark True
