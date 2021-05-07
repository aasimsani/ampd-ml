import pytorch_lightning as pl
from argparse import ArgumentParser
from data.preprocessing.create_tags import create_speech_bubble_data
from training.train import train_model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--create_tags", action="store_true")

    args = parser.parse_args()

    if args.create_tags:
        create_speech_bubble_data()

    if args.train_model:
        train_model(args)
