from data.pytorch_dataset import MangaPanelDataset
import torch

def train_model():
    # Loading dataset
    labels_file = "data/datasets/speech_bubble_locations.csv"
    image_folder = "data/datasets/page_images/"
    train_dataset = MangaPanelDataset(
                                labels_file=labels_file,
                                image_folder=image_folder,
                                split=[0.0, 0.8]
                                )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=6
                                               )

