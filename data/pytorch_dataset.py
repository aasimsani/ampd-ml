import os
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToTensor


class MangaPanelDataset(object):
    """
    Manga dataset class which will serve
    the a Manga panel with a it's speech bubble metadata

    :param labels_file: CSV filepath with the file references and
    the speech bubble data

    :type labels_file: str

    :param image_folder: Path to the images

    :type image_folder: str

    :param split: How to split the dataset. A set of two floats between 0.0
    and 1.0, a start and an end percentage.

    :type split: list

    :param transforms: Whether or not to transform images, defaults to False

    :type transforms: bool, optional
    """    

    def __init__(self, labels_file, image_folder, split, transforms=False):
        """
        Constructor method
        """        

        df = pd.read_csv(labels_file)

        start = round(split[0]*len(df))
        end = round(split[1]*len(df))

        self.labels = df.iloc[start:end]
        self.image_folder = image_folder
        self.images = df['file_id'].unique()

        self.transforms = Compose([ToTensor()])
    
    def __getitem__(self, idx):
        """
        Provide an image with a set of targets for the data loader
        to process and serve to the model.

        :param idx: Index of desired image/target paid

        :type idx: int

        :return: Tuple of image and target

        :rtype: tuple
        """        

        # Get the set of bubbles that coresspond to this image
        file_id = self.images[idx] 
        speech_bubbles = self.labels[self.labels['file_id'] == file_id]

        num_objs = len(speech_bubbles)
        
        boxes = []
        # Get the bubble coordinates
        for bubble in speech_bubbles.iterrows():
            bubble = bubble[1]
            x1 = bubble['x1']
            y1 = bubble['y1']

            x2 = bubble['x2']
            y2 = bubble['y2']
            box = [x1, y1, x2, y2]
            boxes.append(box)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}

        target["boxes"] = boxes
        target["labels"] = boxes
        target["image_id"] = file_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img_path = os.path.join(self.image_folder, file_id+".png")
        img = Image.open(img_path)
        img = self.transforms(img)

        return img, target
    
    def __len__(self):

        return len(self.images)
    