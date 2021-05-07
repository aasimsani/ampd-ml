import os
from .supporting_classes_and_functions import Page, get_leaf_panels
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd


def create_speech_bubble_data():
    """
    This function takes teh page metadata json files
    from data/datasets/page_metadata/ and harvests the
    information about the speech bubbles on each page.
    It then outputs this as a pandas readable CSV at
    data/datasets/speech_bubble_locations.csv
    """

    data_dir = "data/datasets/page_metadata/"
    image_dir = "data/datasets/page_images/"
    bubble_dir = "data/"
    tags_filename = "data/datasets/speech_bubble_locations.csv"

    data = {
        "file_id": [],
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": []
    }
    for filename in tqdm(os.listdir(data_dir)):
        page = Page()
        page.load_data(data_dir+filename)
        leaf_children = []
        get_leaf_panels(page, leaf_children)

        # Page width and height
        page_img = Image.open(image_dir+page.name+".png")
        p_w, p_h = page_img.size
        for child in leaf_children:
            if len(child.speech_bubbles) < 1:
                continue

            for bubble in child.speech_bubbles:
                x1y1 = bubble.location

                x1 = x1y1[0]
                y1 = x1y1[1]

                file_ref = page.name

                img = Image.open(bubble_dir+bubble.speech_bubble)

                w = img.size[0]
                h = img.size[1]

                # Since bubbles go through transformations make sure you
                # get the correct width and heights
                aspect_ratio = h/w
                new_height = round(np.sqrt(bubble.resize_to/aspect_ratio))
                new_width = round(new_height * aspect_ratio)

                x2 = x1 + new_height
                y2 = y1 + new_width

                # Convert tags to ratio of width and height
                x1 = x1/p_w
                y1 = y1/p_h
                x2 = x2/p_w
                y2 = y2/p_h

                data['file_id'].append(file_ref)
                data['x1'].append(x1)
                data['y1'].append(y1)
                data['x2'].append(x2)
                data['y2'].append(y2)

    df = pd.DataFrame(data)
    df.to_csv(tags_filename)
