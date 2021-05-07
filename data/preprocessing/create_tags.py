import os
from .supporting_classes_and_functions import Page, get_leaf_panels

def create_speech_bubble_data():

    data_dir = "data/datasets/page_metadata/"

    for filename in os.listdir(data_dir)[0:1]:
        print(filename)
