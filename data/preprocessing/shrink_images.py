import os
from PIL import Image
import concurrent
from tqdm import tqdm


def shrink_single_image(data):
    """
    Wrapper function for image shrinking
    used to run operation conrurrently

    :param data: Filepaths and filenames to read and write

    :type data: tuple

    """

    img_dir = data[0]
    out_dir = data[1]
    filename = data[2]
    shrink_factor = data[3]

    # Open and resize image
    img = Image.open(img_dir+filename).convert("L")
    nw, nh = round(img.size[0]/shrink_factor), round(img.size[1]/shrink_factor)
    img = img.resize((nw, nh))

    img.save(out_dir + filename)


def shrink_images():
    """
    Shrink image size to enable faster training
    """

    shrink_factor = 16

    img_dir = "data/datasets/page_images/"
    out_dir = "data/datasets/page_images_shrunk/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    files = [
        (img_dir, out_dir, filename, shrink_factor)
        for filename in os.listdir(img_dir)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(shrink_single_image, files),
                       total=len(files)))
