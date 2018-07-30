import os
from flask import request
from PIL import Image
import numpy as np

#global APP_ROOT


def test(x):
    print(x)

def data_to_images(APP_ROOT):
    """ This function performs a request for reading data from a repository

    and then transform the images into numpy arrays"""
    x_data = []
    target = os.path.join(APP_ROOT, 'images')
    print("our target = ", target)

    if not os.path.isdir(target):
        os.mkdir(target)
    print(len(request.files.getlist("my_file")))

    for file in request.files.getlist("my_file"):
        print(" file object  = ", file)
        filename = file.filename
        destination = "/".join([target, filename])
        print("path to image =", destination)

        file.save(destination)
        # taking inputs and saving 'em into tensor form
        img = Image.open(destination)

        # shape should be adjusted to model requieries  default (128,128)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.asarray(img)
        x_data.append(img)
        os.remove(destination)
    # my data tensor
    x_data = np.array(x_data)
    return x_data


def process_data(APP_ROOT):
    """ this function processes data and make them centered,

    it first uses the pre-defined function data_to_images to fulfill a request for reading images """

    x_data = data_to_images(APP_ROOT)
    x_data = (x_data - np.min(x_data, 0)) / (np.max(x_data, 0) + 0.0001)
    return x_data