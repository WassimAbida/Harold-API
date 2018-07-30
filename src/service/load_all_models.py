import os
from keras.models import model_from_json


def load_all_models(Path_to_models):

    """ This function takes as input the path to your saved deep learning models

    Path should contain a folder for each model, and each model need to have a
     .h5 file for the weights and a json file for the architecture

    output is a dictionnary containing the names of the models and paths to their files

    """
    all_models = dict()
    paths = []
    for root, dirs, files in os.walk(Path_to_models):
        files_to_upload = []
        for filename in files:
            files_to_upload.append(filename)
        if len(files_to_upload) != 0:
            paths.append(files_to_upload)

    models_availables = [x[1] for x in os.walk(Path_to_models)][0]

    for i in range(len(models_availables)):
        all_models.update({models_availables[i]: {paths[i][0], paths[i][1]}})

    return all_models


def upload_model_weights(path_to_model='ConvAE_model.json', path_to_weights="ConvAE_model.h5"):
    """ This function takes as input paths to weights and architecture of a model

    Output is an instance of the model
    """
    """ This function takes as input the path to the model architecture and weights"""
    # load json and create model

    my_json_file = open(path_to_model, 'r')
    loaded_model_json = my_json_file.read()
    my_json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path_to_weights)

    print("Loaded  model and weights from disk")

    return model