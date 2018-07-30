
from src.service.decoding_prediction import probabilities_to_labels
import tensorflow as tf
graph = tf.get_default_graph()


def predict_function(model,image,dictionnary):
    """ This function takes into input a pre-trained model, and returns the prediction of an image by the model"""
    with graph.as_default():
        y_tilde = model.predict(image)
    prediction = probabilities_to_labels(y_tilde, dictionnary)

    return prediction


