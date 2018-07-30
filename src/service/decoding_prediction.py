import numpy as np
def get_key(ss_value, dictionnary):
    """ This function returns the key in dict of a value donated
    example :
    Dict = { 'first' :1 , 'second':2}

    get_key(1,Dict) = 'first'
    get_key(2,Dict) = 'second'
    """
    lkey = [key for key, value in dictionnary.items() if value == ss_value][0]

    return lkey


def probabilities_to_labels(preds, dictionnary, top=3):
    """ This function decodes the output of a model applied and adapted to a given dictionnary

    Input :
            preds:  predictions : output of the model

            dictionnary : the dictionnary linking classes to labels

            top: first top-element to take into account in display

            top = 3 'default'

    Output:

         results : probability distribution given by model for each test sample

         y_pred : major predicted labels with higher probability


    """
    if len(preds.shape) != 2 or preds.shape[1] != len(dictionnary):
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, Num_CLASSES)). '
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = []
        for i in range(len(top_indices)):
            result.append([get_key(top_indices[i], dictionnary), (pred[top_indices[i]])])

        results.append(result)


    return results