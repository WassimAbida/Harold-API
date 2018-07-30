import os
from flask import Flask, render_template, request, jsonify
import codecs, json
import tensorflow as tf
from src.service.load_all_models import load_all_models, upload_model_weights
from src.service.prediction import predict_function

from src.service.image_processing import process_data


graph = tf.get_default_graph()
Path_to_models = '/home/abida/Desktop/empty_project/src/models/'
app = Flask(__name__)
global  APP_ROOT,model
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


with open('dictionnary.json') as json_file:
    dictionnary = json.load(json_file)

@app.route('/')
def index():
    return render_template("predict.html")


@app.route("/predict", methods = ['POST'])
def predict():
    """ This function upload data from computer and send'em to the server for treatment
    It currently saves'em in a directory within the app folder """

# processing data
    if request.method == "POST":
        x_data= process_data(APP_ROOT)

        print('saving data ended ')
    # upload model and weights

        print("  loaded model architecture : ")
        model.summary()
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("number of layers ",len(model.layers))
        print(" input data shape  ", x_data.shape)

        class_predicted=[]
        for i in range(len(x_data)):

        # For a single Image
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            image = x_data[i].reshape(1, x_data[i].shape[0], x_data[i].shape[1], x_data[i].shape[2])
            with graph.as_default():
                print("Probabilities of the loaded model : ",model.predict(image))
                prediction = predict_function(model, image,dictionnary)
            print('Prediction of the model : ', prediction, '\n')
            class_predicted.append(prediction[0][0])

        for i in range(len(class_predicted)):
            print('Class', class_predicted[i][0], ' has a majority with pourcentage', class_predicted[i][1] * 100, '%')
        print( 'Label of your input image', class_predicted)



    # preparing the json output
        data = {"success_classification": False}
        data["predictions"] = []
    # loop over the results and add them to the list of
    # returned predictions
        for (label, prob) in class_predicted:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)
    # indicate that the request was a success
        data["success_classification"] = True
    return jsonify(data)


if __name__ == "__main__":

    Dictio_models = load_all_models(Path_to_models)
    print("deep learning models available", list(Dictio_models.keys()))

    model_choice = input('Choisir le modèle  : ')
    paths = list(Dictio_models[model_choice])
    print(paths)
    if (paths[1]).endswith('h5'):

        path_to_weights, path_to_model = Path_to_models + model_choice + '/' + paths[
            1], Path_to_models + model_choice + '/' + paths[0]
    else:
        path_to_weights, path_to_model = Path_to_models + model_choice + '/' + paths[
            0], Path_to_models + model_choice + '/' + paths[1]

    model = upload_model_weights(path_to_model=path_to_model, path_to_weights=path_to_weights)
    app.run(port=4555,debug=True)

# test.com/upload




















