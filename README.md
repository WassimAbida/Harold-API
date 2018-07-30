# Harold-API

This project is the fruit of my internship within Harold Waste

It sum the deep research in deep learning models in order to perform a classification task over waste images.

Waste images, having large characteristics and many features, can only be classified using sophisticated models.

For that in this work we intend to use the Autoencoder theory, Restricted Boltzmann Machines as well as the possbility of transfer learning for pre-trained models to classify our waste images.



Access to labeled data is not always easy to do, for that two possibilities could be made:

- perform data augmentation over a certain data set 
- unsing the scraping method to download data from web 


models as Autoencoders, RBM and DBN (Deep Belief Network : a stack of RBM) are trained via a specific process: 
First, model is been trained with unsupervised data in order to learn how to match it's input to it's output, in other words to learn how to reconstruct it's input from the hidden code or so-called features.

Then the model is fine-tuned using labeled data


Finally model is tested on unseen data in order to measure its accuracy and efficency.



An small guide to understand the code developed : 


- Folder Data contains a first folder including waste images  (data in orginal form) and a second folder containing the data obtained via data augmentation method
- Folder Models contains two sub-folders :
	- Clustering : resumes the cluster model, inspired by k-means backed with sklearn library
	- Deep_Learning : includes python scripts of models developed during my internship, (FC Autoencoder, Convolutional Autoencoder, Denoising Autoencoder, RBM, DBN, Convolutional neural network, VGG16 backed with transfer learning...)

- Folder Application, contains graphs of trained model, h5 files for saving model's weights, json files for saving model's architectures and a python file detailing the training process and clustering results


- Folder Services, sums the utilities function used to develop models and to perform clustering and classification tasks






