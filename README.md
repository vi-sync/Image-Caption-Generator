# Image-Caption-Generator
This project aims to generate captions for images using an encoder-decoder architecture in the training phase. The encoder receives an image and its caption, and the image is converted into a vector of shape (None, 2048) using the Inceptionv3 model. During inference, only the image is passed to the encoder, and the model predicts its caption based on its learning.

The decoder then receives a combined vector of the image and its caption, which is vectorized using an embedding layer. In the decoder, this vector is passed through a dense layer using softmax as the activation function with units being equal to the vocabulary size of the dataset.

During training, a custom data generator is used to generate data on the go. The reason for using a data generator is that it helps in generating data on the fly, which is essential for training deep learning models with large datasets that may not fit in memory. The data generator generates batches of data during each iteration of the training process and applies data augmentation techniques to increase the diversity of the training data. The overall combination of the image and each word of the caption is taken as a datapoint for training the model.
# TO-DO
Add attention layer to further increase the learning capability of the model and produce much better results . Currently its's not very good( Less data might be also one of the reasons
