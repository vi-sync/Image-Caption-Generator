try:
    import pickle
    from tensorflow import keras
    # from tensorflow.keras.models import Sequential, model_from_json
    # loading
    import time
    from tensorflow.keras import models
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    #import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.applications.inception_v3 import  preprocess_input   #InceptionV3,
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union

    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

#Files importer
with open (r"idx_word_dict.pkl" ,'rb') as file:
  idx_word_dict = pickle.load(file)

with open (r"C:\Users\jocke\\3D Objects\image caption\word_idx_dict.pkl" ,'rb') as file:
  word_idx_dict = pickle.load(file)

with open (r'C:\Users\jocke\\3D Objects\image caption\max_len.pkl' ,'rb') as file:
  max_len = pickle.load(file)

with open (r'C:\Users\jocke\\3D Objects\image caption\caption_tokenizer.pkl' ,'rb') as file:
  caption_tokenizer = pickle.load(file)


with open(r'C:\Users\jocke\\3D Objects\image caption\image_features','rb') as file:
    image_features = pickle.load(file)


json_file = open(r"C:\Users\jocke\\3D Objects\image caption\model.json", 'r')
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)
# load weights into new model
model.load_weights(r"C:\Users\jocke\\3D Objects\image caption\model.h5")
print("Loaded model from disk")

# v3 model
# load json and create model

json_file = open(r"C:\Users\jocke\\3D Objects\image caption\image_encoder_model.json", 'r')
image_encoder_model_json = json_file.read()
json_file.close()
image_encoder_model = models.model_from_json(image_encoder_model_json)
image_encoder_model.load_weights(r"C:\Users\jocke\\3D Objects\image caption\image_encoder_model.h5")


#########################################################

def add_bg_from_url():   #for the background video/gif
    st.markdown(
        f"""
         <style>
         .stApp {{

             background-image: url('https://i.gifer.com/1pX9.gif');
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


def predict_captions(image_features, idx_word_dict, word_idx_dict):
    image_features = np.expand_dims(image_features, axis=0)
    #st.balloons()
    start_txt = 'startseq'
    predicted_caption = ''
    st.text('Working....')
    for idx in range(max_len):
        seq = caption_tokenizer.texts_to_sequences([start_txt])[0]
        paded_seq = pad_sequences([seq], max_len, padding='post')[0]
        paded_seq = np.expand_dims(a=paded_seq, axis=0)
        model_prediction = model.predict([image_features, paded_seq], verbose=0)

        predicted_word_idx = np.argmax(model_prediction)
        if predicted_word_idx == word_idx_dict['endseq']:
            return predicted_caption

        predicted_word = idx_word_dict[predicted_word_idx]

        predicted_caption = predicted_caption + ' ' + predicted_word
        start_txt = predicted_caption.split(' ')
    #st.balloons()
    st.text('Almost Done......')
    return predicted_caption


#https://www.youtube.com/watch?v=Uh_2F6ENjHs&ab_channel=SoumilShah
def main():
    add_bg_from_url()
    st.title("""Image Caption Generator""")
    st.text("This mode is trained on a very small dataset . Its performance can be bad and unpredictable on not so common pictures")

    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg"]))
        return

    if isinstance(file, BytesIO):
        st.write('You have choosed this image ')
        st.image(file)
        img = load_img(file, target_size=(299, 299))
        img = img_to_array(img)  # (299, 299, 3)
        img = np.expand_dims(img, axis=0)  # (1,299, 299, 3) batches of 1
        img = preprocess_input(img)
        if st.button('Click me to know the predicted caption '):
            st.balloons()
            prediction = image_encoder_model.predict(img, verbose=0)  # gives (1, 2048)
            st.text(
                'Working...'
            )
            image_features = np.reshape(prediction, prediction.shape[1])
            # predicting caption
            caption = predict_captions(image_features, idx_word_dict, word_idx_dict)
            st.balloons()
            st.snow()
            st.text('''   
                                The predicted caption is  ''')
            st.success(caption)


    file.close()
main()