import streamlit as st
import shutil
import os
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os
import pickle
import streamlit as st
from PIL import Image
import io
# Importing the Libraries

 # load doc into memory
file = open("shonaa.txt","r")

import re

# Load the model and tokenizer

model = load_model('bestbidirectionalstm.h5')
tokenizer = pickle.load(open('token.pk1', 'rb'))




def main():

    """Object detection App"""

    st.title("Next Word Prediction App in Shona")

    html_temp = """
    <body style="background-color:green;">
    <div style="background-color:red ;padding:10px">
    <h2 style="color:white;text-align:center;">Izwi Rinotevera</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title(" Get the Predictions")
    seed_text = st.text_input("nyora manzwi mashanu: ")

    if seed_text is not None:

        try:
            next_words = 1
            suggested_word = []
            # temp = seed_text
            for _ in range(next_words ):
                # Tokenize and pad the text
                sequence = tokenizer.texts_to_sequences([seed_text])[0]
                sequence = pad_sequences([sequence], maxlen=5, padding='pre')

                # Predict the next word
                predicted_probs = model.predict(sequence, verbose=0)
                predicted = np.argmax(predicted_probs, axis=-1)

                # Convert the predicted word index to a word
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break

                # Append the predicted word to the text
                #seed_text += " " + output_word

            #return ' '.join(text.split(' ')[-next_words :])

            seed_text += " " + output_word
            print("Suggested next  word  : ", suggested_word)

           # print(seed_text)
        except Exception as e:
            print("Error occurred: ", e)


    if st.button("Suggested_words"):
        st.success(seed_text)


if __name__ == '__main__':
    main()
