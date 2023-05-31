import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 
import streamlit as st
import pickle
import tensorflow as tf
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Rice-leaf Disease Detection')
st.text('Upload the image')



# model = pickle.load(open('my_model.pkl','rb'))
model=tf.keras.models.load_model('my_model1')


uploaded_file= st.file_uploader("Choose an image",type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image')

if st.button('Predict'):
    CATEGORIES =['BrownSpot','Healthy','Hispa','LeafBlast']
    st.write('Result')
    flat_data=[]
    img = np.array(img)
    img_resize = cv2.resize(img,(224,224))
    img_scaled = img_resize/255
    img_reshaped = np.reshape(img_scaled,[1,224,224,3])
    input_pred = model.predict(img_reshaped)
    input_label = np.argmax(input_pred)
    print(input_label)
    if input_label == 0:
        st.write("HISPA")
    elif input_label == 1:
        st.write("Healthy")
    elif input_label == 2:
        st.write("BrownSpot")
    elif input_label== 3:
        st.write('LeafBlast')
    # img_resized = resize(img,(150,150,3))
    # flat_data.append(img_resized.flatten())
    # flat_data= np.array(flat_data)
    # print(img.shape)
    # plt.imshow(img_resized)
    # y_out = model.predict(flat_data)
    # y_out = CATEGORIES[y_out[0]]
    # st.write(f'output: {input_label}')
