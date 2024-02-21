import keras
import h5py
import cv2
import numpy as np
import streamlit as st
import json


page_bg_img ="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://cordis.europa.eu/docs/news/images/2021-12/435395.jpg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

st.title('Human emotion detection web application')
st.header('angry,disgusted,fear,happy,neutral and sad')
st.sidebar.header("""More features coming soon!""")

choice = st.selectbox('pick anyone to start',['Classify Image','Real Time Classify'])
if choice=='Classify Image':
    try:
       upload = st.file_uploader('upload the image',type=['png','jpeg','jpg'])
       file_bytes = np.asarray(bytearray(upload.read()), np.uint8)
       opencv_image = cv2.imdecode(file_bytes, 1)

       mod1 = keras.models.load_model("am.h5")

       img1 = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)

       data = np.ndarray(shape=(1, 80,80), dtype=np.float32)
       img1 = cv2.resize(img1, (80,80))

       image_array = np.asarray(img1)

       normalized_image_array = (image_array.astype(np.float32) / 255.0)

       data[0] = normalized_image_array

       pred = np.asarray(mod1.predict(data))

       font = cv2.FONT_HERSHEY_SIMPLEX

       org = (14,40)

       fontScale = 1

       color_yellow = (255,255,0)
       color_red = (255, 0, 0)
       color_blue = (0, 0, 255)
       color_green = (0, 255, 0)
       color_teal = (0, 255, 255)
       color_purple = (180, 0, 255)

       thickness = 2

       if np.argmax(pred)==0:
          image = cv2.putText(opencv_image, 'Angry', org, font,
                              fontScale, color_red, thickness, cv2.LINE_AA)
          st.image(image)
       elif np.argmax(pred)==1:
          image = cv2.putText(opencv_image, 'Fear', org, font,
                              fontScale, color_blue, thickness, cv2.LINE_AA)
          st.image(image)
       elif np.argmax(pred)==2:
          image = cv2.putText(opencv_image, 'Happy', org, font,
                              fontScale, color_yellow, thickness, cv2.LINE_AA)
          st.image(image)
       elif np.argmax(pred)==3:
          image = cv2.putText(opencv_image, 'Neutral', org, font,
                              fontScale, color_purple, thickness, cv2.LINE_AA)
          st.image(image)
       elif np.argmax(pred)==4:
          image = cv2.putText(opencv_image, 'Sad', org, font,
                              fontScale, color_teal, thickness, cv2.LINE_AA)
          st.image(image)
    except AttributeError:
        st.write('please provide with a photo')

elif choice=='Real Time Classify':
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = np.ndarray(shape=(1, 80,80), dtype=np.float32)
        img1 = cv2.resize(frame, (80,80))

        image_array = np.asarray(img1)

        normalized_image_array = (image_array.astype(np.float32) / 255.0)

        data[0] = normalized_image_array

        mod2 = keras.models.load_model("am.h5")

        pred = np.asarray(mod2.predict(data))

        font = cv2.FONT_HERSHEY_SIMPLEX

        org = (14, 40)

        fontScale = 1

        color_yellow = (255, 255, 0)
        color_red = (255, 0, 0)
        color_blue = (0, 0, 255)
        color_green = (0, 255, 0)
        color_teal = (0, 255, 255)
        color_purple = (180, 0, 255)

        thickness = 2

        if np.argmax(pred) == 0:
            image = cv2.putText(f, 'Angry', org, font,
                                fontScale, color_red, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(image)
        elif np.argmax(pred) == 1:
            image = cv2.putText(f, 'Fear', org, font,
                                fontScale, color_blue, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(image)
        elif np.argmax(pred) == 2:
            image = cv2.putText(f, 'Happy', org, font,
                                fontScale, color_yellow, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(image)
        elif np.argmax(pred) == 3:
            image = cv2.putText(f, 'Neutral', org, font,
                                fontScale, color_purple, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(image)
        elif np.argmax(pred) == 4:
            image = cv2.putText(f, 'Sad', org, font,
                                fontScale, color_teal, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(image)
    else:
        st.write('Stopped')
else:
    st.write('...')
