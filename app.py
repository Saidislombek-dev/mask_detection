import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import cvlib as cv
import numpy as np

st.title("Mask Detection")

flip = st.checkbox("Flip")

model = load_model("mask.h5")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:] if flip else img

    face,confindence = cv.detect_face(flipped)
    for index,yuz in enumerate(face):
        (startX,startY,endX,endY) = yuz[0],yuz[1],yuz[2],yuz[3]
        
        yuz_np = np.copy(flipped[startY:endY,startX:endX])
        if yuz_np.shape[0] < 10 or yuz_np.shape[1] < 10:
            continue
        yuz_np = cv2.resize(yuz_np,(96,96))
        yuz_np = img_to_array(yuz_np)
        yuz_np = np.expand_dims(yuz_np,axis=0)
        
        bashorat = model.predict(yuz_np)[0][0]
        
        if round(bashorat)==1:
            color = (0,255,0)
            label = "Mask"
        else:
            color = (0,0,255)
            label = "No mask"
            bashorat = 1 - bashorat
            
        label = f"{label} {np.around(bashorat*100,2)}"
        
        if startY-10 > 10:
            Y=startY-10
        else:
            Y=startY+10
        
        cv2.rectangle(flipped,(startX,startY),(endX,endY),color,2)
        cv2.putText(flipped,label,(startX,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
