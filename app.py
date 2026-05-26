import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import pandas as pd

st.set_page_config(page_title="AI皮肤辅助筛查", page_icon="🏥", layout="wide")
st.title("🏥 智能皮肤皮损辅助筛查系统")
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('melanoma_efficientnet_best.h5', compile=False)

model = load_model()
uploaded_file = st.file_uploader("上传皮损照片", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=250)
    if st.button("启动分析"):
        arr = np.expand_dims(np.array(image.resize((224, 224)))/255.0, axis=0)
        score = float(model.predict(arr, verbose=0)[0][0])
        st.write(f"恶性概率: {round(score*100, 1)}%")
