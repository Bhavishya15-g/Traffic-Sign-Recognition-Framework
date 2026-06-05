import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from class_names import CLASS_NAMES

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Recognition Framework",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "traffic_sign_cnn.h5"
    )

    return model

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚦 Traffic Sign Recognition Framework")

st.markdown("""
### CNN-Based Traffic Sign Classification System

Upload one or more traffic sign images and the model
will identify which traffic sign is present.
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Traffic Sign Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_files:

    results = []

    st.subheader("Prediction Results")

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file)

        img = image.convert("RGB")

        # IMPORTANT:
        # Use 64x64 if your CNN was trained on 64x64 images.
        img = img.resize((64, 64))

        img_array = np.array(img)

        img_array = img_array.astype("float32") / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        ) * 100

        # Store Results

        results.append({
            "Image": uploaded_file.name,
            "Prediction": CLASS_NAMES[predicted_class],
            "Confidence (%)": round(confidence, 2)
        })

        # Display Image + Prediction

        col1, col2 = st.columns([1, 2])

        with col1:

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        with col2:

            st.success(
                f"Predicted Sign: {CLASS_NAMES[predicted_class]}"
            )

            st.info(
                f"Confidence Score: {confidence:.2f}%"
            )

            st.write("### Top 3 Predictions")

            top3 = np.argsort(
                prediction[0]
            )[-3:][::-1]

            for idx in top3:

                st.write(
                    f"• {CLASS_NAMES[idx]} "
                    f"({prediction[0][idx]*100:.2f}%)"
                )

        st.divider()

    # --------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------

    st.subheader("Prediction Summary")

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------------

    csv = results_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="traffic_sign_predictions.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    **Traffic Sign Recognition Framework**
    
    Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
    
    Model: Convolutional Neural Network (CNN)
    """
)
