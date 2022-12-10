import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Image Classifier_APP")



st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./my_model.h5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [256, 256])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Leaf Classifier detection')

file = st.file_uploader("Upload an image ", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)

