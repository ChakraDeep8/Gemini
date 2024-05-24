import streamlit as st
from PIL import Image
import google.generativeai as genai
import time

# Configure the Generative AI model
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Function to get response from the Gemini model
def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text:
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# Initialize the Streamlit app
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

# Input prompt
input_text = st.text_input("Input Prompt:", key="input")

# Sidebar for image upload
with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.format == 'WEBP':
            image = image.convert("RGB")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Tell me about the image")

# Generate response when submit button is clicked
if submit and image:
    response = get_gemini_response(input_text, image)
    st.subheader("Answer 👇")
    st.write(response)
