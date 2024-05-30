import streamlit as st
from PIL import Image
import google.generativeai as genai
import asyncio

# Configure the API key directly using Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# Function to get response from the Gemini model
def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text and image:
        response = model.generate_content([input_text, image])
    elif input_text:
        response = model.generate_content(input_text)
    elif image:
        response = model.generate_content(image)
    else:
        response = "No input provided."

    # Extract the text from the response
    if response and response.candidates:
        output_text = response.candidates[0].content.parts[0].text
    else:
        output_text = "No response generated."

    return output_text


def clear_chat_history():
    st.session_state.messages = []


# Initialize the Streamlit app
def main():
    st.set_page_config(page_title="Gemini Image Chat", page_icon="🤖")
    st.header("Chat with Image")

    # Input prompt
    input_text = st.chat_input("Tell me about the image", key="input")

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
    if submit and (input_text or image):
        with st.spinner("Thinking..."):
            response = get_gemini_response(input_text, image)
            st.subheader("Answer 👇")
            st.write(response)

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Display chat messages and bot response
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if input_text or uploaded_file:
        if submit:
            st.session_state.messages.append(
                {"role": "user", "content": input_text if input_text else "Uploaded image"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)


if __name__ == "__main__":
    main()
