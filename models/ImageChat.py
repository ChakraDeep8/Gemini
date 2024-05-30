def show_image():
    import streamlit as st
    from PIL import Image
    import google.generativeai as genai
    import asyncio
    import io

    # Configure the API key directly using Streamlit secrets
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    def get_or_create_eventloop():
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

    loop = get_or_create_eventloop()

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
            return "No input provided."

        # Extract the text from the response
        if response and response.candidates:
            output_text = response.candidates[0].content.parts[0].text if response.candidates[
                0].content.parts else "No content parts found."
        else:
            output_text = "No response generated."

        return output_text

    def clear_chat_history():
        st.session_state.messages = []
        if 'uploaded_image' in st.session_state:
            del st.session_state.uploaded_image

    # Initialize the Streamlit app
    #st.set_page_config(page_title="Gemini Image Chat", page_icon="🤖")
    st.header("Chat with Image")

    # Sidebar for image upload
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.format == 'WEBP':
                image = image.convert("RGB")
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image.", use_column_width=True)
        elif 'uploaded_image' in st.session_state:
            image = st.session_state.uploaded_image
            st.image(image, caption="Uploaded Image.", use_column_width=True)
        else:
            image = None

        st.button('Clear Chat History', on_click=clear_chat_history)

    # Input prompt
    input_text = st.chat_input("Tell me about the image")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Generate response when submit button is clicked
    if (st.button("Tell me about the image") and image) or (input_text and image):
        with st.spinner("Thinking..."):
            response = get_gemini_response(input_text, image)

            # Add user message to chat history
            user_message = input_text if input_text else "Uploaded image"
            st.session_state.messages.append({"role": "user", "content": user_message})

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat messages and bot response
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


