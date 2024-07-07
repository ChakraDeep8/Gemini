def gemini_text2image():
    import streamlit as st
    from PIL import Image, UnidentifiedImageError
    import io
    import requests
    import google.generativeai as genai

    # Configure the API key directly using Streamlit secrets
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    # Initialize session_state_history if it doesn't exist
    if 'session_state_history' not in st.session_state:
        st.session_state.session_state_history = []

    # Function to query the Stability Diffusion API
    def query_stabilitydiff(prompt, headers):
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        return response.content, response.status_code

    # Function to clear chat history in the session state
    def clear_chat_history():
        st.session_state.session_state_history = []

    # Function to generate a prompt using Google's Generative AI
    def generate_prompt(user_input):
        generation_configure = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 1000,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_configure,
            safety_settings=safety_settings
        )

        structured_prompt = (
            f"Create a visually appealing image based of {user_input}.",
            st.secrets["prompt"]
        )

        response = model.generate_content(structured_prompt)
        if response and response.candidates:
            output_text = response.candidates[0].content.parts[0].text if response.candidates[
                0].content.parts else "No content parts found."
            # Ensure output text is valid and clean
            output_text = "".join([char for char in output_text if char.isprintable()])
        else:
            output_text = "No response generated."

        return output_text

    # Function to generate an image from a prompt
    def image_generation(input_prompt):
        headers = {"Authorization": f"Bearer {st.secrets.api_key}"}
        image_bytes, status_code = query_stabilitydiff(input_prompt, headers)

        try:
            image = Image.open(io.BytesIO(image_bytes))
            st.session_state.session_state_history.append(
                {"role": "assistant", "content": f"Generated image based on prompt: {input_prompt}", "image": image}
            )
            with st.chat_message("assistant"):
                st.image(image, caption=input_prompt, use_column_width=True)
        except (UnidentifiedImageError, IOError) as e:
            error_msg = str(e) if status_code != 200 else "Failed to generate image."
            st.session_state.session_state_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.write(error_msg)

    # Set up the Streamlit page configuration
    st.title("Generate Image From Text🤔")

    # Sidebar options
    st.sidebar.markdown("Use this option to generate descriptive prompt 👇")
    use_prompt_generation = st.sidebar.checkbox('Enhance Image', value=False)
    if st.sidebar.button('Clear Chat History', on_click=clear_chat_history):
        st.session_state.session_state_history = []

    # Display existing chat history
    for message in st.session_state.session_state_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image" in message:
                caption = message.get("prompt", "No prompt available")
                st.image(message["image"], caption=caption, use_column_width=True)

    # Get user input
    prompt = st.chat_input("Write your imagination")

    if prompt:
        st.session_state.session_state_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(f"You: {prompt}")

        if use_prompt_generation:
            descriptive_prompt = generate_prompt(prompt)
            st.session_state.session_state_history.append(
                {"role": "assistant", "content": f"Generated prompt: {descriptive_prompt}"})
            with st.chat_message("assistant"):
                with st.spinner('Generating prompt...'):
                    st.write(f"Generated prompt: {descriptive_prompt}")
            with st.spinner('Generating image...'):
                image_generation(prompt)
        else:
            with st.spinner('Generating image...'):
                image_generation(prompt)
