def show_text2img():
    import streamlit as st
    import google.generativeai as genai
    import asyncio
    import requests
    import io
    from PIL import Image, UnidentifiedImageError

    # Configure the API key directly using Streamlit secrets
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    def query_stabilitydiff(prompt, headers):
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        return response.content, response.status_code

    def get_or_create_eventloop():
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

    loop = get_or_create_eventloop()

    def clear_chat_history():
        st.session_state.messages = []

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
            f"Generate a suggestive prompt of the following concept image: {user_input}. "
            "Make it short. "
            "Make sure to include elements such as colors, environment, mood, and specific objects. "
            "The description should be suitable for creating a high-quality, visually appealing image."
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

    def image_generation(prompt):
        headers = {"Authorization": f"Bearer {st.secrets.api_key}"}
        image_bytes, status_code = query_stabilitydiff(prompt, headers)

        try:
            image = Image.open(io.BytesIO(image_bytes))
            st.session_state.messages.append(
                {"role": "assistant", "content": "", "image": image, "prompt": prompt}
            )
            with st.chat_message("assistant"):
                st.image(image, caption=prompt, use_column_width=True)
        except UnidentifiedImageError:
            error_msg = image_bytes.decode("utf-8") if status_code != 200 else "Failed to generate image."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.write(error_msg)

    def main():
        st.set_page_config(page_title="Chatty The ChatBot", page_icon="🤖")

        st.title("Generate Image From Text🖼️")
        st.write("Welcome to the chat!")
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        use_prompt_generation = st.sidebar.checkbox('Generate Prompts for Image', value=False)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "image" in message:
                    st.image(message["image"], caption=message["prompt"], use_column_width=True)

        prompt = st.chat_input("Write your imagination")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            if use_prompt_generation:
                descriptive_prompt = generate_prompt(prompt)
                st.session_state["generated_prompt"] = descriptive_prompt
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Generated prompt: {descriptive_prompt}"})
                with st.chat_message("assistant"):
                    st.write(f"Generated prompt: {descriptive_prompt}")

            else:
                image_generation(prompt)

        if use_prompt_generation and st.button('Generate Image'):
            generated_prompt = st.session_state.get("generated_prompt", "")
            if generated_prompt:
                image_generation(generated_prompt)
            else:
                st.warning("No prompt generated. Please enter input and generate a prompt first.")

    if __name__ == "__main__":
        main()


