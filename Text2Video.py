#pip install --upgrade diffusers
import streamlit as st
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Set device and dtype
device = "cuda"
dtype = torch.float16

# Model parameters
step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

# Initialize Streamlit interface
st.title("AnimateDiff with Streamlit")
prompt = st.text_input("Enter prompt", "A girl smiling")
guidance_scale = st.slider("Guidance Scale", 0.1, 10.0, 1.0)
num_inference_steps = st.slider("Number of Inference Steps", 1, 8, step)

if st.button("Generate Animation"):
    # Load adapter and pipeline
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    # Generate animation
    with st.spinner("Generating animation..."):
        output = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        gif_path = "animation.gif"
        export_to_gif(output.frames[0], gif_path)

    # Display the result
    st.image(gif_path, caption="Generated Animation", use_column_width=True)

if st.button("Download GIF"):
    with open(gif_path, "rb") as f:
        btn = st.download_button(
            label="Download animation.gif",
            data=f,
            file_name="animation.gif",
            mime="image/gif"
        )

