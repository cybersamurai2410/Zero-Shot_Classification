import gradio as gr
from zs_audio import classify_audio
from zs_image import classify_image

audio_interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs=gr.Label(),
    title="Zero-Shot Audio Classification",
    description="Classify audio into predefined categories without prior training.",
    allow_flagging="never",
)

image_interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(),
    title="Zero-Shot Image Classification",
    description="Classify an image into predefined categories using CLIP.",
    allow_flagging="never",
)

app = gr.TabbedInterface(
    [audio_interface, image_interface],
    ["Audio Classification", "Image Classification"]
)

app.launch()
