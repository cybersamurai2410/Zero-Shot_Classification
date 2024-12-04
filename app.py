import gradio as gr
from zs_audio import classify_audio
from zs_image import classify_image

audio_interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Label(),
    title="Zero-Shot Audio Classification",
    description="Classify audio into predefined categories without prior training."
)

image_interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(),
    title="Zero-Shot Image Classification",
    description="Classify an image into predefined categories using CLIP."
)

app = gr.TabbedInterface(
    interfaces=[audio_interface, image_interface],
    tab_names=["Audio Classification", "Image Classification"]
)

if __name__ == "__main__":
    app.launch()
