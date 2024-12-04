from transformers import CLIPModel, AutoProcessor
from PIL import Image

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
)
processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

labels = ["a photo of a cat", "a photo of a dog"]

def classify_image(image_path):
    """
    Perform zero-shot classification on a single image.
    
    Args:
        image_path (str): Path to the image.
    
    Returns:
        dict: Classification probabilities for the image.
    """
    
    image = Image.open(image_path) # Open the image
    
    # Preprocess the image and labels
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Perform inference using the CLIP model
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]  # Calculate probabilities
    
    # Return results as a dictionary with label and probability pairs
    return {labels[i]: probs[i].item() for i in range(len(labels))}
