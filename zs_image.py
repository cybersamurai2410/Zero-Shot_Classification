from transformers import CLIPModel, AutoProcessor
from PIL import Image

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
)
processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

labels = ["a photo of a cat", "a photo of a dog"]

image = Image.open("./cats.jpeg")
inputs = processor(
  text=labels,
  images=image,
  return_tensors="pt",
  padding=True
)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)[0]
probs = list(probs)
for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
