from datasets import load_dataset, Audio
from transformers import pipeline

# Collection of different sounds of 5 seconds
dataset = load_dataset("ashraq/esc50", split="train[0:10]")

zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="./models/laion/clap-htsat-unfused"
)

# Cast dataset to correct sampling rate
dataset = dataset.cast_column( 
    "audio",
     Audio(sampling_rate=48_000)
)

candidate_labels = [
  "Sound of a dog",
  "Sound of vacuum cleaner",
  "Sound of a child crying",
  "Sound of a bird singing",
  "Sound of an airplane",
]

zero_shot_classifier(
  audio_sample["audio"]["array"],
  candidate_labels=candidate_labels
)
