from datasets import load_dataset, Audio  
from transformers import pipeline   
import torchaudio
import numpy as np     

# Initialize the zero-shot audio classification pipeline
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",  
    model="laion/clap-htsat-unfused"  
)

# Define the candidate labels for classification
candidate_labels = [
    "Sound of a dog barking",
    "Sound of car driving",
    "Sound of a person talking",
    "Sound of a bird singing",
    "Sound of a plane flying",
]

# Function to perform inference on a dataset
def audio_dataset_inference():
    # Load a dataset containing different 5-second sound clips
    dataset = load_dataset("ashraq/esc50", split="train[0:10]")
    
    # Ensure all audio samples in the dataset have the same sampling rate (48kHz)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=48_000))
    
    # Select the first audio sample from the dataset
    audio_sample = dataset[0]
    
    # Perform zero-shot classification on the selected audio sample
    result = zero_shot_classifier(
        audio_sample["audio"]["array"],  # Extract the audio array from the dataset sample
        candidate_labels=candidate_labels  # Pass the candidate labels for classification
    )
    print(result)

def classify_audio(audio_file):
    """
    Perform zero-shot classification on a single audio file.
    
    Args:
        audio_file (str): Path to the audio file to classify.

    Returns:
        dict: Classification labels and their corresponding scores.
    """
    try:
        # Load audio file using torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Resample audio to 48kHz (if necessary)
        if sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
            waveform = resampler(waveform)
        
        # Convert waveform to NumPy array
        audio_array = waveform.squeeze().numpy()

        # Perform zero-shot classification
        result = zero_shot_classifier(
            audio_array,  # Pass the audio array
            candidate_labels=candidate_labels
        )
        return {label['label']: label['score'] for label in result}
    except Exception as e:
        print(f"Error in classify_audio: {e}")
        return {"Error": str(e)}
        
