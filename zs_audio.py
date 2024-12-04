from datasets import load_dataset, Audio  
from transformers import pipeline        

# Initialize the zero-shot audio classification pipeline
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",  
    model="laion/clap-htsat-unfused"  
)

# Define the candidate labels for classification
candidate_labels = [
    "Sound of a dog",
    "Sound of vacuum cleaner",
    "Sound of a child crying",
    "Sound of a bird singing",
    "Sound of an airplane",
]

# Function to perform inference on a dataset
def audio_dataset_inference():
    # Load a dataset containing different 5-second sound clips
    dataset = load_dataset("ashraq/esc50", split="train[0:10]")
    
    # Ensure all audio samples in the dataset have the same sampling rate (48kHz)
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=48_000)
    )
    
    # Select the first audio sample from the dataset
    audio_sample = dataset[0]
    
    # Perform zero-shot classification on the selected audio sample
    result = zero_shot_classifier(
        audio_sample["audio"]["array"],  # Extract the audio array from the dataset sample
        candidate_labels=candidate_labels  # Pass the candidate labels for classification
    )
    print(result)

# Function to classify a single audio file
def classify_audio(audio_file):
    """
    Perform zero-shot classification on a single audio file.
    This function processes the input audio file, ensuring it has the correct sampling rate,
    and classifies it using the zero-shot classifier.
    
    Args:
        audio_file (str): Path to the audio file to classify.

    Returns:
        dict: Classification labels and their corresponding scores.
    """
    # Load and process the audio file to match the expected sampling rate (48kHz)
    audio_data = Audio(sampling_rate=48_000).decode_example(audio_file)
    
    # Perform zero-shot classification on the processed audio data
    result = zero_shot_classifier(
        audio_data["array"],  # Extract the audio array from the processed file
        candidate_labels=candidate_labels  # Pass the candidate labels for classification
    )
    
    # Return the classification results as a dictionary
    return {label['label']: label['score'] for label in result}
