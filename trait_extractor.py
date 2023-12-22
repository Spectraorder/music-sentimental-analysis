import librosa
import numpy as np
import pandas as pd
import os


def extract_musical_traits(filename):
    """Extracts musical traits from a WAV file and saves them to a CSV file.

    Args:
        filename (str): Path to the WAV file.

    Returns:
        None
    """

    y, sr = librosa.load(filename)

    # Extract features
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

    # Format features as strings
    tempo_str = f"{tempo:.1f}"
    spectral_centroid_value = spectral_centroid[0]  # Access the first value
    spectral_centroid_str = f"{spectral_centroid_value:.1f}"
    spectral_bandwidth_value = spectral_bandwidth[0]  # Access the first value
    spectral_bandwidth_str = f"{spectral_bandwidth_value:.1f}"
    chroma_stft_str = str(np.mean(chroma_stft_mean))
    zero_crossing_rate_value = zero_crossing_rate[0]  # Access the first value
    zero_crossing_rate_str = f"{zero_crossing_rate_value:.1}"

    # Create a CSV file
    traits = {
        "Tempo": tempo_str,
        "Spectral Centroid": spectral_centroid_str,
        "Spectral Bandwidth": spectral_bandwidth_str,
        "Chroma Features": chroma_stft_str,
        "Zero Crossing Rate": zero_crossing_rate_str
    }

    return traits


def process_audio_files(folder_path):
    """Processes multiple audio files in a folder and saves the results to a CSV file.

    Args:
        folder_path (str): Path to the folder containing the audio files.

    Returns:
        None
    """
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Process each audio file and append the results to the DataFrame
    for file in files:
        file_path = os.path.join(folder_path, file)
        traits = extract_musical_traits(file_path)
        result_df = result_df.append(traits, ignore_index=True)
        print(f"Processing file: {file}")

    # Save the combined results to a CSV file
    result_df.to_csv("musical_traits.csv", index=False)

    print("Combined Musical Traits:")
    print(result_df)


folder_path = '../Dataset/'
process_audio_files(folder_path)
