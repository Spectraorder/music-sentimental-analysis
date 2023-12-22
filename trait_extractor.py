import librosa
import numpy as np
import pandas as pd


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
    tempo_str = f"{tempo:.1f} BPM"
    spectral_centroid_value = spectral_centroid[0]  # Access the first value
    spectral_centroid_str = f"{spectral_centroid_value:.1f} Hz"
    spectral_bandwidth_value = spectral_bandwidth[0]  # Access the first value
    spectral_bandwidth_str = f"{spectral_bandwidth_value:.1f} Hz"
    chroma_stft_str = str(np.mean(chroma_stft_mean))
    zero_crossing_rate_value = zero_crossing_rate[0]  # Access the first value
    zero_crossing_rate_str = f"{zero_crossing_rate_value:.1%}"

    # Create a CSV file
    data = {
        "Tempo": tempo_str,
        "Spectral Centroid": spectral_centroid_str,
        "Spectral Bandwidth": spectral_bandwidth_str,
        "Chroma Features": chroma_stft_str,
        "Zero Crossing Rate": zero_crossing_rate_str
        # "Loudness": loudness_str
    }
    print("Extracted Musical Traits:")
    print(f"Tempo: {tempo_str}")
    print(f"Spectral Centroid: {spectral_centroid_str}")
    print(f"Spectral Bandwidth: {spectral_bandwidth_str}")
    print(f"Chroma Features: {chroma_stft_str}")
    print(f"Zero Crossing Rate: {zero_crossing_rate_str}")

    df = pd.DataFrame(data, index=[0])
    df.to_csv("musical_traits.csv", index=False)


# Example usage
filename = '../Dataset/blues.00000.wav'
extract_musical_traits(filename)
