import streamlit as st
import pandas as pd
from pathlib import Path
import json
from pydub import AudioSegment
import os

def load_dataset(json_path: str) -> pd.DataFrame:
    """Load alignment data into a pandas DataFrame."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data["segments"])
    df["audio_file"] = data["audio_file"]
    return df

def extract_audio_segment(audio_path: str, start: float, end: float, output_dir: str = "temp") -> str:
    """Extract a segment from an audio file and save as wav.
    
    Args:
        audio_path: Path to audio file
        start: Start time in seconds
        end: End time in seconds
        output_dir: Directory to save wav files
        
    Returns:
        Path to wav file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path
    segment_name = f"segment_{start:.2f}_{end:.2f}.wav"
    output_path = os.path.join(output_dir, segment_name)
    
    # Only create segment if it doesn't exist
    if not os.path.exists(output_path):
        audio = AudioSegment.from_file(audio_path, format="opus")
        segment = audio[start * 1000:end * 1000]  # pydub works in milliseconds
        segment.export(output_path, format="wav")
    
    return output_path

def create_audio_player(wav_path: str) -> str:
    """Create an HTML audio player element.
    
    Args:
        wav_path: Path to wav file
        
    Returns:
        HTML string for audio player
    """
    return f'<audio controls src="file://{wav_path}" style="width: 200px;"></audio>'

def main():
    st.title("Audio Alignment Explorer")
    
    # Load dataset
    json_path = "7501579_960s_aligned.json"
    df = load_dataset(json_path)
    
    # Extract all segments first
    df["wav_path"] = df.apply(
        lambda row: extract_audio_segment(
            row["audio_file"], 
            row["start"], 
            row["end"]
        ), 
        axis=1
    )
    
    # Create audio player column
    df["audio_player"] = df["wav_path"].apply(create_audio_player)
    
    # Display each segment with its audio
    for _, row in df.iterrows():
        st.write("---")  # Separator between segments
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.components.v1.html(row["audio_player"], height=50)
            st.write(f"Time: {row['start']:.1f}s → {row['end']:.1f}s")
            st.write(f"CER: {row['cer']:.3f}")
        
        with col2:
            st.write("ASR Text:")
            st.text(row["asr_text"])
            st.write("Human Text:")
            st.text(row["human_text"])
    
    # Cleanup old segments (optional)
    # You might want to implement cleanup of old segment files here

if __name__ == "__main__":
    main() 