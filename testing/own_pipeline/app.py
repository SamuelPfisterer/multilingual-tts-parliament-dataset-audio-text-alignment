import streamlit as st
import json
import io
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
from pydub import AudioSegment
import os
import glob

# Use caching so that loading the audio file and JSON file are fast on each re-run
@st.cache_data
def load_segments(json_file: str) -> Tuple[pd.DataFrame, str]:
    """
    Load JSON data and create a DataFrame that includes a duration field.
    The JSON should contain keys "audio_file" and "segments".
    
    Args:
        json_file: Path to the JSON file containing segments data
        
    Returns:
        Tuple of (DataFrame with segment data, path to audio file)
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or "segments" not in data or "audio_file" not in data:
            st.error("Invalid JSON format: must contain 'segments' and 'audio_file' keys")
            return pd.DataFrame(), ""
            
        df = pd.DataFrame(data["segments"])
        # Compute duration (in seconds) for each segment
        df["duration"] = df["end"] - df["start"]
        audio_file = data["audio_file"]
        return df, audio_file
    except Exception as e:
        st.error(f"Error loading segments: {str(e)}")
        return pd.DataFrame(), ""


@st.cache_data
def get_available_json_files(directory: str) -> List[str]:
    """
    Get a list of all JSON files in the specified directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        List of filenames (without path)
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory):
            return []
            
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(directory, "*.json"))
        
        # Extract just the filenames without the path
        filenames = [os.path.basename(f) for f in json_files]
        
        # Sort alphabetically
        filenames.sort()
        
        return filenames
    except Exception as e:
        st.error(f"Error listing JSON files: {str(e)}")
        return []


@st.cache_data
def load_audio(audio_file_path: str) -> Optional[AudioSegment]:
    """
    Load the full audio file (once).
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        AudioSegment object or None if loading fails
    """
    try:
        if not audio_file_path.endswith(('.opus', '.wav')):
            raise ValueError("Audio file must be .opus or .wav format")
            
        wav_path = audio_file_path.replace('.opus', '.wav')
        if not os.path.exists(wav_path):
            print(f"Converting {audio_file_path} to {wav_path}")
            cmd = f'ffmpeg -y -i {audio_file_path} -ac 1 -ar 16000 {wav_path}'
            os.system(cmd)

        return AudioSegment.from_file(wav_path, format="wav")
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None


def extract_segment_audio(audio: AudioSegment, start: float, end: float) -> io.BytesIO:
    """
    Extract and return the audio subsegment as an in-memory BytesIO buffer.
    
    Args:
        audio: The full AudioSegment
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        BytesIO buffer containing the audio segment in WAV format
    """
    # pydub works in milliseconds
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    
    # Ensure start and end are within bounds
    start_ms = max(0, start_ms)
    end_ms = min(len(audio), end_ms)
    
    segment = audio[start_ms:end_ms]
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    buf.seek(0)
    return buf


def main():
    st.set_page_config(
        page_title="Audio Alignment Explorer",
        page_icon="🔊",
        layout="wide"
    )
    
    st.title("Audio Alignment Explorer with Filters")

    # Default JSON file path and directory
    alignments_dir = "output/alignments"
    default_json_file = os.path.join(alignments_dir, "7501579_960s_aligned.json")
    
    # File input options
    st.sidebar.header("File Selection")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Default file", "Select from available files", "Enter filename", "Upload file"]
    )
    
    if input_method == "Select from available files":
        # Get available JSON files
        available_files = get_available_json_files(alignments_dir)
        
        if not available_files:
            st.sidebar.warning(f"No JSON files found in {alignments_dir}")
            json_file = default_json_file
        else:
            # Create a selectbox with search functionality
            selected_file = st.sidebar.selectbox(
                "Select a JSON file",
                available_files,
                index=available_files.index("7501579_960s_aligned.json") if "7501579_960s_aligned.json" in available_files else 0
            )
            
            json_file = os.path.join(alignments_dir, selected_file)
    
    elif input_method == "Enter filename":
        # Allow user to input a filename
        json_filename = st.sidebar.text_input(
            "Enter JSON filename (in output/alignments/)",
            value="7501579_960s_aligned.json"
        )
        
        # Construct the full path
        json_file = os.path.join(alignments_dir, json_filename)
        
        # Check if file exists
        if not os.path.exists(json_file):
            st.sidebar.warning(f"File not found: {json_file}")
            st.sidebar.info(f"Using default file instead: {default_json_file}")
            json_file = default_json_file
    
    elif input_method == "Upload file":
        # Allow user to upload a JSON file
        uploaded_json = st.sidebar.file_uploader("Upload JSON file", type=["json"])
        
        if uploaded_json:
            # Save the uploaded file to a temporary location
            with open("temp_uploaded.json", "wb") as f:
                f.write(uploaded_json.getvalue())
            json_file = "temp_uploaded.json"
        else:
            json_file = default_json_file
    
    else:  # Default file
        json_file = default_json_file
    
    # Display the selected file
    st.sidebar.write(f"Using JSON file: {json_file}")
        
    # Load segments data
    df, audio_file_name = load_segments(json_file)
    print(f"audio_file_name: {audio_file_name}")
    audio_file_name = audio_file_name.split("/")[-1]
    audio_file_path = f"data/audio/{audio_file_name}"
    print(f"audio_file_path: {audio_file_path}")
    
    if df.empty:
        st.warning("No segments data available. Please check your JSON file.")
        return
        
    st.write(f"Using Audio File: {audio_file_path}")

    # Load full audio file (caching avoids repeated loads)
    audio = load_audio(audio_file_path)
    
    if audio is None:
        st.error(f"Failed to load audio file: {audio_file_path}")
        return

    # ================================
    # Sidebar Filters
    # ================================
    st.sidebar.header("Filter Options")

    # Filter by CER (Character Error Rate)
    if "cer" in df.columns:
        min_cer_value = float(df["cer"].min())
        max_cer_value = float(df["cer"].max())
        
        min_cer = st.sidebar.number_input(
            "Min CER", 
            value=min_cer_value, 
            min_value=min_cer_value, 
            max_value=max_cer_value, 
            step=0.01
        )
        max_cer = st.sidebar.number_input(
            "Max CER", 
            value=max_cer_value, 
            min_value=min_cer_value, 
            max_value=max_cer_value, 
            step=0.01
        )
    else:
        st.sidebar.warning("CER data not available in the JSON file")
        min_cer, max_cer = 0.0, 1.0

    # Filter by segment duration (in seconds)
    min_duration_value = float(df["duration"].min())
    max_duration_value = float(df["duration"].max())
    
    min_duration = st.sidebar.number_input(
        "Min Duration (s)", 
        value=min_duration_value, 
        min_value=min_duration_value, 
        max_value=max_duration_value, 
        step=0.5
    )
    max_duration = st.sidebar.number_input(
        "Max Duration (s)", 
        value=max_duration_value, 
        min_value=min_duration_value, 
        max_value=max_duration_value, 
        step=0.5
    )

    # Apply filters to the DataFrame
    filtered_df = df.copy()
    
    # Apply CER filter if column exists
    if "cer" in df.columns:
        filtered_df = filtered_df[
            (filtered_df["cer"] >= min_cer) & (filtered_df["cer"] <= max_cer)
        ]
    
    # Apply duration filter
    filtered_df = filtered_df[
        (filtered_df["duration"] >= min_duration) & (filtered_df["duration"] <= max_duration)
    ]

    # Sort options
    sort_options = ["start", "duration", "cer"] if "cer" in df.columns else ["start", "duration"]
    sort_by = st.sidebar.selectbox("Sort by", sort_options)
    sort_ascending = st.sidebar.checkbox("Sort ascending", value=True)
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)

    # Display segment count
    st.write(f"Showing {len(filtered_df)} segments after filtering.")

    # Optionally, show the filtered DataFrame as a table
    if st.checkbox("Show segments table"):
        st.dataframe(filtered_df)

    # Pagination controls
    segments_per_page = st.sidebar.slider("Segments per page", 5, 50, 10)
    total_pages = max(1, (len(filtered_df) + segments_per_page - 1) // segments_per_page)
    
    if total_pages > 1:
        page_number = st.sidebar.number_input(
            f"Page (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=1
        )
    else:
        page_number = 1

    # Calculate slice for current page
    start_idx = (page_number - 1) * segments_per_page
    end_idx = min(start_idx + segments_per_page, len(filtered_df))
    
    page_df = filtered_df.iloc[start_idx:end_idx]

    # ================================
    # Display Each Segment
    # ================================
    for idx, row in page_df.iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**Segment #{idx}**")
            st.write(f"Time: {row['start']:.2f}s → {row['end']:.2f}s")
            st.write(f"Duration: {row['duration']:.2f}s")
            
            if "cer" in row:
                st.write(f"CER: {row['cer']:.3f}")
                
            # When the "Play" button is clicked, extract and play the segment
            if st.button("Play", key=f"play_{idx}"):
                buf = extract_segment_audio(audio, row["start"], row["end"])
                st.audio(buf, format="audio/wav")
                
        with col2:
            if "asr_text" in row:
                st.write("**ASR Text:**")
                st.text(row["asr_text"])
            
            if "human_text" in row:
                st.write("**Human Text:**")
                st.text(row["human_text"])


if __name__ == "__main__":
    main()
