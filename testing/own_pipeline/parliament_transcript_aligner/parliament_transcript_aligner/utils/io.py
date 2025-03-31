import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ..data_models.models import TranscribedSegment, AlignedTranscript

def save_alignments(aligned_segments: List[AlignedTranscript], 
                   audio_path: str,
                   output_path: str) -> None:
    """Save aligned segments to JSON file.
    
    Args:
        aligned_segments: List of aligned transcripts
        audio_path: Path to original audio file
        output_path: Path to save JSON file
    """
    data = {
        "audio_file": audio_path,
        "segments": [segment.to_dict() for segment in aligned_segments]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_alignments(json_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Load aligned segments from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (audio_path, list of segment dictionaries)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["audio_file"], data["segments"]

def save_transcribed_segments(segments: List[TranscribedSegment], output_path: Path) -> None:
    """Save transcribed segments using pickle.
    
    Args:
        segments: List of TranscribedSegments
        output_path: Path to save pickle file
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(segments, f)

def load_transcribed_segments(pickle_path: Path) -> List[TranscribedSegment]:
    """Load transcribed segments from pickle file.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        List of TranscribedSegment objects
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f) 