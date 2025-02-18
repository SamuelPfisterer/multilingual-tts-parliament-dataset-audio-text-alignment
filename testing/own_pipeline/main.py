from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, Annotation, Timeline
from pyannote.audio import Model
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from huggingface_hub import login
import torch.serialization
from omegaconf.listconfig import ListConfig
from typing import Union, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import os
from pydub import AudioSegment
import Levenshtein  # for CER calculation
import time
import json
from tqdm import tqdm  # Add this import at the top
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

@dataclass
class TranscribedSegment:
    segment: Segment   
    text: str
    
    # Convenience properties to access segment attributes
    @property
    def start(self) -> float:
        return self.segment.start
        
    @property
    def end(self) -> float:
        return self.segment.end
        
    @property
    def duration(self) -> float:
        return self.segment.duration
        
    def __str__(self) -> str:
        return f"[{self.start:.1f}s → {self.end:.1f}s] {self.text}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }

@dataclass
class AlignedTranscript:
    """Represents an aligned segment between ASR and human transcript."""
    asr_segment: TranscribedSegment
    human_text: str
    start_idx: int  # Token index in human transcript
    end_idx: int    # Token index in human transcript
    cer: float
    
    # Convenience properties to access ASR segment timing
    @property
    def start(self) -> float:
        return self.asr_segment.start
        
    @property
    def end(self) -> float:
        return self.asr_segment.end
        
    @property
    def duration(self) -> float:
        return self.asr_segment.duration
        
    @property
    def asr_text(self) -> str:
        return self.asr_segment.text
    
    def __str__(self) -> str:
        return (f"[{self.start:.1f}s → {self.end:.1f}s]\n"
                f"ASR: {self.asr_text}\n"
                f"Human: {self.human_text}\n"
                f"CER: {self.cer:.3f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "asr_text": self.asr_text,
            "human_text": self.human_text,
            "cer": self.cer,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx
        }

class AudioSegmenter:
    def __init__(self, vad_pipeline: VoiceActivityDetection, window_min_size: float = 10.0, window_max_size: float = 20.0):
        """Initialize the AudioSegmenter.
        
        Args:
            vad_pipeline: PyAnnote VAD pipeline
            window_min_size: Minimum size of the window to look for silence in seconds
            window_max_size: Maximum size of the window to look for silence in seconds
        """
        self.vad_pipeline = vad_pipeline
        self.window_min_size = window_min_size
        self.window_max_size = window_max_size
        
        # Set cache directory for Hugging Face
        cache_dir = os.getenv("HF_CACHE_DIR")
        
        # Set environment variable to ensure HF uses the correct cache
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['TORCH_HOME'] = cache_dir

        '''
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir=cache_dir
        )
        '''
        # Load the model and processor with cache_dir
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3",
            cache_dir=cache_dir,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

        processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3",
            cache_dir=cache_dir
        )

        # Create the pipeline using the loaded model and processor
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor
        )
        
    def get_longest_silence(self, non_speech_regions: Timeline, start: float, end: float) -> Optional[Segment]:
        """Get the longest silence segment in the given time window.
        
        Args:
            non_speech_regions: Timeline containing non-speech segments
            start: Start time of window to search in
            end: End time of window to search in
            
        Returns:
            The longest silence segment in the window, or None if no silence found
        """
        # Create a support segment for the window we're analyzing
        support = Segment(start, end)
        
        # Get all silence segments in this window
        window_silences = non_speech_regions.crop(support=support, mode="intersection")
        
        # Find the longest silence segment
        if not window_silences:
            return None
            
        return max(window_silences, key=lambda s: s.duration)

    def convert_audio_to_wav(self, audio_path: str) -> str:
        """Convert audio file to wav format using ffmpeg if needed.
        
        Args:
            audio_path: Path to audio file (.opus or .wav)
            
        Returns:
            Path to wav file
        """
        if not audio_path.endswith(('.opus', '.wav')):
            raise ValueError("Audio file must be .opus or .wav format")
            
        wav_path = audio_path.replace('.opus', '.wav')
        if not os.path.exists(wav_path):
            print(f"Converting {audio_path} to {wav_path}")
            cmd = f'ffmpeg -y -i {audio_path} -ac 1 -ar 16000 {wav_path}'
            os.system(cmd)
        return wav_path

    def extract_audio_segment(self, audio_path: str, start: float, end: float) -> str:
        """Extract a segment from an audio file and return a temporary path.
        
        Args:
            audio_path: Path to audio file (.opus or .wav)
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Path to temporary wav file containing the segment
        """
        # Convert to wav if needed
        wav_path = self.convert_audio_to_wav(audio_path)
        
        audio = AudioSegment.from_file(wav_path)
        segment = audio[start * 1000:end * 1000]  # pydub works in milliseconds
        temp_path = f"/tmp/segment_{start}_{end}.wav"
        segment.export(temp_path, format="wav")
        return temp_path

    def segment_and_transcribe(self, audio_path: str) -> List[TranscribedSegment]:
        """Segment audio file and transcribe each segment.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of TranscribedSegments containing timing and text
        """
        # First get all segments
        segments_timeline = self.segment_audio(audio_path)
        
        # Transcribe each segment with progress bar
        transcribed_segments = []
        
        for segment in tqdm(segments_timeline, desc="Transcribing segments"):
            # Extract the segment
            temp_path = self.extract_audio_segment(audio_path, segment.start, segment.end)
            
            # Transcribe
            text = self.asr_pipeline(temp_path)["text"].strip()
            
            # Clean up
            os.remove(temp_path)
            
            transcribed_segments.append(TranscribedSegment(segment, text))
            
        return transcribed_segments

    def segment_audio(self, audio_path: str) -> Timeline:
        """Segment audio file based on silence detection.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Timeline containing all segments
        """
        # Get speech regions for the entire audio
        speech_regions = self.vad_pipeline(audio_path)
        non_speech_regions = speech_regions.get_timeline().gaps()
        
        segments = Timeline()
        current_pos = 0.0
        audio_end = speech_regions.get_timeline().extent().end
        
        while current_pos < audio_end:
            # Define the window we're looking at
            window_start = current_pos + self.window_min_size
            window_end = current_pos + self.window_max_size
            
            # Get the longest silence in this window
            max_silence = self.get_longest_silence(
                non_speech_regions, 
                window_start, 
                window_end
            )
            
            if max_silence is None:
                # If no silence found, end segment at window_end
                segments.add(Segment(current_pos, window_end))
                current_pos = window_end
            else:
                # End segment at middle of silence
                silence_middle = max_silence.middle
                segments.add(Segment(current_pos, silence_middle))
                current_pos = silence_middle
                
        return segments

def initialize_vad_pipeline():
    """
    Initialize the pyannote VAD pipeline.
    """
    cache_dir = os.getenv("HF_CACHE_DIR")
    print(f"Cache directory: {cache_dir}")
    
    # First load the model with cache_dir
    model = Model.from_pretrained(
        "pyannote/segmentation",
        use_auth_token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    
    # Then create the VAD pipeline using the loaded model
    vad_pipeline = VoiceActivityDetection(segmentation=model)
    
    # Configure the pipeline parameters
    vad_pipeline.instantiate({
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    })
    
    return vad_pipeline

def get_max_segment(segments: Union[Timeline, Annotation]) -> Optional[Segment]:
    if len(segments) == 0:
        return None
    max_segment = max(segments, key=lambda s: s.duration)
    return max_segment

class TranscriptAligner:
    def __init__(self, 
                 window_token_margin: int = 30,
                 cer_threshold: float = 0.3):
        """Initialize the TranscriptAligner.
        
        Args:
            window_token_margin: Extra tokens to consider on each side of the window
            cer_threshold: Maximum allowable Character Error Rate for early stopping
        """
        self.window_token_margin = window_token_margin
        self.cer_threshold = cer_threshold
        
    def compute_cer(self, asr_text: str, human_text: str) -> float:
        """Compute Character Error Rate between two strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Character Error Rate between 0 and 1
        """
        distance = Levenshtein.distance(asr_text, human_text)
        asr_len = len(asr_text)  # Length of ASR text. We use this as baseline lenght for CER, as otherwise taking a longer human text would always result in a lower CER, if ASR text and human text don't agree at all.
        return distance / asr_len if asr_len > 0 else 1.0
        
    def find_best_match(self, 
                       asr_segment: TranscribedSegment,
                       transcript_tokens: List[str],
                       start_search_idx: int = 0) -> AlignedTranscript:
        """Find best matching segment in human transcript for ASR segment.
        
        Args:
            asr_segment: TranscribedSegment from ASR
            transcript_tokens: Tokenized human transcript
            start_search_idx: Index to start searching from
            
        Returns:
            AlignedTranscript containing the best match
        """
        asr_tokens = asr_segment.text.split()
        print(f"ASR Tokens: {asr_tokens}")
        print(f"Number of tokens: {len(asr_tokens)}")
        print("Individual tokens:")
        for i, token in enumerate(asr_tokens):
            print(f"  {i}: {token}")
        num_predicted = len(asr_tokens)
        window_size = num_predicted + 2 * self.window_token_margin
        
        best_cer = float('inf')
        best_match = None
        
        # Iterate over possible start positions
        for start_offset in range(-self.window_token_margin, self.window_token_margin + 1):
            candidate_start = start_search_idx + start_offset
            if candidate_start < 0:
                continue
                
            # Iterate over possible window sizes
            for window_tokens in range(num_predicted - self.window_token_margin, 
                                    num_predicted + self.window_token_margin + 1):
                candidate_end = candidate_start + window_tokens
                if candidate_end > len(transcript_tokens):
                    break
                    
                # Extract and compare candidate text
                candidate_text = " ".join(transcript_tokens[candidate_start:candidate_end])
                cer = self.compute_cer(asr_segment.text, candidate_text)
                
                if cer < best_cer:
                    best_cer = cer
                    best_match = AlignedTranscript(
                        asr_segment=asr_segment,
                        human_text=candidate_text,
                        start_idx=candidate_start,
                        end_idx=candidate_end,
                        cer=cer
                    )
                    
                    # Early stopping if we found a very good match
                    if cer <= self.cer_threshold:
                        return best_match
                        
        return best_match

    def align_transcript(self, 
                        transcribed_segments: List[TranscribedSegment],
                        human_transcript: str) -> List[AlignedTranscript]:
        """Align all ASR segments with human transcript.
        
        Args:
            transcribed_segments: List of TranscribedSegments from ASR
            human_transcript: Full human transcript text
            
        Returns:
            List of AlignedTranscript objects
        """
        transcript_tokens = human_transcript.split()
        aligned_segments = []
        last_end_idx = 0
        
        # Add progress bar for alignment
        for segment in tqdm(transcribed_segments, desc="Aligning segments"):
            aligned = self.find_best_match(
                segment, 
                transcript_tokens,
                start_search_idx=last_end_idx
            )
            aligned_segments.append(aligned)
            last_end_idx = aligned.end_idx if aligned else last_end_idx
            
        return aligned_segments

def save_alignments(aligned_segments: List[AlignedTranscript], 
                   audio_path: str,
                   output_path: str):
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

def save_transcribed_segments(segments: List[TranscribedSegment], output_path: Path):
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

def main():
    # Define directory structure
    data_dir = Path("data")
    cache_dir = Path("cache")
    output_dir = Path("output")
    
    # Create subdirectories
    audio_dir = data_dir / "audio"  # for .opus and .wav files
    transcript_dir = data_dir / "transcripts"  # for .txt files
    cache_dir = cache_dir / "transcribed_segments"  # for cached .pkl files
    alignment_dir = output_dir / "alignments"  # for final .json files
    
    # Create all directories
    for directory in [audio_dir, transcript_dir, cache_dir, alignment_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    file_name = "7501579_960s"
    audio_path = audio_dir / f"{file_name}.opus"
    #transcript_path = transcript_dir / f"{file_name}.txt"
    transcript_path = "output.txt"
    cache_path = cache_dir / f"{file_name}_transcribed.pkl"
    alignment_path = alignment_dir / f"{file_name}_aligned.json"
    
    # Check if we have cached transcribed segments
    if cache_path.exists():
        print("Loading cached transcribed segments...")
        transcribed_segments = load_transcribed_segments(cache_path)
    else:
        print("Transcribing audio segments...")
        vad_pipeline = initialize_vad_pipeline()
        segmenter = AudioSegmenter(vad_pipeline)
        transcribed_segments = segmenter.segment_and_transcribe(str(audio_path))
        
        # Cache the results
        print("Caching transcribed segments...")
        save_transcribed_segments(transcribed_segments, cache_path)
    
    # Load human transcript
    with open(transcript_path, "r") as f:
        human_transcript = f.read()
    
    print(f"Starting alignment...")
    start_time = time.time()
    # Align transcripts
    aligner = TranscriptAligner(window_token_margin=30)
    aligned_segments = aligner.align_transcript(transcribed_segments, human_transcript)
    end_time = time.time()
    print(f"Alignment completed in {end_time - start_time} seconds")
    
    # Save alignments
    # Save final alignments as JSON (keeping this as JSON since it's useful to inspect)
    save_alignments(aligned_segments, str(audio_path), alignment_path)

if __name__ == "__main__":
    main()