from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, Annotation, Timeline
from pyannote.audio import Model, Pipeline
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
from datetime import timedelta


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
    def __init__(self, vad_pipeline: VoiceActivityDetection, diarization_pipeline: Pipeline, window_min_size: float = 10.0, window_max_size: float = 20.0):
        """Initialize the AudioSegmenter.
        
        Args:
            vad_pipeline: PyAnnote VAD pipeline
            window_min_size: Minimum size of the window to look for silence in seconds
            window_max_size: Maximum size of the window to look for silence in seconds
        """
        self.vad_pipeline = vad_pipeline
        self.diarization_pipeline = diarization_pipeline
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
        print(f"Converting audio file to wav before segmenting: {audio_path}")
        audio_path = self.convert_audio_to_wav(audio_path)
        print(f"Segmenting audio file: {audio_path}")
        speech_regions = self.vad_pipeline(audio_path)
        print(f"Speech regions: {speech_regions}")
        print(f"Speech regions type: {type(speech_regions)}")
        non_speech_regions = speech_regions.get_timeline().gaps()
        print(f"Non speech regions: {non_speech_regions}")
        print(f"Non speech regions type: {type(non_speech_regions)}")
        diarization = self.diarization_pipeline(audio_path)
        print(f"Diarization: {diarization}")
        print(f"Diarization type: {type(diarization)}")
        # get the timeline with overlapping speaker segments its of type Timeline
        overlapping_speaker_segments = diarization.get_overlap()


        
        segments = Timeline()
        # we skip initial silence
        current_pos = speech_regions.get_timeline().extent().start
        # we also skip the last silence
        audio_end = speech_regions.get_timeline().extent().end
        
        while current_pos < audio_end:
            print(f"Current position: {current_pos}")
            # Define the window we're looking at
            window_start = current_pos + self.window_min_size
            window_end = current_pos + self.window_max_size

            # check if the window only contains silence
            max_segment_silence = self.get_longest_silence(
                non_speech_regions, 
                current_pos, 
                window_end
            )
            print(f"Max segment silence: {max_segment_silence}")
            if max_segment_silence and max_segment_silence.start == current_pos and max_segment_silence.duration > self.window_min_size:
                # if the segment we are checking only contains silence, we skip it and adjust the current position accordingly
                current_pos = max_segment_silence.end # we move the current position to the end of the silence
                continue
            
            # check if the segment starts with an overlapping speaker segment
            print(f"Overlapping speaker segments: {overlapping_speaker_segments}")
            print(f"current pos in time-format like 00:12:52.264: {timedelta(seconds=current_pos)} ")
            overlapping_speaker_segments_window_start = overlapping_speaker_segments.crop(Segment(current_pos, window_end), mode="loose")
            print(f"Overlapping speaker segments window start: {overlapping_speaker_segments_window_start}")
            if overlapping_speaker_segments_window_start:
                # if there is an overlapping speaker segment, we just move the current position to the end of this overlap
                current_pos = overlapping_speaker_segments_window_start[0].end + 1e-6
                continue

            
            # check if there is a speaker change in this segment
            overlapping_segments = diarization.crop(Segment(current_pos, window_end), mode="intersection")
            print(f"Overlapping segments: {overlapping_segments}")
            last_speaker = None
            speaker_change_detected = False
            
            # Correct way to iterate through an Annotation object
            for segment, track, label in overlapping_segments.itertracks(yield_label=True):
                if last_speaker is None:
                    last_speaker = label
                    print(f"Last speaker: {last_speaker}")
                elif label != last_speaker:
                    print(f"Speaker change detected at {segment.start}s")
                    print(f"Current position: {current_pos}")
                    print(f"Segment start: {segment.start}")
                    print(f"Last speaker: {last_speaker}")
                    print(f"New speaker: {label}")
                    segments.add(Segment(current_pos, segment.start))
                    current_pos = segment.start
                    speaker_change_detected = True
                    break
                    
            if speaker_change_detected:
                # if there is a speaker change, we have already added the segment and moved the current position, so we continue to the next segment
                continue
                
            # If we reach here, no speaker change was detected
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

def initialize_diarization_pipeline(): 
    """
    Initialize the pyannote diarization pipeline.
    """
    hf_token = os.getenv("HF_AUTH_TOKEN")
    print(f"HF token: {hf_token}")
    cache_dir = os.getenv("HF_CACHE_DIR")
    print(f"Cache directory: {cache_dir}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token,
        cache_dir=cache_dir
    )

    pipeline.to(torch.device("cuda"))

    return pipeline

def get_max_segment(segments: Union[Timeline, Annotation]) -> Optional[Segment]:
    if len(segments) == 0:
        return None
    max_segment = max(segments, key=lambda s: s.duration)
    return max_segment

class TranscriptAligner:
    def __init__(self, 
                 window_token_margin: int = 30,
                 region_cer_threshold: float = 0.3,
                 finetune_cer_threshold: float = 0.05):
        """Initialize the TranscriptAligner.
        
        Args:
            window_token_margin: Extra tokens to consider on each side of the window
            region_cer_threshold: Maximum allowable Character Error Rate for a region to be considered a good match
            finetune_cer_threshold: Maximum allowable Character Error Rate for early stopping during fine-tuning
        """
        self.window_token_margin = window_token_margin
        self.region_cer_threshold = region_cer_threshold
        self.finetune_cer_threshold = finetune_cer_threshold
        
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
        
        Uses a two-phase approach:
        1. Find the most promising region in the transcript
        2. Fine-tune the exact match boundaries within that region
        
        Args:
            asr_segment: TranscribedSegment from ASR
            transcript_tokens: Tokenized human transcript
            start_search_idx: Index to start searching from
            
        Returns:
            AlignedTranscript containing the best match
        """
        # Phase 1: Find the best matching region
        region_start_idx = self._find_match_region(
            asr_segment.text,
            transcript_tokens,
            start_search_idx, 
            coarse_window_size = len(asr_segment.text.split())
        )
        
        if region_start_idx is None:
            # No good matching region found, so we try to find a region at the beginning of the transcript
            region_start_idx = self._find_match_region(
                asr_segment.text,
                transcript_tokens,
                0,
                coarse_window_size = len(asr_segment.text.split()),
            )
            if region_start_idx is None:
                return self._create_fallback_alignment(asr_segment, transcript_tokens, start_search_idx)
        
        # Phase 2: Fine-tune the match within the identified region
        return self._fine_tune_match(
            asr_segment,
            transcript_tokens,
            region_start_idx
        )

    def _find_match_region(self,
                          asr_text: str,
                          transcript_tokens: List[str],
                          start_search_idx: int,
                          coarse_window_size: int = 50,
                          region_cer_threshold: float = 0.3,
                          max_backward_search: int = 250,  # Maximum tokens to search backward
                          forward_priority: int = 5  # Check this many forward windows before one backward
                          ) -> Optional[int]:
        """Find the most promising region for matching.
        
        Uses an expanding search pattern that continues until either:
        1. A good match is found (CER below threshold)
        2. The entire remaining transcript has been searched
        Uses an expanding search pattern that prioritizes forward search.
        For every backward step, checks forward_priority number of forward steps first.
        
        Args:
            asr_text: Text from ASR segment
            transcript_tokens: Full transcript tokens
            start_search_idx: Starting point for search
            coarse_window_size: Size of window for coarse search
            region_cer_threshold: Maximum CER to consider a region as promising
            max_backward_search: Maximum number of tokens to search backward
            forward_priority: Number of forward windows to check before each backward window
        
        Returns:
            Starting index of best matching region or None if no good match found
        """
        
         # Initialize debug info collection
        debug_info = [f"Finding best matching region for {asr_text}"]
        if start_search_idx == 0:
            debug_info.append(f"Start search index is 0, so we are at the beginning of the transcript")

        best_cer = float('inf')
        best_start_idx = None
        
        # Initialize search boundaries
        backward_limit = max(0, start_search_idx - max_backward_search)
        forward_limit = len(transcript_tokens)
        debug_info.append(f"Search boundaries: backward_limit={backward_limit}, forward_limit={forward_limit}")
        
        # Initialize positions
        forward_pos = start_search_idx
        backward_pos = start_search_idx
        forward_steps = 0  # Counter for forward steps taken
        
        while True:
            # Check forward positions with priority
            while forward_steps < forward_priority and forward_pos < forward_limit:
                if forward_pos + coarse_window_size > forward_limit:
                    # if we are close to the end of the trancript, we want to still have a last full window to check
                    forward_pos = forward_limit - coarse_window_size
                candidate_end = min(forward_pos + coarse_window_size, forward_limit)
                candidate_text = " ".join(transcript_tokens[forward_pos:candidate_end])
                cer = self.compute_cer(asr_text, candidate_text)
                
                if cer < best_cer:
                    best_cer = cer
                    best_start_idx = forward_pos
                    
                    if cer <= region_cer_threshold:
                        debug_info.extend([
                            f"Best match found at index {forward_pos} with CER {cer}",
                            f"Best match Region is {transcript_tokens[forward_pos:candidate_end]}"
                        ])
                        # Write debug info before returning
                        with open("debugging.txt", "a", encoding="utf-8") as debug_file:
                            if debug_file.tell() > 0:
                                debug_file.write("\n")
                            debug_file.write("\n".join(debug_info) + "\n")
                        return forward_pos
                
                forward_pos += coarse_window_size
                forward_steps += 1
            
            # Reset forward steps counter
            forward_steps = 0
            
            # Try one backward position if within bounds
            if backward_pos > backward_limit:
                backward_pos = max(backward_pos - coarse_window_size, backward_limit)
                candidate_end = min(backward_pos + coarse_window_size, forward_limit)
                candidate_text = " ".join(transcript_tokens[backward_pos:candidate_end])
                cer = self.compute_cer(asr_text, candidate_text)
                
                if cer < best_cer:
                    best_cer = cer
                    best_start_idx = backward_pos
                    
                    if cer <= region_cer_threshold:
                        debug_info.extend([
                            f"Best match found at index {backward_pos} with CER {cer}",
                            f"Best match Region is {transcript_tokens[backward_pos:candidate_end]}"
                        ])
                        # Write debug info before returning
                        with open("debugging.txt", "a", encoding="utf-8") as debug_file:
                            if debug_file.tell() > 0:
                                debug_file.write("\n")
                            debug_file.write("\n".join(debug_info) + "\n")
                        return backward_pos
            
            # Stop if we've searched the entire valid range
            if forward_pos >= forward_limit and backward_pos <= backward_limit:
                break
        
        # If we've searched everything and found no good match,
        # return the best match we found if it's reasonable, otherwise None
        
        # Add final debug info for no good match case
        debug_info.extend([
            f"Best match found at index {best_start_idx} with CER {best_cer}",
            f"Best match Region is {transcript_tokens[best_start_idx:best_start_idx + coarse_window_size]}"
        ])
        
        # Write all debug info at the end
        with open("debugging.txt", "a", encoding="utf-8") as debug_file:
            if debug_file.tell() > 0:
                debug_file.write("\n")
            debug_file.write("\n".join(debug_info) + "\n")

        return best_start_idx if best_cer <= region_cer_threshold * 1.5 else None

    def _fine_tune_match(self,
                        asr_segment: TranscribedSegment,
                        transcript_tokens: List[str],
                        region_start_idx: int) -> AlignedTranscript:
        """Fine-tune the exact match boundaries within the identified region.
        
        This uses the original window-based approach but focused on the identified region.
        """
        debug_info = [f"Fine-tuning match for {asr_segment.text}"]
        
        asr_tokens = asr_segment.text.split()
        num_predicted = len(asr_tokens)
        best_cer = float('inf')
        crossed_cer_threshold = False
        best_match = None
        
        # Search within a smaller window around the identified region
        local_margin = self.window_token_margin // 2  # Use smaller margin for fine-tuning
        
        
        for start_offset in range(-local_margin, local_margin + 1):
            candidate_start = region_start_idx + start_offset
            if candidate_start < 0:
                continue
            best_cer_for_candidate_start = float('inf')  # Variable representing the best CER achieved for the current candidate_start
                
            for window_tokens in range(num_predicted - local_margin, 
                                     num_predicted + local_margin + 1):
                candidate_end = candidate_start + window_tokens
                if candidate_end > len(transcript_tokens):
                    break
                    
                candidate_text = " ".join(transcript_tokens[candidate_start:candidate_end])
                cer = self.compute_cer(asr_segment.text, candidate_text)
                best_cer_for_candidate_start = min(best_cer_for_candidate_start, cer)
                
                if cer < best_cer:
                    best_cer = cer
                    best_match = AlignedTranscript(
                        asr_segment=asr_segment,
                        human_text=candidate_text,
                        start_idx=candidate_start,
                        end_idx=candidate_end,
                        cer=cer
                    )
                    
                    if cer <= self.finetune_cer_threshold:
                        crossed_cer_threshold = True
            
            if crossed_cer_threshold and best_cer_for_candidate_start > self.finetune_cer_threshold: # we stop fine-tuning if we have already crossed the CER threshold and we have now moved out of the CER threshold
                debug_info.extend([
                    f"Best match found for {asr_segment.text}",
                    f"Human text: {best_match.human_text}",
                    f"CER: {best_cer}",
                    f"Segment: {transcript_tokens[best_match.start_idx:best_match.end_idx]}\n"
                ])
                with open("debugging.txt", "a") as debug_file:
                    debug_file.write("\n".join(debug_info))
                return best_match
        # Add final debug info for case where no match below threshold was found
        debug_info.extend([
            f"Best match found for {asr_segment.text}",
            f"Human text: {best_match.human_text}",
            f"CER: {best_cer}",
            f"Segment: {transcript_tokens[best_match.start_idx:best_match.end_idx]}\n"
        ])
        with open("debugging.txt", "a") as debug_file:
            debug_file.write("\n".join(debug_info))
        return best_match

    def _create_fallback_alignment(self,
                                 asr_segment: TranscribedSegment,
                                 transcript_tokens: List[str],
                                 start_search_idx: int) -> AlignedTranscript:
        """Create a fallback alignment when no good match is found."""
        # if we are at the beginning of the transcript, we just take the first window
        # we simply apply the fine-tuning from this start_search_idx
        return self._fine_tune_match(asr_segment, transcript_tokens, start_search_idx)
        '''
        # Use a minimal window as fallback
        end_idx = min(start_search_idx + len(asr_segment.text.split()), len(transcript_tokens))

        debug_info = [f"Fallback alignment for {asr_segment.text}"]
        debug_info.append(f"Human text: {transcript_tokens[start_search_idx:end_idx]}")
        debug_info.append(f"Start index: {start_search_idx}, End index: {end_idx}")
        debug_info.append("")
        with open("debugging.txt", "a") as debug_file:
            debug_file.write("\n".join(debug_info))

        return AlignedTranscript(
            asr_segment=asr_segment,
            human_text=" ".join(transcript_tokens[start_search_idx:end_idx]),
            start_idx=start_search_idx,
            end_idx=end_idx,
            cer=1.0  # Maximum CER to indicate poor match
        )
        '''

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
    file_name = "7501579_3840s"
    audio_path = audio_dir / f"{file_name}.opus"
    #transcript_path = transcript_dir / f"{file_name}.txt"
    transcript_path = "7501579_llm.txt"
    cache_path = cache_dir / f"{file_name}_transcribed.pkl"
    alignment_path = alignment_dir / f"{file_name}_aligned.json"
    
    # Check if we have cached transcribed segments
    if cache_path.exists():
        print("Loading cached transcribed segments...")
        transcribed_segments = load_transcribed_segments(cache_path)
    else:
        print("Transcribing audio segments...")
        vad_pipeline = initialize_vad_pipeline()
        diarization_pipeline = initialize_diarization_pipeline()    
        segmenter = AudioSegmenter(vad_pipeline, diarization_pipeline)
        transcribed_segments = segmenter.segment_and_transcribe(str(audio_path))
        
        # Cache the results
        print("Caching transcribed segments...")
        save_transcribed_segments(transcribed_segments, cache_path)
    
    # Load human transcript
    with open(transcript_path, "r") as f:
        print(f"Loading human transcript from {transcript_path}")
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