from typing import List, Optional, Union
import os
import warnings
from pathlib import Path
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, Timeline
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment
from pydub import silence  # Added this import for silence detection
import numpy as np
from tqdm import tqdm  # Added tqdm for progress bar

from ..data_models.models import TranscribedSegment
from ..audio_processing.vad.silero_vad import get_silero_vad  # Import get_silero_vad directly

class AudioSegmenter:
    def __init__(self, 
                 vad_pipeline: VoiceActivityDetection, 
                 diarization_pipeline: Pipeline, 
                 window_min_size: float = 10.0, 
                 window_max_size: float = 20.0,
                 hf_cache_dir: Optional[Union[Path, str]] = None,
                 delete_wav_files: bool = False,
                 wav_directory: Optional[Union[Path, str]] = None,
                 with_diarization: bool = False,
                 language: str = "en"):
        """Initialize the AudioSegmenter.
        
        Args:
            vad_pipeline: PyAnnote VAD pipeline
            diarization_pipeline: PyAnnote diarization pipeline
            window_min_size: Minimum size of the window to look for silence in seconds
            window_max_size: Maximum size of the window to look for silence in seconds
            hf_cache_dir: Optional directory for Hugging Face cache
            delete_wav_files: Whether to delete temporary WAV files after processing (default: False)
            wav_directory: Optional directory to store WAV files (default: same directory as audio file)
            with_diarization: Whether to use diarization (default: False)
            language: Audio language code using ISO 639-1 standard (default: "en" for English). Examples: "es" for Spanish, "fr" for French, "de" for German.


        """
        self.vad_pipeline = vad_pipeline
        self.diarization_pipeline = diarization_pipeline
        self.window_min_size = window_min_size
        self.window_max_size = window_max_size
        self.with_diarization = with_diarization
        self.language = language
        # Set cache directory for Hugging Face
        hf_cache_dir = hf_cache_dir if hf_cache_dir is not None else os.getenv("HF_CACHE_DIR")
        
        # Set environment variable to ensure HF uses the correct cache
        if hf_cache_dir is not None:
            os.environ['HF_HOME'] = str(hf_cache_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(hf_cache_dir)
            os.environ['TORCH_HOME'] = str(hf_cache_dir)
        else:
            warnings.warn("No cache directory provided, using default cache directory")
            print("No cache directory provided, using default cache directory")

        #model_name = "openai/whisper-large-v3"
        #model_name = "distil-whisper/distil-large-v3"
        #model_name = "distil-whisper/distil-large-v3.5" # doesn't support languages other than English unfortunatelly
        model_name = "openai/whisper-large-v3-turbo"

        # Load the model and processor with cache_dir
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            #torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            #attn_implementation="flash_attention_2",
            cache_dir=hf_cache_dir,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir
        )

        # Create the pipeline using the loaded model and processor
        print(f"Using language: {self.language}")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            generate_kwargs={"language": self.language}
        )
        
        self.delete_wav_files = delete_wav_files
        self.wav_directory = Path(wav_directory) if wav_directory is not None else None
        
    def get_longest_silence(self, 
                          non_speech_regions: Timeline, 
                          start: float, 
                          end: float) -> Optional[Segment]:
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
        
        if audio_path.endswith('.wav'):
            return audio_path
        
        if self.wav_directory:
            self.wav_directory.mkdir(parents=True, exist_ok=True)
            wav_path = str(self.wav_directory / Path(audio_path).with_suffix('.wav').name)
        else:
            wav_path = str(Path(audio_path).with_suffix('.wav'))
        
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
        converted_wav_path = None
        try:
            # Convert to WAV if needed
            print(f"Converting audio file to wav before segmenting: {audio_path}")
            converted_wav_path = self.convert_audio_to_wav(audio_path)
            
            segments_timeline = self.segment_audio(converted_wav_path)
            transcribed_segments = []

            # Use tqdm to create a progress bar for segment processing
            for segment in tqdm(segments_timeline, desc="Transcribing segments", unit="segment"):
                temp_path = self.extract_audio_segment(converted_wav_path, segment.start, segment.end)
                text = self.asr_pipeline(temp_path)["text"].strip()
                
                # Clean up
                os.remove(temp_path)
                
                transcribed_segments.append(TranscribedSegment(segment, text))
            
            return transcribed_segments
        finally:
            # Clean up the converted WAV file if needed
            if self.delete_wav_files and converted_wav_path and converted_wav_path != audio_path:
                os.remove(converted_wav_path)

    def segment_audio(self, audio_path: str) -> Timeline:
        """Segment audio file based on silence detection.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Timeline containing all segments
        """
        # Get speech regions for the entire audio
        print(f"Segmenting audio file: {audio_path}")
        
        # Using PyAnnote VAD
        #speech_regions = self.vad_pipeline(audio_path)
        
        # Instead of deriving non_speech_regions from PyAnnote, use Silero VAD directly
        # non_speech_regions = speech_regions.get_timeline().gaps()
        non_speech_regions = get_silero_vad(audio_path)
        
        if self.with_diarization:
            diarization = self.diarization_pipeline(audio_path)
            overlapping_speaker_segments = diarization.get_overlap()
        else:
            diarization = None
            overlapping_speaker_segments = None
        
        audio = AudioSegment.from_file(audio_path)
        audio = audio.normalize(headroom=5)
        silence_threshold = np.percentile([frame.rms for frame in audio[::100]], 15)
        silences = silence.detect_silence(audio, min_silence_len=200, silence_thresh=silence_threshold, seek_step=15)
        silence_regions = Timeline([Segment(start/1000, end/1000) for start, end in silences])
        
        segments = Timeline()

        # TODO: I think we don't need this necessarily as we should skip full silences anyways
        """
        # Skip initial silence
        current_pos = speech_regions.get_timeline().extent().start
        # Skip the last silence
        audio_end = speech_regions.get_timeline().extent().end
        """
        current_pos = non_speech_regions.extent().start
        audio_end = non_speech_regions.extent().end
        
        while current_pos < audio_end:
            # Define the window we're looking at
            window_start = current_pos + self.window_min_size
            window_end = current_pos + self.window_max_size

            # Check if the window only contains silence
            max_segment_silence = self.get_longest_silence(
                non_speech_regions, 
                current_pos, 
                window_end
            )
            if max_segment_silence and max_segment_silence.start == current_pos and max_segment_silence.duration > self.window_min_size:
                # If the segment we are checking only contains silence, skip it
                current_pos = max_segment_silence.end
                continue
            
            # Check if the segment starts with an overlapping speaker segment
            if overlapping_speaker_segments and diarization:
                overlapping_speaker_segments_window_start = overlapping_speaker_segments.crop(
                    Segment(current_pos, window_end), 
                    mode="loose"
                )
                if overlapping_speaker_segments_window_start:
                    # If there is an overlapping speaker segment, move to its end
                    current_pos = overlapping_speaker_segments_window_start[0].end + 1e-6
                    continue

                # Check if there is a speaker change in this segment
                overlapping_segments = diarization.crop(
                    Segment(current_pos, window_end), 
                    mode="intersection"
                )
                last_speaker = None
                speaker_change_detected = False
            
                # Iterate through speaker segments
                for segment, track, label in overlapping_segments.itertracks(yield_label=True):
                    if last_speaker is None:
                        last_speaker = label
                    elif label != last_speaker:
                        segments.add(Segment(current_pos, segment.start))
                        current_pos = segment.start
                        speaker_change_detected = True
                        break
                        
                if speaker_change_detected:
                    continue
                
            # Get the longest silence in this window
            max_silence = self.get_longest_silence(
                non_speech_regions, 
                window_start, 
                window_end
            )
            
            if max_silence is None:
                # If no silence found with pyannote, try pydub
                max_silence = self.get_longest_silence(
                    silence_regions, 
                    window_start, 
                    window_end
                )
                if max_silence is None:
                    # If no silence found with pydub, add the whole window
                    segments.add(Segment(current_pos, window_end))
                    current_pos = window_end
                else:
                    # End segment at middle of silence
                    silence_middle = max_silence.middle
                    segments.add(Segment(current_pos, silence_middle))
                    current_pos = silence_middle
            else:
                # End segment at middle of silence
                silence_middle = max_silence.middle
                segments.add(Segment(current_pos, silence_middle))
                current_pos = silence_middle
                
        return segments 