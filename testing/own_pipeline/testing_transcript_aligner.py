from parliament_transcript_aligner.parliament_transcript_aligner import (
    AlignmentPipeline, 

    AudioSegmenter, 
    initialize_vad_pipeline, 
    initialize_diarization_pipeline,

    TranscriptAligner,
    create_preprocessor,

    save_alignments, 
    save_transcribed_segments, 
    load_transcribed_segments,

    TranscribedSegment,
    AlignedTranscript
)


def main(): 
    print('hello')



if __name__ == "__main__":
    main()



'''
Code from main.py script: 
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
    file_name = "7501579_1920s"
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
'''