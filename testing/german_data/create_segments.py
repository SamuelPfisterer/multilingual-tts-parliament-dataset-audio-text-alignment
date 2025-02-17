import os
import subprocess
from datetime import datetime
import pysrt
from pathlib import Path

def time_to_seconds(time_str):
    """Convert SRT time format (HH:MM:SS,mmm) to seconds"""
    time_obj = datetime.strptime(time_str.replace(',', '.'), '%H:%M:%S.%f')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000

def find_closest_subtitle_end(subtitles, target_seconds):
    """Find the subtitle whose end time is closest to the target duration"""
    closest_sub = None
    min_diff = float('inf')
    
    for sub in subtitles:
        end_time = time_to_seconds(str(sub.end))
        diff = abs(end_time - target_seconds)
        if diff < min_diff:
            min_diff = diff
            closest_sub = sub
            
        if end_time > target_seconds:
            break
            
    return closest_sub

def create_segment(input_audio, input_srt, output_dir, duration, actual_end_time):
    """Create audio segment and corresponding SRT file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_audio = os.path.join(output_dir, f"{base_name}_{duration}s.opus")
    output_srt = os.path.join(output_dir, f"{base_name}_{duration}s.srt")
    output_txt = os.path.join(output_dir, f"{base_name}_{duration}s.txt")
    
    # Skip if all files already exist
    if os.path.exists(output_audio) and os.path.exists(output_srt) and os.path.exists(output_txt):
        print(f"Skipping {duration}s segment - files already exist")
        return output_audio, output_srt
    
    # Cut audio file using ffmpeg if needed
    if not os.path.exists(output_audio):
        cmd = [
            'ffmpeg', '-i', input_audio,
            '-t', str(actual_end_time),
            '-c', 'copy',
            output_audio
        ]
        subprocess.run(cmd)
    
    # Create new SRT file with only the subtitles up to the target duration if needed
    if not os.path.exists(output_srt):
        subs = pysrt.open(input_srt)
        new_subs = pysrt.SubRipFile()
        
        for sub in subs:
            if time_to_seconds(str(sub.end)) <= actual_end_time:
                new_subs.append(sub)
            else:
                break
        
        new_subs.save(output_srt, encoding='utf-8')
    
    # Create TXT file if needed
    if not os.path.exists(output_txt):
        subs = pysrt.open(output_srt)
        text_content = "\n".join([sub.text for sub in subs])
        with open(output_txt, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)
    
    return output_audio, output_srt

def get_audio_duration(input_audio):
    """Get the duration of an audio file in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_audio
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def process_files(audio_path, srt_path, target_durations):
    """Process audio and SRT files for each target duration"""
    subs = pysrt.open(srt_path)
    base_dir = os.path.dirname(audio_path)
    
    # Get the original audio duration
    original_duration = get_audio_duration(audio_path)
    print(f"Original audio duration: {original_duration:.2f}s")
    
    for duration in target_durations:
        # Check if duration exceeds original audio length
        if duration > original_duration:
            print(f"Skipping {duration}s segment - exceeds original audio duration")
            continue
            
        # Find closest subtitle to target duration
        closest_sub = find_closest_subtitle_end(subs, duration)
        if closest_sub is None:
            print(f"No suitable subtitle found for duration {duration}s")
            continue
            
        actual_end_time = time_to_seconds(str(closest_sub.end))
        
        # Create output directory for this duration
        output_dir = os.path.join(base_dir, f"{duration}s_segments")
        
        # Create the segment
        output_audio, output_srt = create_segment(
            audio_path, srt_path, output_dir, duration, actual_end_time
        )
        
        print(f"Created {duration}s segment (actual duration: {actual_end_time:.2f}s)")
        print(f"Audio: {output_audio}")
        print(f"SRT: {output_srt}\n")

def main():
    # Define target durations (in seconds)
    target_durations = [
        30, 60, 120, 240, 480, 960,  # Original durations
        32 * 60,  # 32 minutes
        64 * 60,  # 64 minutes
        128 * 60,  # 128 minutes
        256 * 60,  # 256 minutes
    ]
    
    # Path to your data
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(base_path, "7501579.opus")
    srt_file = os.path.join(base_path, "7501579.srt")
    
    # Process the files
    process_files(audio_file, srt_file, target_durations)

if __name__ == "__main__":
    main()