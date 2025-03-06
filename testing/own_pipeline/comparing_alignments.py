import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# Check if Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable must be set to use Gemini LLM")

# Configuration
ALIGNMENTS_DIR = "output/alignments"
FILE1 = "7501579_3840s_aligned_old.json"
FILE2 = "7501579_3840s_aligned.json"
FRAME_DURATION = 200  # seconds
OUTPUT_DIR = "output/comparisons"
MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_alignment_data(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load JSON alignment data and create a DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (DataFrame with segment data, audio file path)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or "segments" not in data or "audio_file" not in data:
            print(f"Invalid JSON format in {file_path}: must contain 'segments' and 'audio_file' keys")
            return pd.DataFrame(), ""
            
        df = pd.DataFrame(data["segments"])
        # Compute duration (in seconds) for each segment
        df["duration"] = df["end"] - df["start"]
        audio_file = data["audio_file"]
        return df, audio_file
    except Exception as e:
        print(f"Error loading segments from {file_path}: {str(e)}")
        return pd.DataFrame(), ""

def get_segments_in_timeframe(df: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    """
    Get segments that fall within the specified time frame.
    
    Args:
        df: DataFrame containing segments
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        DataFrame with segments in the time frame
    """
    # Get segments that start within the time frame or overlap with it
    return df[
        ((df["start"] >= start_time) & (df["start"] < end_time)) |  # Starts within frame
        ((df["start"] < start_time) & (df["end"] > start_time))      # Overlaps with frame start
    ].copy()

def process_with_gemini(text: str, system_prompt: str) -> str:
    """
    Process text through Google's Gemini API with retry logic.
    
    Args:
        text: The text to process
        system_prompt: The system prompt to use
        
    Returns:
        The processed text
    """
    if not text.strip():
        return "No content to analyze."
        
    import random
    
    for attempt in range(MAX_RETRIES):
        try:
            # Initialize the Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,  # Internal retries by the LangChain client
            )
            
            # Create messages with system prompt and user content
            messages = [
                ("system", system_prompt),
                ("human", text)
            ]
            
            # Invoke the model
            ai_msg = llm.invoke(messages)
            return ai_msg.content
                
        except Exception as e:
            # Check if this is the last attempt
            if attempt == MAX_RETRIES - 1:
                print(f"Failed after {MAX_RETRIES} attempts. Last error: {e}")
                return f"Error processing content: {str(e)}"
                
            # Calculate backoff time with exponential increase and some randomness
            backoff_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 5), 60)
            
            # Check for rate limit errors specifically
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                print(f"Rate limit exceeded (attempt {attempt+1}/{MAX_RETRIES}). Waiting {backoff_time:.2f} seconds...")
                # For rate limits, we might want to wait a bit longer, but cap at 60 seconds
                backoff_time = min(backoff_time + 15, 60)
            else:
                print(f"API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"Retrying in {backoff_time:.2f} seconds...")
            
            # Wait before retrying
            time.sleep(backoff_time)

def generate_summary():
    """
    Generate a summary from the detailed comparison file.
    """
    detailed_output_path = os.path.join(OUTPUT_DIR, "detailed_comparison.txt")
    
    # Check if the detailed comparison file exists
    if not os.path.exists(detailed_output_path):
        print(f"Error: Detailed comparison file not found at {detailed_output_path}")
        print("Please run the full comparison first or specify the correct file path.")
        return
    
    # Read the detailed comparison
    with open(detailed_output_path, "r", encoding="utf-8") as f:
        full_comparison = f.read()
    
    print("Generating summary of all comparisons...")
    
    # Enhanced summary prompt with request for specific examples
    summary_prompt = """
    You are an expert in audio transcription and alignment analysis. I will provide you with a detailed comparison of two versions of an audio alignment system.
    
    Your task is to create a comprehensive summary that:
    1. Identifies the key patterns and differences between the two versions
    2. Explains the most likely reasons for performance differences
    3. Highlights systematic issues or improvements
    4. Provides recommendations for further improvements
    
    IMPORTANT: At the end of your summary, include a section titled "KEY EXAMPLES TO INVESTIGATE" that lists:
    - Specific time ranges (with exact timestamps) where the most significant differences occur
    - Segments that appear in only one version with notably poor performance
    - Examples of speaker change boundaries that may be problematic
    - Any segments with extreme CER differences between versions
    
    For each example, include:
    - The time range (start-end in seconds)
    - Which frame it appears in
    - A brief description of what makes this example notable
    - Why a human reviewer should examine this specific example
    
    Format this section as a numbered list to make it easy for researchers to prioritize their manual review.
    
    Focus on creating a clear, well-structured summary that captures the most important findings from the detailed analysis.
    """
    
    summary_input = f"""
    # Comparison Summary Request
    
    Below is a detailed comparison of two alignment versions:
    - {FILE1} (older version)
    - {FILE2} (newer version)
    
    Please provide a comprehensive summary of the key differences, patterns, and potential reasons for performance changes.
    
    {full_comparison}
    """
    
    summary_result = process_with_gemini(summary_input, summary_prompt)
    
    # Save the summary
    summary_output_path = os.path.join(OUTPUT_DIR, "summary_changes.txt")
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_result)
    
    print(f"Summary saved to {summary_output_path}")
    print("Summary generation complete!")

def compare_alignments():
    """
    Compare two alignment JSON files and generate analysis.
    """
    print(f"Comparing alignment files: {FILE1} and {FILE2}")
    
    # Load alignment data
    file1_path = os.path.join(ALIGNMENTS_DIR, FILE1)
    file2_path = os.path.join(ALIGNMENTS_DIR, FILE2)
    
    df1, audio_file1 = load_alignment_data(file1_path)
    df2, audio_file2 = load_alignment_data(file2_path)
    
    if df1.empty or df2.empty:
        print("Error: One or both alignment files could not be loaded.")
        return
    
    # Determine the total duration to analyze
    max_end_time1 = df1["end"].max()
    max_end_time2 = df2["end"].max()
    total_duration = min(max_end_time1, max_end_time2)
    
    print(f"Analyzing {total_duration:.2f} seconds of audio in {FRAME_DURATION}-second frames")
    
    # System prompt for comparison
    system_prompt = """
    You are an expert in audio transcription and alignment analysis. I will provide you with two sets of aligned audio segments from two different versions of the same alignment system.
    
    The segments contain:
    - start time (in seconds)
    - end time (in seconds)
    - ASR text (automatic speech recognition output)
    - human text (reference transcript)
    - CER (character error rate)
    
    Your task is to:
    1. Compare the two versions and identify key differences in alignment quality, segmentation, and transcription accuracy
    2. Analyze patterns in where and how the alignments differ
    3. Identify potential reasons for differences in performance
    4. Note any patterns related to speaker changes or segment boundaries
    
    Focus on:
    - Differences in segment boundaries
    - Differences in CER values
    - Differences in how text is aligned
    - Potential issues with speaker changes
    - Any systematic patterns in the differences
    
    Provide a detailed, analytical comparison with specific examples from the data.
    """
    
    # Process each time frame
    all_comparisons = []
    frame_count = int(total_duration / FRAME_DURATION) + 1
    
    for i in range(frame_count):
        start_time = i * FRAME_DURATION
        end_time = min((i + 1) * FRAME_DURATION, total_duration)
        
        print(f"Processing frame {i+1}/{frame_count}: {start_time}s to {end_time}s")
        
        # Get segments in this time frame
        segments1 = get_segments_in_timeframe(df1, start_time, end_time)
        segments2 = get_segments_in_timeframe(df2, start_time, end_time)
        
        if segments1.empty and segments2.empty:
            print(f"No segments found in frame {i+1}")
            continue
            
        # Convert segments to JSON for the LLM
        segments1_json = segments1.to_json(orient="records", indent=2)
        segments2_json = segments2.to_json(orient="records", indent=2)
        
        # Prepare input for the LLM
        comparison_input = f"""
        # Time Frame: {start_time}s to {end_time}s
        
        ## Version 1 Segments ({FILE1}):
        ```json
        {segments1_json}
        ```
        
        ## Version 2 Segments ({FILE2}):
        ```json
        {segments2_json}
        ```
        
        Please analyze the differences between these two sets of segments and explain what has changed and why the performance might differ.
        """
        
        # Process with Gemini
        print(f"Sending frame {i+1} to Gemini API...")
        comparison_result = process_with_gemini(comparison_input, system_prompt)
        
        # Add frame header
        frame_analysis = f"\n\n{'='*80}\n"
        frame_analysis += f"FRAME {i+1}: {start_time}s to {end_time}s\n"
        frame_analysis += f"{'='*80}\n\n"
        frame_analysis += comparison_result
        
        all_comparisons.append(frame_analysis)
        
        # Save individual frame analysis
        frame_output_path = os.path.join(OUTPUT_DIR, f"frame_{i+1:03d}_{start_time:.0f}s-{end_time:.0f}s.txt")
        with open(frame_output_path, "w", encoding="utf-8") as f:
            f.write(frame_analysis)
        
        # Add a small delay to avoid rate limiting
        time.sleep(2)
    
    # Combine all comparisons
    full_comparison = "\n\n".join(all_comparisons)
    
    # Save the full detailed comparison
    detailed_output_path = os.path.join(OUTPUT_DIR, "detailed_comparison.txt")
    with open(detailed_output_path, "w", encoding="utf-8") as f:
        f.write(full_comparison)
    
    print(f"Detailed comparison saved to {detailed_output_path}")
    
    # Generate summary
    generate_summary()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare alignment JSON files or generate summary.")
    parser.add_argument("--summary-only", action="store_true", help="Only generate the summary from existing detailed comparison")
    args = parser.parse_args()
    
    if args.summary_only:
        generate_summary()
    else:
        compare_alignments()
