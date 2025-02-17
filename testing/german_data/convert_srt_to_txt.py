import os
import pysrt

def convert_srt_to_txt(srt_path):
    """Convert SRT file to TXT file containing only the text."""
    subs = pysrt.open(srt_path)
    text_content = "\n".join([sub.text for sub in subs])
    
    # Create the output TXT file path
    txt_path = os.path.splitext(srt_path)[0] + '.txt'
    
    # Write the text content to the TXT file
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)
    
    print(f"Created TXT file: {txt_path}")

def process_directory(directory):
    """Process all SRT files in the specified directory and its subdirectories."""
    for sub_directory in os.listdir(directory):
        sub_directory_path = os.path.join(directory, sub_directory)
        if os.path.isdir(sub_directory_path):
            # Process the sub-directory
            process_directory(sub_directory_path)
        elif sub_directory.endswith('.srt'):
            srt_file_path = os.path.join(directory, sub_directory)
            convert_srt_to_txt(srt_file_path)

def main():
    # Define the base path where the SRT files are located
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Process the directory
    process_directory(base_path)

if __name__ == "__main__":
    main() 