import re
import sys
import pathlib
import pypandoc

def clean_transcript(transcript):
    """
    Processes a parliamentary transcript to remove annotations and extract spoken text.

    Args:
        transcript (str): The raw markdown transcript.

    Returns:
        str: A cleaned markdown transcript with most annotations removed.
    """

    # 1. Remove page numbers and headers/footers
    transcript = re.sub(r"-----.*Deutscher Bundestag.*Sitzung.*Berlin,.*-----\n?", "", transcript, flags=re.MULTILINE)

    # 2. Remove table of contents-like sections
    transcript = re.sub(r"I n h a l t :.*?Tagesordnung \.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\. 26207 D", "", transcript, flags=re.DOTALL)

    # 3. Remove Tagesordnungspunkt listings, Zusatzpunkte and Anlage
    transcript = re.sub(r"Tagesordnungspunkt \d+:\n.*?(\n{2,})", r"\1", transcript, flags=re.DOTALL)
    transcript = re.sub(r"Zusatzpunkt \d+:\n.*?(\n{2,})", r"\1", transcript, flags=re.DOTALL)
    transcript = re.sub(r"Anlage \d+\n.*?(\n{2,})", r"\1", transcript, flags=re.DOTALL)

    # 4.  Remove Drucksache lines
    transcript = re.sub(r"Drucksache \d+/\d+ \.+( \.){2,} \d+ [A-D]\n", "", transcript)
    transcript = re.sub(r"Überweisungsvorschlag:.*?Ausschussordnung\n", "", transcript, flags=re.DOTALL)

    # 5. Remove seemingly unnecessary line breaks, multiple whitespace.
    transcript = re.sub(r'\n{3,}', '\n\n', transcript)
    transcript = re.sub(r' +', ' ', transcript)

    # 6. Speaker annotation removal
    transcript = re.sub(r"(\w+ \w+)( \([\w\/ ]+\)):", r"**\1**:", transcript)
    transcript = re.sub(r"(\w+ \w+),", r"**\1**,", transcript)
    transcript = re.sub(r"(\w+ \w+),", r"**\1**,", transcript)

    # 7. Clean sections containing only A, B, C or D.
    transcript = re.sub(r"^\(?[A-D]\)?\n", "", transcript, flags=re.MULTILINE)

    return transcript

def process_markdown_file(input_md_path):
    """
    Process a markdown file by cleaning the transcript and converting to txt.
    
    Args:
        input_md_path (str): Path to the input markdown file
    """
    # Create path objects
    input_path = pathlib.Path(input_md_path)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file {input_md_path} does not exist.")
        return
    
    # Generate output filenames
    cleaned_md_path = input_path.with_stem(f"{input_path.stem}_cleaned")
    output_txt_path = input_path.with_stem(f"{input_path.stem}_cleaned").with_suffix('.txt')
    
    # Read the markdown file
    md_content = input_path.read_text(encoding='utf-8')
    
    # Clean the transcript
    cleaned_md = clean_transcript(md_content)
    
    # Write the cleaned markdown to a temporary file
    cleaned_md_path.write_text(cleaned_md, encoding='utf-8')
    
    # Convert the cleaned markdown to plain text using pypandoc
    pypandoc.convert_file(
        str(cleaned_md_path),
        'plain',
        outputfile=str(output_txt_path)
    )
    
    
    
    print(f"Processed {input_md_path} and saved as {output_txt_path}")

def main():
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_markdown_file>")
        return
    
    input_md_path = sys.argv[1]
    process_markdown_file(input_md_path)

if __name__ == "__main__":
    main()
