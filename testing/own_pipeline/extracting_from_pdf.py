import pymupdf4llm
import pathlib
import pypandoc
import my_pymupdf4llm




def main():
    # get pdf path
    system_prompt = """
    You are a multilingual assistant specialized in processing parliamentary transcripts. Your task is to clean the provided transcript page by removing all unnecessary metadata, annotations etc. while preserving only the literal spoken dialogue. Please follow these instructions:

    Remove the speaker labels that appear as headers before each speaker’s dialogue. These are the names (and titles, or other metadata) to indicate who is speaking.
    Remove all annotations, procedural notes, timestamps, and non-verbal cues (e.g., “[Laughter]”, “(Interruption)”).
    Typically, it is quite clear which parts of the transcript are actual spoken words and which are background annotations or cues that are not present in the audio’s main dialogue. Text in brackets usually represents these non-spoken elements and can be safely removed.
    However, if you are unsure whether a piece of text is part of the spoken dialogue, please keep it.
    Ensure that only and all the spoken dialogue is in your response.
    Respond in the same language as the input and do not alter the spoken text.
    """
    pdf_path = "data/pdf/7501579.pdf"
    md_text = my_pymupdf4llm.to_markdown(pdf_path, debug_columns=True, use_llm=True, llm_system_prompt= system_prompt)
    pathlib.Path("7501579_llm.md").write_bytes(md_text.encode())

    output = pypandoc.convert_file(
        '7501579_llm.md',
        'plain',
        outputfile='7501579_llm.txt'
    )

if __name__ == "__main__":
    main()