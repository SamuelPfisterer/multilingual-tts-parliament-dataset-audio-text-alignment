import pymupdf4llm
import pathlib
import pypandoc
import my_pymupdf4llm




def main():
    # get pdf path
    pdf_path = "data/pdf/7501579.pdf"
    md_text = my_pymupdf4llm.to_markdown(pdf_path, debug_columns=True)
    pathlib.Path("7501579.md").write_bytes(md_text.encode())

    output = pypandoc.convert_file(
        '7501579.md',
        'plain',
        outputfile='7501579.txt'
    )

if __name__ == "__main__":
    main()