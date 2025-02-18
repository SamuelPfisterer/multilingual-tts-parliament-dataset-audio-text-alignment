import pymupdf4llm
import pathlib
import pypandoc




def main():
    # get pdf path
    pdf_path = "data/pdf/7501579.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    pathlib.Path("output.md").write_bytes(md_text.encode())

    output = pypandoc.convert_file(
        'output.md',
        'plain',
        outputfile='output.txt'
    )

if __name__ == "__main__":
    main()