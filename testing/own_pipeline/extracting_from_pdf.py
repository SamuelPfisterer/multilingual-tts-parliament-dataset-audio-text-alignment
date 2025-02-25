import pymupdf4llm
import pathlib
import pypandoc




def main():
    # get pdf path
    pdf_path = "data/pdf/norway-10168-1.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    pathlib.Path("norway-10168-1.md").write_bytes(md_text.encode())

    output = pypandoc.convert_file(
        'norway-10168-1.md',
        'plain',
        outputfile='norway-10168-1.txt'
    )

if __name__ == "__main__":
    main()