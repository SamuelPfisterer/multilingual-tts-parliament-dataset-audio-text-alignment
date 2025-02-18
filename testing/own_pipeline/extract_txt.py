import sys, pathlib, pymupdf




def main(): 
    fname = "data/pdf/7501579.pdf"
    with pymupdf.open(fname) as doc:  # open document
        text = chr(12).join([page.get_text("blocks", sort=True) for page in doc])
        pathlib.Path("7501579.txt").write_bytes(text.encode())
    

if __name__ == "__main__":
    main()