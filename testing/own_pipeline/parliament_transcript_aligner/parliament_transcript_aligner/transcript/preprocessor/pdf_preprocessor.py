from pathlib import Path
from .base import TranscriptPreprocessor

class PdfPreprocessor(TranscriptPreprocessor):
    """Preprocessor for PDF transcript files."""
    
    DEFAULT_SYSTEM_PROMPT = """
    You are a multilingual assistant specialized in processing parliamentary transcripts. 
    Your task is to clean the provided transcript page by removing all unnecessary metadata, 
    annotations etc. while preserving only the literal spoken dialogue. Please follow these instructions:

    Remove the speaker labels that appear as headers before each speaker's dialogue.
    Remove all annotations, procedural notes, timestamps, and non-verbal cues.
    Ensure that only and all the spoken dialogue is in your response.
    Respond in the same language as the input and do not alter the spoken text.
    """
    
    @classmethod
    def can_process(cls, file_path: str) -> bool:
        """Check if the file is a PDF.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            True if the file has a .pdf extension
        """
        return Path(file_path).suffix.lower() == '.pdf'
    
    def preprocess(self, file_path: str) -> str:
        """Preprocess a PDF transcript file.
        
        Uses pymupdf4llm to extract text, with optional LLM cleaning.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Preprocessed transcript text
        """
        try:
            import my_pymupdf4llm
            import pypandoc
        except ImportError:
            raise ImportError("Please install pymupdf4llm and pypandoc for PDF processing")
        
        # Check if LLM processing is enabled
        use_llm = self.config.get('use_llm', False)
        
        # Define system prompt for LLM if enabled
        system_prompt = None
        if use_llm:
            # Allow custom system prompt from config
            system_prompt = self.config.get('system_prompt', self.DEFAULT_SYSTEM_PROMPT)
        
        # Extract text from PDF
        md_text = my_pymupdf4llm.to_markdown(
            file_path, 
            debug_columns=self.config.get('debug_columns', False),
            use_llm=use_llm,
            llm_system_prompt=system_prompt if use_llm else None
        )
        
        # Convert markdown to plain text if requested
        if self.config.get('convert_to_plain', True):
            # Create a temporary file
            temp_md_path = f"{Path(file_path).stem}_temp.md"
            Path(temp_md_path).write_bytes(md_text.encode())
            
            # Convert to plain text
            txt_path = f"{Path(file_path).stem}_processed.txt"
            pypandoc.convert_file(
                temp_md_path,
                'plain',
                outputfile=txt_path
            )
            
            # Read the plain text result
            with open(txt_path, 'r', encoding='utf-8') as f:
                result = f.read()
                
            # Clean up temporary files
            if not self.config.get('keep_temp_files', False):
                Path(temp_md_path).unlink(missing_ok=True)
                if not self.config.get('keep_output_file', False):
                    Path(txt_path).unlink(missing_ok=True)
                    
            return result
        
        # Return markdown text if plain text conversion not requested
        return md_text
