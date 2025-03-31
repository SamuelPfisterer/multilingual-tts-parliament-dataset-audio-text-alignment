from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class TranscriptPreprocessor(ABC):
    """Base interface for transcript preprocessing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config or {}
    
    @classmethod
    @abstractmethod
    def can_process(cls, file_path: str) -> bool:
        """Check if this preprocessor can handle the given file.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            True if this preprocessor can handle the file, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, file_path: str) -> str:
        """Preprocess the transcript file and return the cleaned text.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Preprocessed transcript text as a string
        """
        pass
