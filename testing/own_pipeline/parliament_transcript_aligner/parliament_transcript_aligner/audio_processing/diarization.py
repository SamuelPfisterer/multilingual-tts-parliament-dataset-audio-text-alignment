import os
import torch
from pyannote.audio import Pipeline

def initialize_diarization_pipeline() -> Pipeline:
    """Initialize the pyannote diarization pipeline.
    
    Returns:
        Configured diarization pipeline
    """
    hf_token = os.getenv("HF_AUTH_TOKEN")
    cache_dir = os.getenv("HF_CACHE_DIR")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token,
        cache_dir=cache_dir
    )

    # Move to GPU if available
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return pipeline 