import os
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

def initialize_vad_pipeline() -> VoiceActivityDetection:
    """Initialize the pyannote VAD pipeline.
    
    Returns:
        Configured VAD pipeline
    """
    cache_dir = os.getenv("HF_CACHE_DIR")
    
    # First load the model with cache_dir
    model = Model.from_pretrained(
        "pyannote/segmentation",
        use_auth_token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    
    # Then create the VAD pipeline using the loaded model
    vad_pipeline = VoiceActivityDetection(segmentation=model)
    
    # Configure the pipeline parameters
    vad_pipeline.instantiate({
        "onset": 0.9,
        "offset": 0.9,
        "min_duration_on": 0.2,
        "min_duration_off": 0.0
    })
    
    return vad_pipeline 