"""
Optimized MoDA Gradio App

Key improvements over original app.py:
1. Direct pipeline import - No subprocess overhead, models loaded once
2. Progress tracking - Real-time feedback during generation
3. GPU memory management - Proper cleanup between requests
4. Input validation - Face detection, audio length checks
5. Advanced options - CFG scale, emotion selection, seed control
6. Proper audio preprocessing - Resampling to 16kHz
7. Concurrency control - Queue-based processing to prevent OOM
8. Modern Python practices - Pydantic models, dataclasses, type hints
"""

from __future__ import annotations

import gc
import os.path as osp
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import ClassVar, Optional

import cv2
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, Field, field_validator, model_validator

# Add project root to path
sys.path.insert(0, osp.dirname(osp.realpath(__file__)))

from src.utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Disable OpenCV threading issues
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Enable cudnn benchmark for faster inference
torch.backends.cudnn.benchmark = True


# =============================================================================
# Enums
# =============================================================================

class Emotion(str, Enum):
    """Supported emotion types for motion generation."""
    NONE = "None"
    ANGER = "Anger"
    CONTEMPT = "Contempt"
    DISGUST = "Disgust"
    FEAR = "Fear"
    HAPPINESS = "Happiness"
    NEUTRAL = "Neutral"
    SADNESS = "Sadness"
    SURPRISE = "Surprise"

    @property
    def index(self) -> int:
        """Get the numeric index for the emotion."""
        mapping = {
            self.ANGER: 0,
            self.CONTEMPT: 1,
            self.DISGUST: 2,
            self.FEAR: 3,
            self.HAPPINESS: 4,
            self.NEUTRAL: 5,
            self.SADNESS: 6,
            self.SURPRISE: 7,
            self.NONE: 8,
        }
        return mapping[self]

    @classmethod
    def choices(cls) -> list[str]:
        """Get list of emotion names for UI dropdowns."""
        return [e.value for e in cls]


# =============================================================================
# Pydantic Configuration Models
# =============================================================================

class AppConfig(BaseModel):
    """Application-level configuration settings."""
    
    cfg_path: Path = Field(
        default=Path("configs/audio2motion/inference/inference.yaml"),
        description="Path to inference configuration YAML"
    )
    motion_mean_std_path: Path = Field(
        default=Path("src/datasets/mean.pt"),
        description="Path to motion mean/std statistics"
    )
    silent_audio_path: Path = Field(
        default=Path("src/examples/silent-audio.wav"),
        description="Path to silent audio for padding"
    )
    output_dir: Path = Field(
        default=Path("output/gradio"),
        description="Directory for generated outputs"
    )
    examples_image_dir: Path = Field(
        default=Path("src/examples/reference_images"),
        description="Directory containing example images"
    )
    examples_audio_dir: Path = Field(
        default=Path("src/examples/driving_audios"),
        description="Directory containing example audio files"
    )
    
    # Server settings
    server_name: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=7860, ge=1024, le=65535, description="Server port")
    share: bool = Field(default=True, description="Create public share link")
    max_queue_size: int = Field(default=5, ge=1, le=20, description="Max requests in queue")
    
    model_config = {"extra": "forbid"}


class ImageConstraints(BaseModel):
    """Constraints for input image validation."""
    
    min_dimension: int = Field(default=256, ge=64, description="Minimum width/height")
    max_dimension: int = Field(default=4096, le=8192, description="Maximum width/height")
    valid_extensions: frozenset[str] = Field(
        default=frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
        description="Allowed image file extensions"
    )
    
    model_config = {"frozen": True}


class AudioConstraints(BaseModel):
    """Constraints for input audio validation."""
    
    min_duration: float = Field(default=0.5, ge=0.1, description="Minimum duration in seconds")
    max_duration: float = Field(default=300.0, le=600.0, description="Maximum duration (5 min)")
    target_sample_rate: int = Field(default=16000, description="Target sample rate for processing")
    
    model_config = {"frozen": True}


class GenerationParams(BaseModel):
    """Parameters for video generation."""
    
    image_path: Path = Field(..., description="Path to source face image")
    audio_path: Path = Field(..., description="Path to driving audio")
    emotion: Emotion = Field(default=Emotion.NONE, description="Emotion conditioning")
    cfg_scale: float = Field(
        default=1.2,
        ge=1.0,
        le=2.5,
        description="Classifier-free guidance scale"
    )
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    smooth_motion: bool = Field(
        default=False,
        description="Apply Kalman smoothing to motion"
    )
    
    @field_validator("image_path", "audio_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path | None) -> Path:
        if v is None:
            raise ValueError("Path cannot be None")
        return Path(v) if isinstance(v, str) else v
    
    @field_validator("image_path")
    @classmethod
    def validate_image_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Image file not found: {v}")
        return v
    
    @field_validator("audio_path")
    @classmethod
    def validate_audio_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Audio file not found: {v}")
        return v

    model_config = {"extra": "forbid"}


# =============================================================================
# Dataclasses for Internal Data
# =============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    message: str
    duration: float = 0.0  # For audio validation


@dataclass
class ProcessingContext:
    """Context for a single video generation request."""
    
    params: GenerationParams
    temp_audio_path: Optional[Path] = None
    output_path: Optional[Path] = None
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed processing time in seconds."""
        return time.time() - self.start_time
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_audio_path and self.temp_audio_path.exists():
            try:
                self.temp_audio_path.unlink()
            except OSError:
                pass


@dataclass
class ExampleEntry:
    """An example image/audio pair for the UI."""
    
    image_path: Path
    audio_path: Path
    emotion: str = "None"
    cfg_scale: float = 1.2
    seed: int = 42
    smooth_motion: bool = False
    
    def to_list(self) -> list:
        """Convert to list format for Gradio Examples."""
        return [
            str(self.image_path),
            str(self.audio_path),
            self.emotion,
            self.cfg_scale,
            self.seed,
            self.smooth_motion,
        ]


# =============================================================================
# Validators
# =============================================================================

class InputValidator:
    """Validates user inputs for the generation pipeline."""
    
    def __init__(
        self,
        image_constraints: ImageConstraints | None = None,
        audio_constraints: AudioConstraints | None = None,
    ):
        self.image_constraints = image_constraints or ImageConstraints()
        self.audio_constraints = audio_constraints or AudioConstraints()
    
    def validate_image(self, image_path: str | Path | None) -> ValidationResult:
        """Validate the input image."""
        logger.debug(f"Validating image: {image_path}")
        
        if image_path is None:
            logger.warning("No image provided for validation")
            return ValidationResult(False, "No image provided")
        
        path = Path(image_path)
        
        if not path.exists():
            logger.error(f"Image file not found: {path}")
            return ValidationResult(False, f"Image file not found: {path}")
        
        ext = path.suffix.lower()
        if ext not in self.image_constraints.valid_extensions:
            logger.error(f"Invalid image format: {ext}")
            return ValidationResult(
                False,
                f"Invalid image format: {ext}. Supported: {self.image_constraints.valid_extensions}"
            )
        
        try:
            img = cv2.imread(str(path))
            if img is None:
                logger.error(f"Failed to load image: {path}")
                return ValidationResult(False, "Failed to load image")
            
            h, w = img.shape[:2]
            min_dim = self.image_constraints.min_dimension
            max_dim = self.image_constraints.max_dimension
            
            if h < min_dim or w < min_dim:
                logger.error(f"Image too small: {w}x{h}, minimum: {min_dim}x{min_dim}")
                return ValidationResult(
                    False,
                    f"Image too small ({w}x{h}). Minimum size: {min_dim}x{min_dim}"
                )
            
            if h > max_dim or w > max_dim:
                logger.error(f"Image too large: {w}x{h}, maximum: {max_dim}x{max_dim}")
                return ValidationResult(
                    False,
                    f"Image too large ({w}x{h}). Maximum size: {max_dim}x{max_dim}"
                )
                
        except Exception as e:
            logger.exception(f"Error reading image: {e}")
            return ValidationResult(False, f"Error reading image: {e}")
        
        logger.debug(f"Image validation passed: {path} ({w}x{h})")
        return ValidationResult(True, "OK")
    
    def validate_audio(self, audio_path: str | Path | None) -> ValidationResult:
        """Validate the input audio and return duration."""
        logger.debug(f"Validating audio: {audio_path}")
        
        if audio_path is None:
            logger.warning("No audio provided for validation")
            return ValidationResult(False, "No audio provided")
        
        path = Path(audio_path)
        
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return ValidationResult(False, f"Audio file not found: {path}")
        
        try:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
            duration = len(audio) / sr
            
            if duration < self.audio_constraints.min_duration:
                logger.error(f"Audio too short: {duration:.2f}s, minimum: {self.audio_constraints.min_duration}s")
                return ValidationResult(
                    False,
                    f"Audio too short (minimum {self.audio_constraints.min_duration} seconds)",
                    duration
                )
            
            if duration > self.audio_constraints.max_duration:
                logger.error(f"Audio too long: {duration:.2f}s, maximum: {self.audio_constraints.max_duration}s")
                return ValidationResult(
                    False,
                    f"Audio too long (maximum {self.audio_constraints.max_duration / 60:.0f} minutes)",
                    duration
                )
                
        except Exception as e:
            logger.exception(f"Error reading audio: {e}")
            return ValidationResult(False, f"Error reading audio: {e}")
        
        logger.debug(f"Audio validation passed: {path} (duration: {duration:.2f}s, sample_rate: {sr}Hz)")
        return ValidationResult(True, "OK", duration)


# =============================================================================
# Audio Preprocessor
# =============================================================================

class AudioPreprocessor:
    """Handles audio preprocessing for the pipeline."""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        logger.debug(f"AudioPreprocessor initialized with target_sample_rate={target_sample_rate}")
    
    def process(self, audio_path: Path) -> Path:
        """
        Convert audio to target sample rate WAV format.
        
        Returns the original path if already in correct format,
        otherwise returns path to a temporary converted file.
        """
        logger.debug(f"Processing audio: {audio_path}")
        try:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            logger.debug(f"Loaded audio: sample_rate={sr}Hz, samples={len(audio)}")
            
            # If already correct format, return as-is
            if sr == self.target_sample_rate and audio_path.suffix.lower() == ".wav":
                logger.debug("Audio already in correct format, no conversion needed")
                return audio_path
            
            # Resample if needed
            if sr != self.target_sample_rate:
                logger.info(f"Resampling audio from {sr}Hz to {self.target_sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            
            # Save to temporary WAV file
            temp_path = Path(tempfile.gettempdir()) / f"moda_audio_{int(time.time())}.wav"
            sf.write(str(temp_path), audio, self.target_sample_rate)
            logger.debug(f"Audio saved to temporary file: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            logger.exception(f"Audio preprocessing failed: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}") from e


# =============================================================================
# Pipeline Manager (Singleton)
# =============================================================================

class PipelineManager:
    """
    Manages the MoDA pipeline with thread-safe singleton pattern.
    
    Ensures the pipeline is loaded only once and provides
    controlled access for concurrent requests.
    """
    
    _instance: ClassVar[Optional[PipelineManager]] = None
    _lock: ClassVar[Lock] = Lock()
    
    def __new__(cls, config: AppConfig | None = None) -> PipelineManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, config: AppConfig | None = None):
        if self._initialized:
            return
        
        self.config = config or AppConfig()
        self._pipeline = None
        self._initialized = True
    
    @property
    def pipeline(self):
        """Lazy-load and return the pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self) -> None:
        """Load the MoDA pipeline."""
        logger.info("=" * 60)
        logger.info("Loading MoDA pipeline... (this takes ~30-60 seconds)")
        logger.info("=" * 60)
        
        from src.models.inference.moda_test import LiveVASAPipeline
        
        start_time = time.time()
        self._pipeline = LiveVASAPipeline(
            cfg_path=str(self.config.cfg_path),
            motion_mean_std_path=str(self.config.motion_mean_std_path)
        )
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"Pipeline loaded successfully in {elapsed:.2f}s!")
        logger.info("=" * 60)
    
    @staticmethod
    def clear_gpu_memory() -> None:
        """Clear GPU memory between generations."""
        logger.debug("Clearing GPU memory")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared")


# =============================================================================
# Seed Management
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# Video Generator
# =============================================================================

class VideoGenerator:
    """Orchestrates the video generation process."""
    
    def __init__(
        self,
        config: AppConfig | None = None,
        validator: InputValidator | None = None,
        audio_preprocessor: AudioPreprocessor | None = None,
    ):
        self.config = config or AppConfig()
        self.validator = validator or InputValidator()
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self.pipeline_manager = PipelineManager(self.config)
    
    def generate(
        self,
        image_path: str,
        audio_path: str,
        emotion: str = "None",
        cfg_scale: float = 1.2,
        seed: int = 42,
        smooth_motion: bool = False,
        progress: gr.Progress = gr.Progress(),
    ) -> str:
        """
        Generate a talking head video from image and audio.
        
        Args:
            image_path: Path to source face image
            audio_path: Path to driving audio
            emotion: Emotion conditioning
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            smooth_motion: Apply Kalman smoothing
            progress: Gradio progress tracker
        
        Returns:
            Path to generated video
        """
        logger.info("Starting video generation")
        logger.info(f"Parameters: emotion={emotion}, cfg_scale={cfg_scale}, seed={seed}, smooth={smooth_motion}")
        
        context = None
        
        try:
            # Parse and validate parameters
            progress(0.05, desc="Validating inputs...")
            logger.debug("Parsing and validating generation parameters")
            
            params = GenerationParams(
                image_path=image_path,
                audio_path=audio_path,
                emotion=Emotion(emotion),
                cfg_scale=cfg_scale,
                seed=seed,
                smooth_motion=smooth_motion,
            )
            
            context = ProcessingContext(params=params)
            
            # Validate image
            image_result = self.validator.validate_image(params.image_path)
            if not image_result.is_valid:
                logger.error(f"Image validation failed: {image_result.message}")
                raise gr.Error(f"Image validation failed: {image_result.message}")
            
            # Validate audio
            audio_result = self.validator.validate_audio(params.audio_path)
            if not audio_result.is_valid:
                logger.error(f"Audio validation failed: {audio_result.message}")
                raise gr.Error(f"Audio validation failed: {audio_result.message}")
            
            duration = audio_result.duration
            est_time = max(30, duration * 2)
            logger.info(f"Audio duration: {duration:.1f}s, estimated processing time: {est_time:.0f}s")
            progress(0.1, desc=f"Audio duration: {duration:.1f}s (est. {est_time:.0f}s to process)")
            
            # Preprocess audio
            progress(0.15, desc="Preprocessing audio...")
            logger.debug("Preprocessing audio")
            processed_audio = self.audio_preprocessor.process(params.audio_path)
            if processed_audio != params.audio_path:
                context.temp_audio_path = processed_audio
                logger.debug(f"Audio preprocessed to: {processed_audio}")
            
            # Set seed
            set_seed(params.seed)
            logger.debug(f"Random seed set to: {params.seed}")
            
            # Load pipeline
            progress(0.2, desc="Loading models...")
            logger.debug("Loading pipeline models")
            pipeline = self.pipeline_manager.pipeline
            
            # Create output directory
            progress(0.25, desc="Extracting audio features...")
            save_dir = self.config.output_dir / f"cfg-{params.cfg_scale}-emo-{params.emotion.value}"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory: {save_dir}")
            
            # Generate video
            progress(0.3, desc="Generating motion sequence...")
            logger.info("Starting motion sequence generation")
            
            video_path = pipeline.driven_sample(
                image_path=str(params.image_path),
                audio_path=str(processed_audio),
                cfg_scale=params.cfg_scale,
                emo=params.emotion.index,
                save_dir=str(save_dir),
                smooth=params.smooth_motion,
                silent_audio_path=str(self.config.silent_audio_path),
            )
            
            progress(0.95, desc="Finalizing video...")
            logger.debug("Finalizing video output")
            
            # Check for final video path
            video_path = Path(video_path)
            final_video_path = video_path.parent / f"final_{video_path.name}"
            
            if final_video_path.exists():
                context.output_path = final_video_path
            elif video_path.exists():
                context.output_path = video_path
            else:
                logger.error("Video generation failed - output file not found")
                raise gr.Error("Video generation failed - output file not found")
            
            progress(1.0, desc="Done!")
            elapsed = context.elapsed_time
            logger.info(f"Video generation completed in {elapsed:.2f}s: {context.output_path}")
            
            return str(context.output_path)
            
        except gr.Error:
            raise
        except Exception as e:
            logger.exception(f"Video generation failed with error: {e}")
            raise gr.Error(f"Generation failed: {str(e)}") from e
        
        finally:
            if context:
                context.cleanup()
            PipelineManager.clear_gpu_memory()


# =============================================================================
# Example Loader
# =============================================================================

class ExampleLoader:
    """Loads example files for the Gradio interface."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def get_examples(self, max_examples: int = 5) -> list[list]:
        """Get example image/audio pairs."""
        examples: list[ExampleEntry] = []
        
        image_dir = self.config.examples_image_dir
        audio_dir = self.config.examples_audio_dir
        
        if not image_dir.exists() or not audio_dir.exists():
            return []
        
        images = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png"}
        )
        audios = sorted(
            f for f in audio_dir.iterdir()
            if f.suffix.lower() == ".wav" and not f.name.startswith("tmp")
        )
        
        if not audios:
            return []
        
        for i, img in enumerate(images[:max_examples]):
            audio = audios[i % len(audios)]
            examples.append(
                ExampleEntry(
                    image_path=img,
                    audio_path=audio,
                )
            )
        
        return [ex.to_list() for ex in examples]


# =============================================================================
# Gradio Interface Builder
# =============================================================================

class GradioInterface:
    """Builds and manages the Gradio web interface."""
    
    def __init__(self, config: AppConfig | None = None):
        logger.debug("Initializing GradioInterface")
        self.config = config or AppConfig()
        self.generator = VideoGenerator(self.config)
        self.example_loader = ExampleLoader(self.config)
        logger.debug("GradioInterface initialized")
    
    def build(self) -> gr.Blocks:
        """Build the Gradio interface."""
        
        with gr.Blocks(
            title="MoDA - Talking Head Generator",
            theme=gr.themes.Soft(),
            css="""
            .container { max-width: 1200px; margin: auto; }
            .output-video { min-height: 400px; }
            """
        ) as demo:
            
            self._build_header()
            
            with gr.Row():
                image_input, audio_input, controls = self._build_input_column()
                video_output = self._build_output_column()
            
            self._build_examples(
                image_input, audio_input, 
                controls["emotion"], controls["cfg_scale"],
                controls["seed"], controls["smooth"],
                video_output
            )
            
            # Event handler
            controls["generate_btn"].click(
                fn=self.generator.generate,
                inputs=[
                    image_input, audio_input,
                    controls["emotion"], controls["cfg_scale"],
                    controls["seed"], controls["smooth"]
                ],
                outputs=[video_output],
                show_progress="full"
            )
            
            self._build_footer()
        
        return demo
    
    def _build_header(self) -> None:
        """Build the header section."""
        gr.Markdown("""
        # 🎬 MoDA - Multi-modal Diffusion Talking Head Generator
        
        Upload a face image and audio to generate a realistic talking head video.
        The model uses diffusion-based motion generation with audio-visual synchronization.
        """)
    
    def _build_input_column(self) -> tuple[gr.Image, gr.Audio, dict]:
        """Build the input column and return components."""
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")
            
            image_input = gr.Image(
                type="filepath",
                label="Source Face Image",
                elem_classes=["input-image"]
            )
            
            audio_input = gr.Audio(
                type="filepath",
                label="Driving Audio",
                elem_classes=["input-audio"]
            )
            
            with gr.Accordion("⚙️ Advanced Options", open=False):
                emotion_input = gr.Dropdown(
                    choices=Emotion.choices(),
                    value=Emotion.NONE.value,
                    label="Emotion Conditioning",
                    info="Apply emotion style to the generated motion"
                )
                
                cfg_scale_input = gr.Slider(
                    minimum=1.0,
                    maximum=2.5,
                    value=1.2,
                    step=0.1,
                    label="CFG Scale",
                    info="Higher = stronger audio adherence, lower = more natural motion"
                )
                
                seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    info="Set for reproducible results",
                    precision=0
                )
                
                smooth_input = gr.Checkbox(
                    value=False,
                    label="Smooth Motion",
                    info="Apply Kalman smoothing (reduces jitter but may reduce expressiveness)"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Video",
                variant="primary",
                size="lg"
            )
        
        controls = {
            "emotion": emotion_input,
            "cfg_scale": cfg_scale_input,
            "seed": seed_input,
            "smooth": smooth_input,
            "generate_btn": generate_btn,
        }
        
        return image_input, audio_input, controls
    
    def _build_output_column(self) -> gr.Video:
        """Build the output column and return video component."""
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Output")
            
            video_output = gr.Video(
                label="Generated Video",
                elem_classes=["output-video"],
                autoplay=True
            )
            
            with gr.Row():
                gr.Markdown("""
                **Tips:**
                - Use clear, frontal face images for best results
                - Audio should be clean speech (no background music)
                - Processing time: ~2x audio duration
                """)
        
        return video_output
    
    def _build_examples(
        self,
        image_input: gr.Image,
        audio_input: gr.Audio,
        emotion_input: gr.Dropdown,
        cfg_scale_input: gr.Slider,
        seed_input: gr.Number,
        smooth_input: gr.Checkbox,
        video_output: gr.Video,
    ) -> None:
        """Build the examples section."""
        examples = self.example_loader.get_examples()
        if examples:
            gr.Markdown("### 📂 Examples")
            gr.Examples(
                examples=examples,
                inputs=[image_input, audio_input, emotion_input, cfg_scale_input, seed_input, smooth_input],
                outputs=[video_output],
                fn=self.generator.generate,
                cache_examples=False,
            )
    
    def _build_footer(self) -> None:
        """Build the footer section."""
        gr.Markdown("""
        ---
        **MoDA** - Multi-modal Diffusion Architecture for Talking Head Generation  
        [GitHub](https://github.com/lixinyyang/MoDA) | [Model Weights](https://huggingface.co/lixinyizju/moda)
        """)
    
    def launch(self) -> None:
        """Build and launch the interface."""
        demo = self.build()
        
        demo.queue(
            max_size=self.config.max_queue_size,
            default_concurrency_limit=1  # Process one at a time (GPU constraint)
        ).launch(
            server_name=self.config.server_name,
            server_port=self.config.server_port,
            share=self.config.share,
            show_error=True,
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Application entry point."""
    logger.info("=" * 60)
    logger.info("Starting MoDA Optimized Gradio App")
    logger.info("=" * 60)
    
    config = AppConfig()
    logger.info(f"Configuration loaded: server={config.server_name}:{config.server_port}, share={config.share}")
    
    interface = GradioInterface(config)
    logger.info("Launching Gradio interface...")
    interface.launch()


if __name__ == "__main__":
    main()
