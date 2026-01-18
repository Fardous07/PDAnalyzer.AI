"""
Transcription service using OpenAI Whisper API
Handles audio and video file transcription
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import openai
from pydub import AudioSegment
import tempfile

logger = logging.getLogger(__name__)

# Supported audio formats
AUDIO_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}

# Supported video formats (will extract audio)
VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

# OpenAI Whisper file size limit (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024


class TranscriptionService:
    """Service for transcribing audio/video files using OpenAI Whisper"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize transcription service
        
        Args:
            api_key: OpenAI API key. If None, reads from environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured. Transcription will not be available.")
        else:
            openai.api_key = self.api_key
            logger.info("Transcription service initialized with OpenAI Whisper")
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is an audio format"""
        ext = Path(file_path).suffix.lower()
        return ext in AUDIO_FORMATS
    
    def is_video_file(self, file_path: str) -> bool:
        """Check if file is a video format"""
        ext = Path(file_path).suffix.lower()
        return ext in VIDEO_FORMATS
    
    def needs_transcription(self, file_path: str) -> bool:
        """Check if file needs transcription"""
        return self.is_audio_file(file_path) or self.is_video_file(file_path)
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file (mp3)
        """
        logger.info(f"Extracting audio from video: {video_path}")
        
        try:
            # Create temporary file for audio
            temp_dir = tempfile.gettempdir()
            audio_filename = f"{Path(video_path).stem}_audio.mp3"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            # Load video and extract audio
            video = AudioSegment.from_file(video_path)
            
            # Export as mp3
            video.export(audio_path, format="mp3")
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    def split_audio_file(self, audio_path: str, chunk_duration_ms: int = 600000) -> list[str]:
        """
        Split large audio file into chunks (10 minutes each by default)
        
        Args:
            audio_path: Path to audio file
            chunk_duration_ms: Chunk duration in milliseconds (default: 10 minutes)
            
        Returns:
            List of paths to audio chunks
        """
        logger.info(f"Splitting large audio file: {audio_path}")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            chunks = []
            temp_dir = tempfile.gettempdir()
            
            # Split into chunks
            for i, start_time in enumerate(range(0, len(audio), chunk_duration_ms)):
                chunk = audio[start_time:start_time + chunk_duration_ms]
                
                chunk_filename = f"{Path(audio_path).stem}_chunk_{i}.mp3"
                chunk_path = os.path.join(temp_dir, chunk_filename)
                
                chunk.export(chunk_path, format="mp3")
                chunks.append(chunk_path)
                
                logger.info(f"Created chunk {i+1}: {chunk_path}")
            
            logger.info(f"Split audio into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split audio file: {e}")
            raise Exception(f"Audio splitting failed: {str(e)}")
    
    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio or video file using OpenAI Whisper
        
        Args:
            file_path: Path to audio/video file
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dictionary with 'text' and 'metadata'
        """
        if not self.api_key:
            raise Exception("OpenAI API key not configured. Cannot transcribe.")
        
        logger.info(f"Starting transcription for: {file_path}")
        
        try:
            # Check if it's a video - extract audio first
            if self.is_video_file(file_path):
                logger.info("Video file detected, extracting audio...")
                audio_path = self.extract_audio_from_video(file_path)
                original_file = file_path
            else:
                audio_path = file_path
                original_file = file_path
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {file_size / (1024*1024):.2f} MB")
            
            # If file is too large, split it
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File exceeds {MAX_FILE_SIZE / (1024*1024)} MB limit, splitting...")
                chunks = self.split_audio_file(audio_path)
                
                # Transcribe each chunk
                full_transcript = []
                for i, chunk_path in enumerate(chunks):
                    logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                    
                    with open(chunk_path, "rb") as audio_file:
                        response = openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language=language,
                            prompt=prompt,
                            response_format="verbose_json"
                        )
                    
                    full_transcript.append(response.text)
                    
                    # Clean up chunk
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
                
                transcript_text = " ".join(full_transcript)
                
                # Create metadata
                metadata = {
                    "duration": None,
                    "language": response.language if hasattr(response, 'language') else language,
                    "chunks": len(chunks)
                }
            else:
                # Transcribe single file
                with open(audio_path, "rb") as audio_file:
                    response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        prompt=prompt,
                        response_format="verbose_json"
                    )
                
                transcript_text = response.text
                
                # Create metadata
                metadata = {
                    "duration": getattr(response, 'duration', None),
                    "language": getattr(response, 'language', language),
                    "chunks": 1
                }
            
            # Clean up extracted audio if it was from video
            if self.is_video_file(original_file) and audio_path != original_file:
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            logger.info(f"Transcription completed. Text length: {len(transcript_text)} chars")
            
            return {
                "text": transcript_text,
                "metadata": metadata
            }
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error during transcription: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def transcribe_with_timestamps(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe with word-level timestamps (if needed for advanced features)
        
        Args:
            file_path: Path to audio/video file
            language: Optional language code
            
        Returns:
            Dictionary with 'text', 'segments', and 'metadata'
        """
        if not self.api_key:
            raise Exception("OpenAI API key not configured. Cannot transcribe.")
        
        logger.info(f"Starting transcription with timestamps: {file_path}")
        
        try:
            # Handle video files
            if self.is_video_file(file_path):
                audio_path = self.extract_audio_from_video(file_path)
            else:
                audio_path = file_path
            
            # Check file size - timestamps only work with single file
            file_size = os.path.getsize(audio_path)
            if file_size > MAX_FILE_SIZE:
                logger.warning("File too large for timestamp transcription, using regular transcription")
                return self.transcribe_file(file_path, language)
            
            # Transcribe with timestamps
            with open(audio_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Extract segments with timestamps
            segments = []
            if hasattr(response, 'segments'):
                for segment in response.segments:
                    segments.append({
                        "start": segment.get('start'),
                        "end": segment.get('end'),
                        "text": segment.get('text', '')
                    })
            
            # Clean up extracted audio if needed
            if self.is_video_file(file_path) and audio_path != file_path:
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            return {
                "text": response.text,
                "segments": segments,
                "metadata": {
                    "duration": getattr(response, 'duration', None),
                    "language": getattr(response, 'language', language)
                }
            }
            
        except Exception as e:
            logger.error(f"Timestamp transcription error: {e}")
            # Fallback to regular transcription
            return self.transcribe_file(file_path, language)


# Singleton instance
_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get or create transcription service singleton"""
    global _transcription_service
    
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    
    return _transcription_service


def transcribe_media_file(
    file_path: str,
    language: Optional[str] = None,
    with_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to transcribe a media file
    
    Args:
        file_path: Path to audio/video file
        language: Optional language code
        with_timestamps: Whether to include timestamps
        
    Returns:
        Dictionary with transcription results
    """
    service = get_transcription_service()
    
    if with_timestamps:
        return service.transcribe_with_timestamps(file_path, language)
    else:
        return service.transcribe_file(file_path, language)