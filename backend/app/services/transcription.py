# backend/app/services/transcription.py

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydub import AudioSegment

logger = logging.getLogger(__name__)

AUDIO_FORMATS: Set[str] = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
VIDEO_FORMATS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024

DEFAULT_CHUNK_DURATION_MS = 10 * 60 * 1000
DEFAULT_EXPORT_FORMAT = "mp3"
DEFAULT_EXPORT_BITRATE = os.getenv("TRANSCRIPTION_MP3_BITRATE", "96k")


def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _is_ext(path: str, exts: Set[str]) -> bool:
    return Path(path).suffix.lower() in exts


def _segment_to_dict(seg: Any, *, offset_seconds: float = 0.0) -> Dict[str, Any]:
    if isinstance(seg, dict):
        start = float(seg.get("start", 0.0) or 0.0) + float(offset_seconds)
        end = float(seg.get("end", 0.0) or 0.0) + float(offset_seconds)
        text = str(seg.get("text", "") or "")
        return {"start": start, "end": end, "text": text}

    start = float(getattr(seg, "start", 0.0) or 0.0) + float(offset_seconds)
    end = float(getattr(seg, "end", 0.0) or 0.0) + float(offset_seconds)
    text = str(getattr(seg, "text", "") or "")
    return {"start": start, "end": end, "text": text}


def _word_to_dict(w: Any, *, offset_seconds: float = 0.0) -> Dict[str, Any]:
    if isinstance(w, dict):
        start = float(w.get("start", 0.0) or 0.0) + float(offset_seconds)
        end = float(w.get("end", 0.0) or 0.0) + float(offset_seconds)
        word = str(w.get("word", "") or "")
        return {"start": start, "end": end, "word": word}

    start = float(getattr(w, "start", 0.0) or 0.0) + float(offset_seconds)
    end = float(getattr(w, "end", 0.0) or 0.0) + float(offset_seconds)
    word = str(getattr(w, "word", "") or "")
    return {"start": start, "end": end, "word": word}


class TranscriptionService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.model = (model or os.getenv("TRANSCRIPTION_MODEL") or "whisper-1").strip()

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not configured. Transcription is unavailable.")
            self.client = None
            return

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            logger.error("openai package not installed or incompatible: %s", e)
            self.client = None
            return

        self.client = OpenAI(api_key=self.api_key)
        logger.info("TranscriptionService initialized with model=%s", self.model)

    def is_audio_file(self, file_path: str) -> bool:
        return _is_ext(file_path, AUDIO_FORMATS)

    def is_video_file(self, file_path: str) -> bool:
        return _is_ext(file_path, VIDEO_FORMATS)

    def needs_transcription(self, file_path: str) -> bool:
        return self.is_audio_file(file_path) or self.is_video_file(file_path)

    def extract_audio_from_video(self, video_path: str, *, export_bitrate: str = DEFAULT_EXPORT_BITRATE) -> str:
        if not video_path:
            raise ValueError("video_path is empty")

        temp_dir = tempfile.gettempdir()
        audio_filename = f"{Path(video_path).stem}_audio_{os.getpid()}.mp3"
        audio_path = os.path.join(temp_dir, audio_filename)

        logger.info("Extracting audio from video: %s", video_path)
        try:
            video = AudioSegment.from_file(video_path)
            video.export(audio_path, format="mp3", bitrate=export_bitrate)
            logger.info("Audio extracted: %s", audio_path)
            return audio_path
        except Exception as e:
            _safe_unlink(audio_path)
            raise RuntimeError(f"Audio extraction failed: {e}") from e

    def split_audio_file(
        self,
        audio_path: str,
        *,
        chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
        export_format: str = DEFAULT_EXPORT_FORMAT,
        export_bitrate: str = DEFAULT_EXPORT_BITRATE,
    ) -> List[Tuple[str, float]]:
        if not audio_path:
            raise ValueError("audio_path is empty")

        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        if total_ms <= 0:
            raise RuntimeError("Audio appears empty or unreadable")

        temp_dir = tempfile.gettempdir()
        chunks: List[Tuple[str, float]] = []

        logger.info("Splitting audio into chunks: %s", audio_path)

        i = 0
        for start_ms in range(0, total_ms, int(chunk_duration_ms)):
            end_ms = min(total_ms, start_ms + int(chunk_duration_ms))
            chunk = audio[start_ms:end_ms]

            chunk_filename = f"{Path(audio_path).stem}_chunk_{i}_{os.getpid()}.{export_format}"
            chunk_path = os.path.join(temp_dir, chunk_filename)

            chunk.export(chunk_path, format=export_format, bitrate=export_bitrate)

            if os.path.getsize(chunk_path) > MAX_FILE_SIZE_BYTES:
                _safe_unlink(chunk_path)
                half = max(30_000, int((end_ms - start_ms) / 2))
                if half >= (end_ms - start_ms):
                    raise RuntimeError("Chunking failed to reduce size below limit.")

                sub_audio = audio[start_ms:end_ms]
                sub_temp_path = os.path.join(temp_dir, f"{Path(audio_path).stem}_sub_{i}_{os.getpid()}.wav")
                sub_audio.export(sub_temp_path, format="wav")
                try:
                    sub_chunks = self.split_audio_file(
                        sub_temp_path,
                        chunk_duration_ms=half,
                        export_format=export_format,
                        export_bitrate=export_bitrate,
                    )
                    for p, off in sub_chunks:
                        chunks.append((p, (start_ms / 1000.0) + off))
                finally:
                    _safe_unlink(sub_temp_path)
            else:
                chunks.append((chunk_path, start_ms / 1000.0))

            i += 1

        logger.info("Created %d chunk(s)", len(chunks))
        return chunks

    def _require_client(self) -> None:
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY and openai package.")

    def _transcribe_one(
        self,
        file_path: str,
        *,
        language: Optional[str],
        prompt: Optional[str],
        with_timestamps: bool,
        word_timestamps: bool,
    ) -> Dict[str, Any]:
        self._require_client()

        timestamp_granularities: List[str] = []
        response_format = "json"

        if with_timestamps:
            response_format = "verbose_json"
            timestamp_granularities.append("segment")
            if word_timestamps:
                timestamp_granularities.append("word")

        with open(file_path, "rb") as f:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "file": f,
                "response_format": response_format,
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            if timestamp_granularities:
                kwargs["timestamp_granularities"] = timestamp_granularities

            resp = self.client.audio.transcriptions.create(**kwargs)

        if isinstance(resp, dict):
            text_out = str(resp.get("text", "") or "").strip()
            segs = resp.get("segments", None)
            wrds = resp.get("words", None)
            duration = resp.get("duration", None)
            lang = resp.get("language", None)
        else:
            text_out = (getattr(resp, "text", None) or "").strip()
            segs = getattr(resp, "segments", None)
            wrds = getattr(resp, "words", None)
            duration = getattr(resp, "duration", None)
            lang = getattr(resp, "language", None)

        segments_out: List[Dict[str, Any]] = []
        words_out: List[Dict[str, Any]] = []

        if with_timestamps and segs:
            for s in segs:
                segments_out.append(_segment_to_dict(s, offset_seconds=0.0))

        if with_timestamps and word_timestamps and wrds:
            for w in wrds:
                words_out.append(_word_to_dict(w, offset_seconds=0.0))

        return {
            "text": text_out,
            "duration": float(duration) if isinstance(duration, (int, float)) else None,
            "language": str(lang) if lang else (language or None),
            "segments": segments_out,
            "words": words_out,
        }

    def transcribe_file(
        self,
        file_path: str,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        with_timestamps: bool = False,
        word_timestamps: bool = False,
        export_bitrate: str = DEFAULT_EXPORT_BITRATE,
    ) -> Dict[str, Any]:
        self._require_client()

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        original_path = file_path
        temp_audio_path: Optional[str] = None

        try:
            if self.is_video_file(file_path):
                temp_audio_path = self.extract_audio_from_video(file_path, export_bitrate=export_bitrate)
                audio_path = temp_audio_path
                source_type = "video"
            else:
                audio_path = file_path
                source_type = "audio"

            size_bytes = os.path.getsize(audio_path)
            logger.info(
                "Transcription input=%s size=%.2fMB model=%s",
                audio_path,
                size_bytes / (1024 * 1024),
                self.model,
            )

            if size_bytes <= MAX_FILE_SIZE_BYTES:
                one = self._transcribe_one(
                    audio_path,
                    language=language,
                    prompt=prompt,
                    with_timestamps=with_timestamps,
                    word_timestamps=word_timestamps,
                )
                return {
                    "text": one["text"],
                    "segments": one.get("segments", []) if with_timestamps else [],
                    "words": one.get("words", []) if (with_timestamps and word_timestamps) else [],
                    "metadata": {
                        "model": self.model,
                        "source_type": source_type,
                        "original_file": str(original_path),
                        "audio_file_used": str(audio_path),
                        "file_size_bytes": int(size_bytes),
                        "chunks": 1,
                        "duration": one.get("duration"),
                        "language": one.get("language"),
                        "with_timestamps": bool(with_timestamps),
                        "word_timestamps": bool(word_timestamps),
                    },
                }

            chunks = self.split_audio_file(audio_path, export_bitrate=export_bitrate)
            all_text: List[str] = []
            all_segments: List[Dict[str, Any]] = []
            all_words: List[Dict[str, Any]] = []

            final_language: Optional[str] = None

            for chunk_path, offset_s in chunks:
                one = self._transcribe_one(
                    chunk_path,
                    language=language,
                    prompt=prompt,
                    with_timestamps=with_timestamps,
                    word_timestamps=word_timestamps,
                )
                if one.get("text"):
                    all_text.append(str(one["text"]))

                if with_timestamps:
                    for s in one.get("segments", []) or []:
                        all_segments.append(_segment_to_dict(s, offset_seconds=offset_s))
                    if word_timestamps:
                        for w in one.get("words", []) or []:
                            all_words.append(_word_to_dict(w, offset_seconds=offset_s))

                if not final_language and one.get("language"):
                    final_language = str(one["language"])

                _safe_unlink(chunk_path)

            stitched_text = " ".join(t.strip() for t in all_text if t and t.strip()).strip()
            final_duration = float(max((s.get("end", 0.0) or 0.0) for s in all_segments)) if (with_timestamps and all_segments) else None

            return {
                "text": stitched_text,
                "segments": all_segments if with_timestamps else [],
                "words": all_words if (with_timestamps and word_timestamps) else [],
                "metadata": {
                    "model": self.model,
                    "source_type": source_type,
                    "original_file": str(original_path),
                    "audio_file_used": str(audio_path),
                    "file_size_bytes": int(size_bytes),
                    "chunks": int(len(chunks)),
                    "duration": final_duration,
                    "language": final_language or language,
                    "with_timestamps": bool(with_timestamps),
                    "word_timestamps": bool(word_timestamps),
                },
            }

        finally:
            if temp_audio_path and temp_audio_path != original_path:
                _safe_unlink(temp_audio_path)

    def transcribe_with_timestamps(
        self,
        file_path: str,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        return self.transcribe_file(
            file_path,
            language=language,
            prompt=prompt,
            with_timestamps=True,
            word_timestamps=word_timestamps,
        )


_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service


def transcribe_media_file(
    file_path: str,
    *,
    language: Optional[str] = None,
    with_timestamps: bool = False,
    word_timestamps: bool = False,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    svc = get_transcription_service()
    return svc.transcribe_file(
        file_path,
        language=language,
        prompt=prompt,
        with_timestamps=with_timestamps,
        word_timestamps=word_timestamps,
    )


__all__ = [
    "TranscriptionService",
    "get_transcription_service",
    "transcribe_media_file",
]