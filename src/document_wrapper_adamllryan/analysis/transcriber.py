import os
import torch
import datetime
import whisperx
import json
from typing import List, Dict
from document_wrapper_adamllryan.doc.document import Document
from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis
import numpy as np


class Transcriber:
    """
    Transcribes audio from a given video file and assigns speaker labels.
    """

    def __init__(self, config: Dict[str, str]):
        self.config = config

        # Ensure required config values exist
        assert "asr_model" in self.config, "ASR model not found in config"
        assert "batch_size" in self.config, "Batch size not found in config"

        # Enable CUDA if available
        use_cuda = torch.cuda.is_available() and not self.config.get(
            "test_transcriber", False
        )
        self.device = "cuda" if use_cuda else "cpu"

        print(f"Loading WhisperX model on {self.device.upper()}...")
        self.asr_model = whisperx.load_model(
            self.config["asr_model"],
            device=self.device,
            compute_type="float16" if use_cuda else "float32",
        )

        # Untested
        # Optional: Load Forced Alignment Model for Word-Level Processing
        self.use_word_level = self.config.get("use_word_level", False)
        if self.use_word_level:
            print("Loading Forced Alignment Model...")
            self.alignment_model, self.metadata = whisperx.load_align_model(
                language="en", device=self.device
            )

        # Load Speaker Diarization Model
        self.use_diarization = self.config.get("use_diarization", False)
        if self.use_diarization:
            print("Loading Speaker Diarization Model...")
            self.diarization_model = whisperx.DiarizationPipeline(
                use_auth_token=True, device=self.device
            )

        # self.recognizer = pipeline(
        #     "automatic-speech-recognition",
        #     model=self.config["asr_model"],
        #     chunk_length_s=self.config.get("chunk_length_s", None),
        #     stride_length_s=self.config.get("stride_length_s", None),
        #     batch_size=self.config["batch_size"],
        #     generate_kwargs=self.config.get("generate_kwargs", {}),
        #     device=0 if use_cuda else -1,
        #     torch_dtype="auto",
        # )
        #
        # self.diarization_pipeline = Pipeline.from_pretrained(
        #     self.config["diarization_model"], use_auth_token=True
        # ).to(torch.device("cuda" if use_cuda else "cpu"))

    def transcribe(self, video_path: str) -> Document:
        """Extracts transcript from video and assigns speakers."""

        try:
            assert os.path.exists(video_path), f"Video file not found: {video_path}"

            audio_path = self._extract_audio(video_path)

            # Run speaker diarization if enabled
            diarization = None
            if self.use_diarization:
                print("Running Speaker Diarization...")
                diarization = self.diarization_model(audio_path)

            # Run WhisperX for transcription
            print("Transcribing with WhisperX (segment-level)...")
            transcription = self.asr_model.transcribe(
                audio_path,
                batch_size=self.config["batch_size"],
            )

            # Optional: Run Forced Alignment for Word-Level Processing
            if self.use_word_level:
                print("Running Forced Alignment...")
                transcription = whisperx.align(
                    transcription["segments"],
                    self.alignment_model,
                    self.metadata,
                    audio_path,
                    self.device,
                )

            # Merge results into the expected data structure
            merged = self._merge_results(transcription, diarization)

            assert len(merged) > 0, "No transcriptions found"

            for element in merged:
                assert "text" in element, "Missing text in transcription"
                assert "timestamp" in element, "Missing timestamp in transcription"
                assert "speaker" in element, "Missing speaker in transcription"
                assert "start" in element, "Missing start in transcription"
                assert "end" in element, "Missing end in transcription"
                assert (
                    element["start"] <= element["end"]
                ), "Start time is greater than end time"

            document = DocumentAnalysis.list_to_document_from_segments(merged)

        except AssertionError as e:
            document = Document([])
            document.add_metadata("error", str(e))

        return document

    def _extract_audio(self, video_path: str) -> str:
        """Extracts audio from video using ffmpeg."""

        assert os.path.exists(video_path), f"Video file not found: {video_path}"

        audio_path = video_path.replace(".mp4", ".wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Ensure ffmpeg is installed
        assert os.system("ffmpeg -version") == 0, "ffmpeg is not installed"
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}")

        return audio_path

    def _merge_results(self, transcription, diarization) -> List[Dict]:
        """Merges ASR and diarization results to form a structured transcript."""

        transcript = []

        # Process segment-based transcription
        segments = transcription["segments"]
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()

            # Assign speaker using diarization results (if enabled)
            current_speaker = "UNKNOWN"
            if self.use_diarization:
                max_overlap = 0
                for speaker_segment in diarization["segments"]:
                    spk_start, spk_end, speaker_label = (
                        speaker_segment["start"],
                        speaker_segment["end"],
                        speaker_segment["speaker"],
                    )
                    if spk_start <= start_time <= spk_end:
                        overlap = min(spk_end, end_time) - max(spk_start, start_time)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            current_speaker = speaker_label

            transcript.append(
                {
                    "text": text,
                    "timestamp": (start_time, end_time),
                    "speaker": current_speaker,
                    "start": start_time,
                    "end": end_time,
                    "formatted_text": f"{current_speaker}: {text}",
                }
            )

        return transcript
