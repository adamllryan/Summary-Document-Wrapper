import os
import torch
import datetime
import json
from typing import List, Dict
from transformers import pipeline
from pyannote.audio import Pipeline
from document_wrapper_adamllryan.doc.document import Document
from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis
import numpy as np


class Transcriber:
    """
    Transcribes audio from a given video file and assigns speaker labels.
    """

    def __init__(self, config: Dict[str, str]):
        self.config = config

        assert "asr_model" in self.config, "ASR model not found in config"
        assert (
            "diarization_model" in self.config
        ), "Diarization model not found in config"
        assert "chunk_length_s" in self.config, "Chunk length not found in config"
        assert "batch_size" in self.config, "Batch size not found in config"

        use_cuda = torch.cuda.is_available() and not self.config.get(
            "test_transcriber", False
        )

        self.recognizer = pipeline(
            "automatic-speech-recognition",
            model=self.config["asr_model"],
            chunk_length_s=self.config.get("chunk_length_s", None),
            stride_length_s=self.config.get("stride_length_s", None),
            batch_size=self.config["batch_size"],
            generate_kwargs=self.config.get("generate_kwargs", {}),
            device=0 if use_cuda else -1,
            torch_dtype="auto",
        )

        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config["diarization_model"], use_auth_token=True
        ).to(torch.device("cuda" if use_cuda else "cpu"))

    def transcribe(self, video_path: str) -> Document:
        """Extracts transcript from video and assigns speakers."""

        try:

            assert os.path.exists(video_path), f"Video file not found: {video_path}"

            audio_path = self._extract_audio(video_path)
            diarization = self.diarization_pipeline(
                {"uri": f"file://{audio_path}", "audio": audio_path}
            )
            transcription = self.recognizer(audio_path, return_timestamps=True)

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

        # check that ffmpeg is installed
        assert os.system("ffmpeg -version") == 0, "ffmpeg is not installed"
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}")

        return audio_path

    def _merge_results(self, result, diarization) -> List[Dict]:
        """Merges ASR and diarization results to form a structured transcript."""

        # assert "chunks" in result, "Transcription result missing 'chunks' key"
        # assert "itertracks" in diarization, "Diarization result missing 'itertracks' key"

        transcript = []

        chunks = result["chunks"]
        size = len(chunks)

        for element, index in zip(chunks, range(size)):
            start_time, end_time = element["timestamp"]

            # Try to fix start time if missing
            if start_time is None:
                # If at beginning
                if index == 0:
                    start_time = 0.0
                    # If not at beginning, try to find previous end time
                elif index > 0 and chunks[index - 1]["timestamp"][1] is not None:
                    start_time = float(chunks[index - 1]["timestamp"][1])
                    # If previous end time is missing, throw an error
                else:
                    raise ValueError(f"Missing start time for chunk {index}")

            # Try to fix end time if missing
            if end_time is None:
                # If at end
                if index == size - 1:
                    end_time = float(
                        (
                            datetime.timedelta.max - datetime.timedelta(days=1)
                        ).total_seconds()
                    )
                # If not at end, try to find next start time
                elif index < size - 1 and chunks[index + 1]["timestamp"][0] is not None:
                    end_time = float(chunks[index + 1]["timestamp"][0])
                    # If next start time is missing, throw an error
                else:
                    raise ValueError(f"Missing end time for chunk {index}")

            formatted_start_time = datetime.timedelta(
                seconds=start_time
            ).total_seconds()
            formatted_end_time = datetime.timedelta(seconds=end_time).total_seconds()

            # Merge by finding the speaker with the most overlap

            max_overlap, current_speaker = 0, "UNKNOWN"
            for segment in diarization.itertracks(yield_label=True):
                ts, _, speaker_label = segment
                if ts.start <= start_time <= ts.end:
                    overlap = min(ts.end, end_time) - max(ts.start, start_time)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        current_speaker = speaker_label

            transcript.append(
                {
                    "text": element["text"].strip(),
                    "timestamp": (formatted_start_time, formatted_end_time),
                    "speaker": current_speaker,
                    "start": formatted_start_time,
                    "end": formatted_end_time,
                    "formatted_text": f"{current_speaker}: {element['text']}",
                }
            )

        # Check for None end type in last element
        if transcript[-1]["end"] is None:
            transcript[-1]["end"] = np.inf
            transcript[-1]["timestamp"] = (transcript[-1]["timestamp"][0], np.inf)

        # # Double check out of order ends
        # for i in range(len(transcript) - 1):
        #     if transcript[i]['end'] < transcript[i]['start']:
        #         transcript[i]['end'] = transcript[i + 1]['start']
        #         transcript[i]['timestamp'] = (transcript[i]['timestamp'][0], transcript[i + 1]['timestamp'][0])

        return transcript
