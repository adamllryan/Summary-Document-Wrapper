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
        assert "diarization_model" in self.config, "Diarization model not found in config"
        assert "chunk_length_s" in self.config, "Chunk length not found in config"
        assert "batch_size" in self.config, "Batch size not found in config"

        use_cuda = torch.cuda.is_available() and not self.config.get("test_transcriber", False)

        self.recognizer = pipeline(
            "automatic-speech-recognition",
            model=self.config["asr_model"],
            chunk_length_s=self.config["chunk_length_s"],
            batch_size=self.config["batch_size"],
            device=0 if use_cuda else -1
        )

        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config["diarization_model"],
            use_auth_token=True
        ).to(torch.device("cuda" if use_cuda else "cpu"))

    def transcribe(self, video_path: str) -> Document:
        """Extracts transcript from video and assigns speakers."""

        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        
        audio_path = self._extract_audio(video_path)
        diarization = self.diarization_pipeline({'uri': f'file://{audio_path}', 'audio': audio_path})
        transcription = self.recognizer(audio_path, return_timestamps=True)

        merged = self._merge_results(transcription, diarization)

        return DocumentAnalysis.list_to_document_from_segments(merged)


    def _extract_audio(self, video_path: str) -> str:
        """Extracts audio from video using ffmpeg."""

        assert os.path.exists(video_path), f"Video file not found: {video_path}"

        audio_path = video_path.replace(".mp4", ".wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}")
        return audio_path

    def _merge_results(self, result, diarization) -> List[Dict]:
        """Merges ASR and diarization results to form a structured transcript."""

        # assert "chunks" in result, "Transcription result missing 'chunks' key"
        # assert "itertracks" in diarization, "Diarization result missing 'itertracks' key"

        transcript = []

        for element in result['chunks']:
            start_time, end_time = element['timestamp']
            formatted_start_time = datetime.timedelta(seconds=start_time).total_seconds()

            # Try to find the next start time and set that as missing end time

            if end_time is None:
                idx = result['chunks'].index(element)
                while result['chunks'][idx]['timestamp'][1] is None and idx < len(result['chunks']) - 1:
                    idx += 1 
                if result['chunks'][idx]['timestamp'][1] is not None:
                    end_time = result['chunks'][idx]['timestamp'][1]
                else: # if missing, skip this element
                    continue
            
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

            transcript.append({
                'text': element['text'].strip(),
                'timestamp': (formatted_start_time, formatted_end_time),
                'speaker': current_speaker,
                'start': formatted_start_time,
                'end': formatted_end_time,
                'formatted_text': f"{current_speaker}: {element['text']}"
            })

        # Check for None end type in last element
        if transcript[-1]['end'] is None:
            transcript[-1]['end'] = np.inf
            transcript[-1]['timestamp'] = (transcript[-1]['timestamp'][0], np.inf)

        return transcript
