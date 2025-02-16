import os
import torch
import datetime
import json
from typing import List, Dict
from transformers import pipeline
from pyannote.audio import Pipeline

class Transcriber:
    """
    Transcribes audio from a given video file and assigns speaker labels.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config

        self.recognizer = pipeline(
            "automatic-speech-recognition",
            model=self.config["asr_model"],
            chunk_length_s=self.config["chunk_length_s"],
            batch_size=self.config["batch_size"],
            device=0 if torch.cuda.is_available() else -1,
        )
        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config["diarization_model"],
            use_auth_token=True
        ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def transcribe(self, video_path: str) -> List[Dict]:
        """Extracts transcript from video and assigns speakers."""
        audio_path = self._extract_audio(video_path)
        diarization = self.diarization_pipeline({'uri': f'file://{audio_path}', 'audio': audio_path})

        print("Building Transcript")
        result = self.recognizer(audio_path, return_timestamps=True)

        return self._merge_results(result, diarization)

    def _extract_audio(self, video_path: str) -> str:
        """Extracts audio from video using ffmpeg."""
        audio_path = video_path.replace(".mp4", ".wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}")
        return audio_path

    def _merge_results(self, result, diarization) -> List[Dict]:
        """Merges ASR and diarization results to form a structured transcript."""
        transcript = []

        for element in result['chunks']:
            start_time, end_time = element['timestamp']
            formatted_start_time = datetime.timedelta(seconds=start_time).total_seconds()
            formatted_end_time = datetime.timedelta(seconds=end_time).total_seconds()

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

        return transcript
