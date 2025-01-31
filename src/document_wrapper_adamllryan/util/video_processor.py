import os
import moviepy.editor as mp
import cv2
from typing import Dict, Any, List, Callable
from transformers import pipeline

class VideoProcessor:
    """
    Handles video processing tasks such as speech-to-text transcription and keyframe extraction,
    allowing modular track definitions using Hugging Face pipelines.
    """
    
    def __init__(self, model_configs: Dict[str, str]):
        """
        Initialize the video processor with specified Hugging Face models.
        :param model_configs: Dictionary mapping track names to model names
        """
        self.pipelines = {track: pipeline(model_name) for track, model_name in model_configs.items()}
        self.tracks = {}
    
    def register_track(self, track_name: str, processor: Callable[[str, str], Any]) -> None:
        """
        Register a track with a processing function.
        :param track_name: Name of the track
        :param processor: Function that takes (input_path, output_path) and processes it
        """
        self.tracks[track_name] = processor
    
    def process_track(self, track_name: str, input_path: str, output_path: str) -> Any:
        """
        Process a registered track.
        :param track_name: Name of the track to process
        :param input_path: Input file path
        :param output_path: Output file path
        :return: Processed data
        """
        if track_name in self.tracks:
            return self.tracks[track_name](input_path, output_path)
        elif track_name in self.pipelines:
            return self.pipelines[track_name](input_path)
        raise ValueError(f"Track {track_name} is not registered.")
    
    def extract_audio(self, video_path: str, audio_path: str) -> None:
        """
        Extracts audio from a video file and saves it as an MP3.
        :param video_path: Path to input video
        :param audio_path: Path to output audio file
        """
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="mp3")
    
    def extract_keyframes(self, video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
        """
        Extracts keyframes from a video at a given frame interval.
        :param video_path: Path to video file
        :param output_dir: Directory where frames will be saved
        :param frame_interval: Interval between frames to capture (default: every 30 frames)
        :return: List of extracted frame file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frame_paths = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
            frame_count += 1
        
        cap.release()
        return frame_paths

if __name__ == "__main__":
    processor = VideoProcessor({"transcription": "openai/whisper-small"})
    
    # Register tracks
    processor.register_track("audio", processor.extract_audio)
    processor.register_track("keyframes", processor.extract_keyframes)
    
    # Example Usage
    video_path = "test_video.mp4"  # Replace with an actual video path
    audio_path = "test_audio.mp3"
    frames_dir = "frames/"
    
    processor.process_track("audio", video_path, audio_path)
    transcription = processor.process_track("transcription", audio_path, None)
    frames = processor.process_track("keyframes", video_path, frames_dir)
    
    print("Transcription:", transcription)
    print("Extracted Frames:", frames)
