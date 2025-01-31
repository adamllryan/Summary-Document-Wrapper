import os
import json
from typing import List, Dict, Any
from transformers import pipeline
from ..analysis.summarizer import Summarizer
from .video_processor import VideoProcessor
from ..analysis.evaluator import Evaluator 

class BatchRunner:
    """
    Orchestrates batch processing for video summarization, integrating video processing, summarization,
    and evaluation into a single pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the batch runner with configuration.
        :param config: Dictionary containing pipeline settings (models, directories, etc.)
        """
        self.config = config
        self.video_processor = VideoProcessor(config.get("models", {}).get("transcription"))
        self.summarizer = Summarizer(config.get("summarization_model", "facebook/bart-large-cnn"))
        self.evaluator = Evaluation()
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file to extract audio, transcribe text, and extract keyframes.
        :param video_path: Path to the video file
        :return: Dictionary containing processed results (transcription, keyframes, etc.)
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.config["output_dir"], f"{base_name}.mp3")
        frames_dir = os.path.join(self.config["output_dir"], f"frames_{base_name}")

        self.video_processor.process_track("audio", video_path, audio_path)
        transcription = self.video_processor.process_track("transcription", audio_path, None)
        frames = self.video_processor.process_track("keyframes", video_path, frames_dir)

        return {"transcription": transcription, "frames": frames}
    
    def summarize_transcription(self, transcription: List[Dict[str, Any]]) -> str:
        """
        Summarizes a transcription.
        :param transcription: List of transcription segments
        :return: Summarized text
        """
        return self.summarizer.summarize_segments(transcription)
    
    def evaluate_summary(self, summary: str, reference: str) -> Dict[str, float]:
        """
        Evaluates the generated summary against a reference transcript.
        :param summary: The generated summary
        :param reference: The reference transcript (ground truth)
        :return: Dictionary with evaluation metrics (ROUGE, cosine similarity, etc.)
        """
        return self.evaluator.compute_metrics(summary, reference)
    
    def run_batch(self, video_files: List[str], reference_transcripts: Dict[str, str]) -> Dict[str, Any]:
        """
        Run batch processing on multiple video files.
        :param video_files: List of video file paths
        :param reference_transcripts: Dictionary mapping video names to reference transcripts
        :return: Dictionary containing results for all videos
        """
        results = {}
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            print(f"Processing {video_name}...")
            
            processed = self.process_video(video_path)
            summary = self.summarize_transcription(processed["transcription"])
            evaluation = self.evaluate_summary(summary, reference_transcripts.get(video_name, ""))
            
            results[video_name] = {
                "summary": summary,
                "evaluation": evaluation,
                "frames": processed["frames"]
            }
        
        # Save results
        output_path = os.path.join(self.config["output_dir"], "batch_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        
        return results

if __name__ == "__main__":
    config = {
        "models": {"transcription": "openai/whisper-small"},
        "summarization_model": "facebook/bart-large-cnn",
        "output_dir": "output/"
    }
    os.makedirs(config["output_dir"], exist_ok=True)
    
    batch_runner = BatchRunner(config)
    
    # Example usage
    video_files = ["test_video1.mp4", "test_video2.mp4"]  # Replace with actual video paths
    reference_transcripts = {"test_video1.mp4": "Expected summary text for video 1.",
                             "test_video2.mp4": "Expected summary text for video 2."}
    
    results = batch_runner.run_batch(video_files, reference_transcripts)
    print("Batch Processing Complete! Results:")
    print(json.dumps(results, indent=4))
