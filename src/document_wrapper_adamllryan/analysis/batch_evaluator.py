import os
import json
import time
import torch
from typing import Dict, List
# from transcriber import Transcriber
# from sentence_scorer import SentenceScorer
# from document_analysis import DocumentAnalysis
import sys 
# not set up yet
sys.exit(1)

class BatchEvaluator:
    def __init__(self, video_ids: List[str], config: Dict[str, str]):
        self.config = config
        self.video_ids = video_ids
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def run(self):
        total_videos = len(self.video_ids)

        for batch_start in range(0, total_videos):
            video_id = self.video_ids[batch_start]
            print(f"Evaluating summary for video {video_id} ({batch_start + 1} of {total_videos})")
            start_time = time.time()

            # Transcribe the summarized video
            transcriber = Transcriber(self.config["batch"]["transcriber"])
            summary_transcript = self.get_or_generate_transcript(video_id, transcriber)
            del transcriber
            torch.cuda.empty_cache()

            # Load ground truth transcript
            ground_truth_path = os.path.join(self.config["evaluate"]["video_dir"], self.config["evaluate"]["ground_truth_file"])
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth_text = f.read().strip()

            # Compute similarity score
            scorer = SentenceScorer(self.config["batch"]["sentence_scorer"])
            similarity_score = self.compute_similarity(summary_transcript, ground_truth_text, scorer)
            del scorer
            torch.cuda.empty_cache()

            # Store evaluation results
            results_path = os.path.join(self.config["output_dir"], video_id, "evaluation_results.json")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"similarity_score": similarity_score}, f, indent=4)

            elapsed_time = time.time() - start_time
            print(f"Completed evaluation for {video_id} in {elapsed_time:.2f} seconds\n")

    def get_or_generate_transcript(self, video_id: str, transcriber: Transcriber) -> str:
        transcript_path = os.path.join(self.config["output_dir"], video_id, "summary_transcript.json")
        video_path = os.path.join(self.config["evaluate"]["video_dir"], self.config["evaluate"]["summary_video_file"])
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                print("Found summary transcript")
                return " ".join(json.load(f))
        print("Generating summary transcript")
        transcript = transcriber.transcribe(video_path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=4)
        return " ".join(transcript)

    def compute_similarity(self, summary_transcript: str, ground_truth_text: str, scorer: SentenceScorer) -> float:
        scorer.assign_sentence_embeddings(summary_transcript)
        scorer.assign_sentence_embeddings(ground_truth_text)
        similarity_score = scorer.compare(summary_transcript, ground_truth_text)
        print(f"Computed similarity score: {similarity_score:.4f}")
        return similarity_score
