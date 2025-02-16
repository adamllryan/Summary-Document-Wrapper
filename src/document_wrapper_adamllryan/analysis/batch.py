# Description: Batch processing of videos. This class orchestrates the entire pipeline for a batch of videos.

import json
import os 
import time 
from typing import List, Dict 
import cv2 
import torch
from document_wrapper_adamllryan.analysis.document_analysis import DocumentAnalysis 
from document_wrapper_adamllryan.analysis.filter import Filter 
from document_wrapper_adamllryan.analysis.keyframe_extractor import KeyframeExtractor 
from document_wrapper_adamllryan.analysis.sentence_scorer import SentenceScorer 
from document_wrapper_adamllryan.analysis.splicer import Splicer 
from document_wrapper_adamllryan.analysis.summarizer import Summarizer 
from document_wrapper_adamllryan.analysis.transcriber import Transcriber 
from document_wrapper_adamllryan.doc.document import Document 



class BatchExecutor:
    def __init__(self, video_ids: List[str], config: Dict[str, str]):
        self.config = config
        self.video_ids = video_ids
        self.batch_size = config.get("batch_size", 1)
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def run(self):
        total_videos = len(self.video_ids)
        print(f"Total videos: {total_videos}")

        # metrics capture
        original_length = 0
        final_length = 0

        for batch_start in range(0, total_videos, self.batch_size):
            batch = self.video_ids[batch_start:batch_start + self.batch_size]
            print(f"Processing batch {batch_start + 1} to {batch_start + len(batch)} of {total_videos}")
            start_time = time.time()

            # Transcription. Should return dict of documents corresponding to video_id
            transcriber = Transcriber(self.config["transcriber"])
            documents = {video_id: self.get_or_generate_transcript(video_id, transcriber) for video_id in batch}
            del transcriber
            torch.cuda.empty_cache()

            # Summarization. Should return dict of summaries corresponding to video_id
            summarizer = Summarizer(self.config["summarizer"])
            summaries = {video_id: self.get_or_generate_summary(video_id, documents[video_id], summarizer) for video_id in batch}
            del summarizer
            torch.cuda.empty_cache()

            # Sentence Scoring. Should return dict of scores corresponding to video_id
            scorer = SentenceScorer(self.config["sentence_scorer"])
            sentence_scores = {video_id: self.get_or_generate_sentence_scores(video_id, documents[video_id], summaries[video_id], scorer) for video_id in batch}
            del scorer
            torch.cuda.empty_cache()

            # Keyframe Extraction. Should return a dict
            keyframe_extractor = KeyframeExtractor(self.config["keyframe_extractor"])
            keyframes = {video_id: self.get_or_generate_keyframes(video_id, documents[video_id], keyframe_extractor) for video_id in batch}
            del keyframe_extractor
            torch.cuda.empty_cache()

            # Filtering
            filterer = Filter(self.config["filterer"])
            filtered_sentences = {video_id: self.filter_sentences(documents[video_id], sentence_scores[video_id], filterer) for video_id in batch}
            del filterer
            torch.cuda.empty_cache()

            # Video Splicing
            splicer = Splicer(self.config["splicer"])
            for video_id in batch:
                self.create_spliced_video(video_id, filtered_sentences[video_id], splicer)
            del splicer
            torch.cuda.empty_cache()

            elapsed_time = time.time() - start_time
            print(f"Completed batch {batch_start + 1} to {batch_start + len(batch)} in {elapsed_time:.2f} seconds\n")

            # Metrics display

            for video_id in batch:
                original_path = os.path.join(self.config["output_dir"], video_id, self.config["video_filename"])
                final_path = os.path.join(self.config["output_dir"], video_id, self.config["spliced_video_filename"])
                original_length += cv2.VideoCapture(original_path).get(cv2.CAP_PROP_FRAME_COUNT)
                final_length += cv2.VideoCapture(final_path).get(cv2.CAP_PROP_FRAME_COUNT)

            print(f"Original video aggregate length: {original_length}")
            print(f"Final video aggregate length: {final_length}")

    def get_or_generate_transcript(self, video_id: str, transcriber: Transcriber) -> Document:
        transcript_path = os.path.join(self.config["output_dir"], video_id, self.config["transcript_filename"])
        video_path = os.path.join(self.config["video_dir"], video_id, self.config["video_filename"])
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                print("Found transcript")
                return DocumentAnalysis.list_to_document(json.load(f))
        print("Generating transcript")
        transcript = transcriber.transcribe(video_path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=4)
        return DocumentAnalysis.list_to_document(transcript)

    def get_or_generate_summary(self, video_id: str, document: Document, summarizer: Summarizer) -> str:
        summary_path = os.path.join(self.config["output_dir"], video_id, self.config["summary_filename"])

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                print("Found summary")
                return f.read()
        print("Generating summary")
        summary = summarizer.summarize("\n".join([str(s) for s in document.sentences]))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return summary

    def get_or_generate_sentence_scores(self, video_id: str, document: Document, summary: str, scorer: SentenceScorer) -> Dict[str, float]:
        """Computes or loads sentence similarity scores."""
        sentence_scores_path = os.path.join(self.config["output_dir"], video_id, "sentence_scores.json")

        if os.path.exists(sentence_scores_path):
            with open(sentence_scores_path, "r", encoding="utf-8") as f:
                print("Found sentence scores")
                return json.load(f)
        print("Computing sentence scores")
        scorer.assign_sentence_embeddings(document)
        print("Computing overall scores")
        sentence_scores = scorer.score(document.sentences, summary)
        with open(sentence_scores_path, "w", encoding="utf-8") as f:
            json.dump(sentence_scores, f, indent=4)
        return sentence_scores

    def get_or_generate_keyframes(self, video_id: str, document: Document, keyframe_extractor: KeyframeExtractor) -> List[int]:
      """Computes or loads keyframe counts per sentence."""
      keyframe_path = os.path.join(self.config["output_dir"], video_id, "keyframes.json")

      if os.path.exists(keyframe_path):
          with open(keyframe_path, "r", encoding="utf-8") as f:
              print("Found keyframe counts")
              return json.load(f)
      print("Computing keyframe counts")
      keyframe_counts = keyframe_extractor.extract(os.path.join(self.config["video_dir"], video_id, self.config["video_filename"]), document.sentences)

      with open(keyframe_path, "w", encoding="utf-8") as f:
          json.dump(keyframe_counts, f, indent=4)
      return keyframe_counts

    def filter_sentences(self, document: Document, sentence_scores: Dict[str, float], filterer: Filter) -> List:
        """Filters sentences based on the computed scores."""
        print("Filtering sentences")

        filtered_sentences = filterer.apply(document.sentences, sentence_scores)

        return filtered_sentences

    def create_spliced_video(self, video_id: str, filtered_sentences: List, splicer: Splicer):
        print("Creating spliced video")
        video_path = os.path.join(self.config["video_dir"], video_id, self.config["video_filename"])
        spliced_video_path = os.path.join(self.config["output_dir"], video_id, self.config["spliced_video_filename"])
        if os.path.exists(spliced_video_path):
            print("Found spliced video")
            return

        timestamps = [(s.start, s.end) for s in filtered_sentences]
        splicer.splice(video_path, timestamps, spliced_video_path)
