# Description: Batch processing of videos. This class orchestrates the entire pipeline for a batch of videos.

import json
import os 
import time 
from typing import List, Dict 
import cv2 
import torch
# import warnings
import logging
from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis 
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
        self.documents: Dict[str, Document] = {}
        os.makedirs(self.config["output_dir"], exist_ok=True)
        self.transcriber = None
        self.summarizer = None
        self.scorer = None
        self.keyframe_extractor = None
        self.filterer = None
        self.splicer = None

        if config["suppress_torch"]:
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

            logging.getLogger("pyannote").setLevel(logging.ERROR)

            # warnings.simplefilter("ignore", category=FutureWarning)
            # warnings.simplefilter("ignore", category=UserWarning)

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

            # Check if we have partially completed batches

            for video_id in batch:
                doc_path = os.path.join(self.config["output_dir"], video_id, self.config["output_filename"])
                if os.path.exists(doc_path):
                    with open(doc_path, "r", encoding="utf-8") as f:
                        transcript_data = json.load(f)
                    # check metadata for error before loading
                    if transcript_data["metadata"].get("error"):
                        print(f"Error in document {video_id}: {transcript_data['metadata']['error']}")
                        batch.remove(video_id)
                        continue
                    self.documents[video_id] = DocumentAnalysis.list_to_document_from_processed(transcript_data["sentences"], transcript_data["metadata"])

            # Step 1: Transcription -> Creates Document objects
            for video_id in batch:
                self.get_or_generate_transcript(video_id)
            if self.transcriber:
                del self.transcriber
                torch.cuda.empty_cache()

            # Step 2: Summarization -> Updates Document objects
            for video_id in batch:
                self.get_or_generate_summary(video_id)
            if self.summarizer:
                del self.summarizer
                torch.cuda.empty_cache()

            # Step 3: Sentence Scoring -> Updates Document objects
            for video_id in batch:
                self.get_or_generate_sentence_scores(video_id)
            if self.scorer:
                del self.scorer
                torch.cuda.empty_cache()

            # Step 4: Keyframe Extraction -> Updates Document objects
            for video_id in batch:
                self.get_or_generate_keyframes(video_id)
            if self.keyframe_extractor:
                del self.keyframe_extractor

            # Step 5: Filtering -> Updates Document objects
            for video_id in batch:
                self.filter_sentences(video_id)
            if self.filterer:
                del self.filterer

            # Step 6: Video Splicing
            for video_id in batch:
                self.create_spliced_video(video_id)
            if self.splicer:
                del self.splicer

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




    def get_or_generate_transcript(self, video_id: str):
        """
        Generates or retrieves a Document containing the transcript.
        """

        output_path = os.path.join(self.config["output_dir"], video_id, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if transcript already exists
        if self.documents.get(video_id):
            if all(s.get_track("text") and s.get_track("text").get_data().get("text", "").strip() for s in self.documents[video_id].sentences):
                print(f"Transcript already exists for video: {video_id}, skipping.")
                return

        # Lazy load transcriber
        if self.transcriber is None:
            self.transcriber = Transcriber(self.config["transcriber"])

        # Generate transcript
        video_path = os.path.join(self.config["video_dir"], video_id, self.config["video_filename"])
        print(f"Generating new transcript for video: {video_id}")

        self.documents[video_id] = self.transcriber.transcribe(video_path)

        # Write aggregated output.json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.documents[video_id].export(), f, indent=4)

    def get_or_generate_summary(self, video_id: str):
        """
        Generates or retrieves a summary.
        """

        output_path = os.path.join(self.config["output_dir"], video_id, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if summary already exists
        if self.documents[video_id] and "summary" in self.documents[video_id].metadata:
            print(f"Summary already exists for video: {video_id}, skipping.")
            return

        # Lazy load summarizer
        if self.summarizer is None:
            self.summarizer = Summarizer(self.config["summarizer"])

        # Generate summary
        print(f"Generating new summary for video: {video_id}")
        document_text = "\n".join(str(s) for s in self.documents[video_id].sentences)
        summary = self.summarizer.summarize(document_text)

        # Store summary in Document metadata
        self.documents[video_id].add_metadata("summary", summary)

        # Write aggregated output.json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.documents[video_id].export(), f, indent=4)

    def get_or_generate_sentence_scores(self, video_id: str):
        """
        Computes or loads sentence similarity scores.
        """

        output_path = os.path.join(self.config["output_dir"], video_id, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("Text score: ", self.documents[video_id].call_track_method("get_embeddings", "text"))
        # Check if scores and embeddings exist
        if self.documents[video_id] and all(s["text"] is not None for s in self.documents[video_id].call_track_method("get_score", "text")) and all(len(s["text"]) > 0 for s in self.documents[video_id].call_track_method("get_embeddings", "text")):
            print(f"Sentence scores already exist for video: {video_id}, skipping.")
            return

        # Lazy load scorer
        if self.scorer is None:
            self.scorer = SentenceScorer(self.config["sentence_scorer"])

        # Compute embeddings and sentence scores
        print(f"Computing sentence embeddings and scores for video: {video_id}")
        self.scorer.score(self.documents[video_id])

        # Write aggregated output.json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.documents[video_id].export(), f, indent=4)
    
    def get_or_generate_keyframes(self, video_id: str):
        """
        Computes or loads keyframe counts per sentence and updates the KeyframeTrack.
        """

        output_path = os.path.join(self.config["output_dir"], video_id, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.documents[video_id] and all(s["keyframe"] is not None for s in self.documents[video_id].call_track_method("get_score", "keyframe")):
            print(f"Keyframe scores already exist for video: {video_id}, skipping.")
            return

        # Lazy load keyframe extractor
        if self.keyframe_extractor is None:
            self.keyframe_extractor = KeyframeExtractor(self.config["keyframe_extractor"])

        # Extract keyframe counts
        print(f"Computing keyframe counts for video: {video_id}")
        video_path = os.path.join(self.config["video_dir"], video_id, self.config["video_filename"])
        self.keyframe_extractor.extract(video_path, self.documents[video_id])

        # Write aggregated output.json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.documents[video_id].export(), f, indent=4)

    def filter_sentences(self, video_id: str):
        """
        Filters sentences based on their scores and updates the Document metadata.
        """

        output_path = os.path.join(self.config["output_dir"], video_id, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if filtering is already completed
        if self.documents[video_id] and "filtered_sentences" in self.documents[video_id].metadata:
            print(f"Filtered sentences already exist for video: {video_id}, skipping.")
            return

        # Lazy load filterer
        if self.filterer is None:
            self.filterer = Filter(self.config["filterer"])

        # Filter sentences based on scores
        print(f"Filtering sentences for video: {video_id}")
        self.filterer.apply(self.documents[video_id])# .sentences, sentence_scores)

        # Store filtered sentences in metadata
        # self.documents[video_id].add_metadata("filtered_sentences", [(s.start, s.end) for s in filtered_sentences])

        # Write aggregated output.json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.documents[video_id].export(), f, indent=4)


    def create_spliced_video(self, video_id: str):
        """
        Creates a spliced video based on the filtered sentences.
        """

        video_path = os.path.join(self.config["video_dir"], video_id, self.config["video_filename"])
        spliced_video_path = os.path.join(self.config["output_dir"], video_id, self.config["spliced_video_filename"])

        # Check if spliced video already exists
        if os.path.exists(spliced_video_path):
            print(f"Spliced video already exists for video: {video_id}, skipping.")
            return

        # Lazy load splicer
        if self.splicer is None:
            self.splicer = Splicer(self.config["splicer"])

        # Get filtered sentences
        filtered_sentences = [
            s for s in self.documents[video_id].sentences
            if s.call_track_method("get_score", "text") is not None
        ]

        # Extract timestamps from filtered sentences
        timestamps = [(s.start, s.end) for s in filtered_sentences]

        # Perform splicing
        print(f"Creating spliced video for video: {video_id}")
        self.splicer.splice(video_path, timestamps, spliced_video_path)
