
from typing import List
import numpy as np 
from document_wrapper_adamllryan.doc.document import Document


class Filter:
    """
    Filters sentences in a Document based on a dynamically computed threshold from the score distribution.
    """

    def __init__(self, config: dict):
        self.config = config

    def apply(self, document: Document, threshold: float = None):
        """
        Filters sentences in the document based on a dynamic threshold.

        Args:
            document (Document): The document containing sentences and scores.
            threshold (float, optional): A predefined threshold; if None, it is computed dynamically.
        """

        print("Filtering sentences")

        # Extract scores from text track
        text_scores = {
            tuple(sentence.timestamp): sentence.call_track_method("get_score", "text")["text"]
            for sentence in document.sentences
            if sentence.call_track_method("get_score", "text") is not None
        }

        # Extract scores from keyframe track
        keyframe_scores = {
            tuple(sentence.timestamp): sentence.call_track_method("get_score", "keyframe")["keyframe"]
            for sentence in document.sentences
            if sentence.call_track_method("get_score", "keyframe") is not None
        }

        # Normalize keyframe scores to [0, 1]
        if keyframe_scores:
            max_score = max(keyframe_scores.values())
            min_score = min(keyframe_scores.values())

            for timestamp in keyframe_scores:
                keyframe_scores[timestamp] = (keyframe_scores[timestamp] - min_score) / (max_score - min_score)

        # Combine scores 

        scores = {timestamp: text_scores.get(timestamp, 0) + keyframe_scores.get(timestamp, 0) for timestamp in text_scores}

        all_scores = list(scores.values())

        # Compute threshold dynamically if not provided
        if threshold is None:
            threshold = np.percentile(all_scores, self.config.get("threshold_percentile", 80))
            print(f"Computed threshold: {threshold}")

        # Select sentences that meet the threshold
        filtered_sentences = [
            timestamp for timestamp, score in scores.items() if score >= threshold
        ]
        print(f"Filtered {len(filtered_sentences)} sentences out of {len(document.sentences)}")

        # Store filtered sentences in Document metadata
        document.add_metadata("filtered_sentences", filtered_sentences)
