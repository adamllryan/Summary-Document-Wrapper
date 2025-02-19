
from typing import List
import numpy as np 
from document_wrapper_adamllryan.doc.sentence import Sentence 


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
        scores = {
            sentence: sentence.call_track_method("get_score", "text")
            for sentence in document.sentences
            if sentence.call_track_method("get_score", "text") is not None
        }

        all_scores = list(scores.values())

        # Compute threshold dynamically if not provided
        if threshold is None:
            threshold = np.percentile(all_scores, self.config.get("threshold_percentile", 80))
            print(f"Computed threshold: {threshold}")

        # Select sentences that meet the threshold
        filtered_sentences = [s for s, score in scores.items() if score >= threshold]
        print(f"Filtered {len(filtered_sentences)} sentences out of {len(document.sentences)}")

        # Store filtered sentences in Document metadata
        document.add_metadata("filtered_sentences", [(s.start, s.end) for s in filtered_sentences])

        return filtered_sentences
