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
            tuple(sentence.timestamp): sentence.call_track_method("get_score", "text")[
                "text"
            ]
            for sentence in document.sentences
            if sentence.call_track_method("get_score", "text") is not None
        }

        # Extract scores from keyframe track
        keyframe_scores = {
            tuple(sentence.timestamp): sentence.call_track_method(
                "get_score", "keyframe"
            )["keyframe"]
            for sentence in document.sentences
            if sentence.call_track_method("get_score", "keyframe") is not None
        }

        # Normalize keyframe scores to [0, 1]
        if keyframe_scores:
            max_score = max(keyframe_scores.values())
            min_score = min(keyframe_scores.values())

            if min_score == max_score:
                keyframe_scores = {
                    timestamp: (1 if max_score == 0 else 0)
                    for timestamp in keyframe_scores
                }
            else:
                for timestamp in keyframe_scores:
                    keyframe_scores[timestamp] = (
                        keyframe_scores[timestamp] - min_score
                    ) / (max_score - min_score)

        # Combine scores

        alpha = self.config.get("alpha", 0.5)

        scores = {
            timestamp: alpha * text_scores.get(timestamp, 0)
            + (1 - alpha) * keyframe_scores.get(timestamp, 0)
            for timestamp in text_scores
        }

        all_scores = list(scores.values())

        if len(all_scores) == 0:
            print("No valid scores found. Skipping filtering.")
            return

        # Compute mean and standard deviation
        mean_score = np.mean(all_scores)
        std_dev = np.std(all_scores)

        # Define the lower cutoff using standard deviation
        std_factor = self.config.get(
            "std_factor", 1
        )  # Default to 1.5 std dev below mean

        lower_cutoff = mean_score - std_factor * std_dev

        # Ensure the top 15% of the best scores remain untouched
        upper_cutoff = np.percentile(
            all_scores, self.config.get("keep_top_percentile", 85)
        )

        print(
            f"Computed mean: {mean_score:.4f}, std_dev: {std_dev:.4f}, lower threshold: {lower_cutoff:.4f}"
        )

        # Filter out sentences below the threshold, but keep the top content
        filtered_sentences = [
            ts
            for ts, score in scores.items()
            if score >= lower_cutoff or score >= upper_cutoff
        ]

        print(
            f"Filtered {len(document.sentences) - len(filtered_sentences)} sentences out of {len(document.sentences)}."
        )

        # Store filtered sentences in Document metadata
        if "filtered_sentences" in document.metadata:
            document.set_metadata("filtered_sentences", filtered_sentences)
        else:
            document.add_metadata("filtered_sentences", filtered_sentences)

        # Update document sentence scores
        document.set_scores(scores.values())
