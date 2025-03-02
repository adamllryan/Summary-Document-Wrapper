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

        # # Compute threshold dynamically if not provided
        # if threshold is None:
        #     threshold = np.percentile(
        #         all_scores, self.config.get("threshold_percentile", 80)
        #     )
        #     print(f"Computed threshold: {threshold}")

        # Dynamic thresholding

        Q1, Q3 = np.percentile(all_scores, [25, 75])
        iqr_value = Q3 - Q1
        std_dev = np.std(all_scores)
        mean_score = np.mean(all_scores)

        # Lower bound filtering
        lower_cutoff = max(Q1 - 1.5 * iqr_value, mean_score - 1.5 * std_dev)
        lower_cutoff = max(
            lower_cutoff,
            np.percentile(all_scores, self.config.get("min_percentile", 5)),
        )

        # Upper bound: Keep the top 10% untouched
        upper_cutoff = np.percentile(
            all_scores, self.config.get("keep_top_percentile", 90)
        )

        # Ensure at least 30% of sentences remain
        min_content_kept = max(
            len(document.sentences) * 0.3,
            len(document.sentences) - len(all_scores) * 0.15,
        )

        print(
            f"Computed lower threshold: {lower_cutoff}, upper threshold: {upper_cutoff}"
        )

        # # Select sentences that meet the threshold
        # filtered_sentences = [
        #     timestamp for timestamp, score in scores.items() if score >= threshold
        # ]

        # Filter out sentences below lower threshold, but always keep the top 10%
        filtered_sentences = [
            ts
            for ts, score in scores.items()
            if score >= lower_cutoff or score >= upper_cutoff
        ]

        # Ensure we are not removing too much content
        while len(filtered_sentences) < min_content_kept:
            lower_cutoff *= 0.9  # Loosen the threshold
            filtered_sentences = [
                ts
                for ts, score in scores.items()
                if score >= lower_cutoff or score >= upper_cutoff
            ]

        print(
            f"Filtered {len(document.sentences) - len(filtered_sentences)} sentences out of {len(document.sentences)}."
        )

        # print(
        #     f"Filtered {len(filtered_sentences)} sentences out of {len(document.sentences)}"
        # )

        # Store filtered sentences in Document metadata
        if document.get_metadata("filtered_sentences"):
            document.set_metadata("filtered_sentences", filtered_sentences)
        else:
            document.set_metadata("filtered_sentences", filtered_sentences)

        # update document sentence scores

        document.set_scores(scores.values())
