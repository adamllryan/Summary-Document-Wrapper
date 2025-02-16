
from typing import List
import numpy as np 
from document_wrapper_adamllryan.doc.sentence import Sentence 


class Filter:
    """
    Filters sentences based on a dynamically computed threshold from the score distribution.
    """
    def __init__(self, config: dict):
        self.config = config

    def apply(self, sentences: List[Sentence], scores: dict, threshold: float = None) -> List[Sentence]:
        """Filters sentences based on a dynamic threshold derived from score distribution."""

        print("Filtering sentences")

        all_scores = list(scores.values())

        if threshold is None:
            threshold = np.percentile(all_scores, self.config.get("threshold_percentile", 80))
            print(f"Computed threshold: {threshold}")

        filtered_sentences = [s for s in sentences if scores[s.get_formatted_text()] >= threshold]
        print(f"Filtered {len(filtered_sentences)} sentences out of {len(sentences)}")

        print(f"Filtered sentences: {filtered_sentences}")

        return filtered_sentences
