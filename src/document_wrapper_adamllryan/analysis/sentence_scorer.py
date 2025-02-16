from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
from document_wrapper_adamllryan.doc.document import Document
from document_wrapper_adamllryan.doc.sentence import Sentence 


class SentenceScorer:
    """
    Scores sentences in a transcript based on similarity to the summary using embeddings.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.model = SentenceTransformer(self.config["embedding_model"])

    def score(self, sentences: List[Sentence], summary: str) -> Dict[str, float]:
        """ Produce embeddings for the summary """
        summary_embeddings = self.model.encode([summary])

        print("Computing sentence scores")
        scores = {}
        for sentence in sentences:
            max_speaker = sentence.get_formatted_text().split(":")[0]
            scores[sentence.get_formatted_text()] = self._get_max_similarity(sentence, summary_embeddings, max_speaker)

        return scores

    def _get_max_similarity(self, sentence: Sentence, embeddings: List[np.ndarray], speaker: Optional[str] = None) -> float:
        """Finds the highest cosine similarity between a sentence and summary embeddings."""

        print("Scoring sentences")
        # print(sentence.embeddings)
        text_embedding = sentence.embeddings['text']

        max_cosine_similarity = 0.0

        if speaker is not None and speaker != sentence.get_formatted_text().split(":")[0]:
            return max_cosine_similarity

        for summary_embedding in embeddings:
            cosine_similarity = util.cos_sim(text_embedding, summary_embedding).item()
            if cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = cosine_similarity

        print(f"Max cosine similarity: {max_cosine_similarity}")

        return max_cosine_similarity

    def assign_sentence_embeddings(self, document: Document):
        """Assigns computed sentence embeddings to the Document object."""
        embeddings = self.model.encode([str(s) for s in document.sentences])
        document.assign_embeddings(embeddings)

if __name__ == "__main__":
    # Example usage
    config = {
        "embedding_model": "stsb-roberta-large"
    }
    scorer = SentenceScorer(config)
