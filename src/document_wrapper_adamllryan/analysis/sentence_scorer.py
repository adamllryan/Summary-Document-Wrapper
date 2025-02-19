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
    
    def score(self, document: Document):
        """Computes similarity scores and assigns embeddings for each sentence in the document."""
        print("Computing sentence scores")
        
        summary = document.metadata.get("summary", "")
        summary_embedding = self.model.encode([summary])  # Encode summary once
        
        scores = []
        embeddings = []

        plaintext_sentences = [
            sentence.call_track_method("get_formatted_text", "text")
            for sentence in document.sentences
        ]

        # print("Sentences", plaintext_sentences)

        embeddings = self.model.encode(plaintext_sentences).tolist()
        scores = util.cos_sim(summary_embedding, embeddings).tolist()[0]
        # print(f"Scores: {scores}")
        # print(f"Embeddings: {embeddings}")
        
        
        
        document.call_track_method("set_score", "text", scores)
        document.call_track_method("set_embeddings", "text", embeddings)
