import numpy as np
import rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Evaluator:
    """
    A class for evaluating summarization performance using ROUGE and cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator with a SentenceTransformer model for text embeddings.
        
        :param model_name: The name of the Hugging Face sentence transformer model.
        """
        self.rouge = rouge.Rouge()
        self.embedding_model = SentenceTransformer(model_name)

    def compute_rouge(self, reference: str, hypothesis: str) -> dict:
        """
        Compute ROUGE scores between the reference and generated text.
        
        :param reference: The ground truth summary.
        :param hypothesis: The generated summary.
        :return: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        scores = self.rouge.get_scores(hypothesis, reference)[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }

    def compute_cosine_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Compute cosine similarity between the reference and generated summary embeddings.
        
        :param reference: The ground truth summary.
        :param hypothesis: The generated summary.
        :return: Cosine similarity score between the two embeddings.
        """
        ref_embedding = self.embedding_model.encode([reference])
        hyp_embedding = self.embedding_model.encode([hypothesis])
        return float(cosine_similarity(ref_embedding, hyp_embedding)[0][0])

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        """
        Evaluate the summary using ROUGE and cosine similarity.
        
        :param reference: The ground truth summary.
        :param hypothesis: The generated summary.
        :return: A dictionary with ROUGE scores and cosine similarity.
        """
        rouge_scores = self.compute_rouge(reference, hypothesis)
        cosine_sim = self.compute_cosine_similarity(reference, hypothesis)
        return {**rouge_scores, "cosine_similarity": cosine_sim}
