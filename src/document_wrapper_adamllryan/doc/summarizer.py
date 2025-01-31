from typing import List, Dict, Any
from transformers import pipeline

class Summarizer:
    """
    Handles summarization of transcriptions and extracted keyframes using Hugging Face pipelines.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with a Hugging Face model.
        :param model_name: Name of the summarization model
        """
        self.summarization_pipeline = pipeline("summarization", model=model_name)
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize a given text.
        :param text: Input text to summarize
        :param max_length: Maximum length of summary
        :param min_length: Minimum length of summary
        :return: Summarized text
        """
        summary = self.summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    
    def summarize_segments(self, segments: List[Dict[str, Any]]) -> str:
        """
        Summarize a list of transcription segments.
        :param segments: List of transcription segments (each with 'text' field)
        :return: Summarized text from the segments
        """
        combined_text = " ".join(segment['text'] for segment in segments)
        return self.summarize_text(combined_text)
    
if __name__ == "__main__":
    summarizer = Summarizer()
    
    sample_text = """
    Artificial Intelligence (AI) is a field of computer science that aims to create intelligent machines
    capable of mimicking human behavior. AI applications range from simple rule-based systems to complex
    deep learning models that can recognize speech, translate languages, and even drive cars autonomously.
    """
    
    print("Summarized Text:", summarizer.summarize_text(sample_text))
