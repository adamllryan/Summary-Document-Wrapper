# import torch
# import os
# import json
# import subprocess
# import torch
# import numpy as np
# import time
# import cv2
# import random
# from datetime import timedelta
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import sys
# from sklearn.cluster import KMeans

# from document_wrapper_adamllryan.doc.document import Document
# from document_wrapper_adamllryan.doc.sentence import Sentence
# from document_wrapper_adamllryan.analysis.transcriber import Transcriber
# from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis
# from document_wrapper_adamllryan.util.downloader import VideoDownloader

from typing import List, Dict, Optional, Tuple
class Summarizer:
    """
    Summarizes transcripts using a transformer-based model.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])

        self.summarizer = pipeline(
            "summarization",
            model=self.config["model"],
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            # device=-1,
            max_length=self.config["max_len"],
            min_length=self.config["min_len"],
            do_sample=self.config["do_sample"]
        )
        self.token_counter = AutoTokenizer.from_pretrained(self.config["model"])
        self.token_limit = self.config["token_limit"]

    def summarize(self, text: str) -> str:
        """Summarizes input text by breaking it into chunks that fit within the token limit."""

        print("Summarizing")

        """ Break into sentences """
        chunks = text.split("\n")
        
        summary = ""
        i = 0

        chunks = [""]
        for chunk in text.split("\n"):
          if len(self.token_counter.tokenize(chunks[-1])) + len(self.token_counter.tokenize(chunk)) < self.config["token_limit"]:
            chunks[-1] += chunk + "\n"
          else:
            chunks.append(chunk + "\n")


        for chunk in chunks:
            summary += self._generate_summary(chunk)


        return summary

    def _generate_summary(self, text: str) -> str:
        """Generates a summary for a given text chunk."""
        assert len(self.token_counter.tokenize(text)) <= self.config["token_limit"]
        return self.summarizer([text], max_length=256, min_length=30, do_sample=False)[0]['summary_text']

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""

        return len(self.token_counter.tokenize(text))
