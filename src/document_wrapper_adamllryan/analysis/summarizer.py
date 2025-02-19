from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import sys
import torch

from typing import List, Dict, Optional, Tuple
class Summarizer:
    """
    Summarizes transcripts using a transformer-based model.
    """
    def __init__(self, config: Dict[str, str]):

        assert "model" in config, "Model not found in config"
        assert "max_len" in config, "Max length not found in config"
        assert "min_len" in config, "Min length not found in config"
        assert "do_sample" in config, "Do sample not found in config"
        assert "token_limit" in config, "Token limit not found in config"

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
        # print(f"Text to summarize: {text}\n----")

        """ Break into sentences """

        chunks = [""]

        for sentence in text.split("\n"):
          tokens = self._count_tokens(sentence)
          current_chunk_len = self._count_tokens(chunks[-1])
          # Need to add this because whisper has been making lower quality
          # content over time, less properly formatted
          if tokens > self.config["token_limit"]:
            # If a sentence is larger than token limit we break down further
            remaining_size = self.config["token_limit"] - current_chunk_len - 1
            subsentences = self._split_large_sentences(sentence, self.config["token_limit"] - 1, remaining_size)
            for subsentence in subsentences:
              sub_length = self._count_tokens(subsentence)
              current_chunk_len = self._count_tokens(chunks[-1])
              if sub_length + current_chunk_len < self.config["token_limit"] - 1:
                chunks[-1]+=subsentence + "\n"
              else:
                chunks.append(subsentence + "\n")
          elif tokens + current_chunk_len < self.config["token_limit"]:
            # Extend the sentence if we are still under the len of token limit
            chunks[-1] += sentence + "\n"
            # print(f"Extending chunk: {chunks[-1]}")
          else:
            # Create a new chunk
            chunks.append(sentence + "\n")

        # print(f"---\nChunks: {chunks}\n---")

        summary = self._generate_summary(chunks)
        # print(f"Summary: {summary}")

        return summary

    def _generate_summary(self, text: list[str] | str) -> str:
        """Generates a summary for a given text chunk."""
        if isinstance(text, str):
          text = [text]
        for chunk in text:
          # print(f"Tokenizing chunk: {chunk}")
          length = self._count_tokens(chunk)
          assert length <= self.config["token_limit"], f"Length was {length}"
        return self.summarizer(text, max_length=self.config["max_len"], min_length=self.config["min_len"], do_sample=False)[0]['summary_text']

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""

        return len(self.token_counter.tokenize(text))


    def _split_large_sentences(self, sentence: str, max_tokens: int, remaining_size: int) -> List[str]:
        """Splits a sentence into smaller chunks that fit within the token limit."""
        words = [word.strip() for word in sentence.split(" ")]
        print(f"words: {words}\n---")
        current_token_count = 0
        subsentences = [""]
        # Grab the remaining words that fit in previous chunk



        for word in words:
            word_token_len = self._count_tokens(word + " ")

            # If we want to fill remaining size first
            if len(subsentences) == 1:
              if current_token_count + word_token_len < remaining_size:
                subsentences[0] += word + " "
                current_token_count += word_token_len
              else:
                subsentences.append(word + " ")
                current_token_count = word_token_len
                # Continue splitting normally
            else:
              if current_token_count + word_token_len < max_tokens:
                subsentences[-1] += word + " "
                current_token_count += word_token_len
              else:
                subsentences.append(word + " ")
                current_token_count = word_token_len
        print(f"Subsentences: {subsentences}\n---")
        return subsentences
