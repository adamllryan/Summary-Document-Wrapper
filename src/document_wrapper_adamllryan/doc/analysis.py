from typing import List
from .document import Document

class DocumentAnalysis:
    """
    Contains static methods for analyzing and processing document data.
    """
    @staticmethod
    def assign_scores(document: Document, text_scores: List[float], keyframe_scores: List[int]) -> None:
        """Assign scores to sentences within a Document and compute aggregated scores."""
        assert len(text_scores) == len(document.sentences)
        assert len(keyframe_scores) == len(document.sentences)

        for i, sentence in enumerate(document.sentences):
            sentence.text_score = text_scores[i]
            sentence.keyframe_score = keyframe_scores[i]

        # Compute normalized scores
        text_min, text_max = min(text_scores), max(text_scores)
        keyframe_min, keyframe_max = min(keyframe_scores), max(keyframe_scores)

        for sentence in document.sentences:
            if text_max == text_min and keyframe_max == keyframe_min:
                sentence.aggregated_score = 0
            elif text_max == text_min:
                sentence.aggregated_score = sentence.keyframe_score
            elif keyframe_max == keyframe_min:
                sentence.aggregated_score = sentence.text_score
            else:
                sentence.aggregated_score = 0.5 * (sentence.text_score - text_min) / (text_max - text_min) + \
                                            0.5 * (sentence.keyframe_score - keyframe_min) / (keyframe_max - keyframe_min)
    
    @staticmethod
    def raw_to_sentences(document: Document) -> List[List[dict]]:
        """Convert raw transcript data within a Document into segmented sentences."""
        sentences, sentence = [], []
        for s in document.sentences:
            sentence_data = [{
                "tracks": seg.get_data(),
                "timestamp": seg.timestamp,
                "start": seg.start,
                "end": seg.end
            } for seg in s.segments]
            sentence.append(sentence_data)
            # Trim spaces and check sentence end 
            sn = s.get_plain_text().strip()
            if sn.endswith(".") or sn.endswith("?") or sn.endswith("!"):
                sentences.append(sentence)
                sentence = []
        if sentence:
            sentences.append(sentence)
        return sentences

    @staticmethod
    def list_to_document(transcript_data: List[dict]) -> Document:
        """Convert a list of transcript data into a Document object."""
        sentences, current_sentence = [], []

        for entry in transcript_data:
            current_sentence.append(entry)
            sn = entry["text"].strip()
            if sn.endswith(".") or sn.endswith("?") or sn.endswith("!") or sn[0].isupper():
                sentences.append(current_sentence)
                current_sentence = []

        if current_sentence:
            sentences.append(current_sentence)

        return Document(sentences)
