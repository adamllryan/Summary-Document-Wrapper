from typing import List, Optional, Dict, Any, Callable
from .sentence import Sentence

class Document:
    """
    Represents a document consisting of multiple sentences.
    """
    def __init__(self, sentences: List[List[dict]], tracks: Optional[Dict[str, Callable]] = None):
        self.sentences: List[Sentence] = [Sentence(s, tracks) for s in sentences]
        
         
    def __str__(self) -> str:
        return "\n".join(f"({s.start}:{s.end}) - {s}" for s in self.sentences)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_plain_text(self) -> str:
        """Retrieve the raw text from the document."""
        return "\n".join(str(s) for s in self.sentences)
    
    def call_track_method(self, method_name: str, track_type: Optional[str] = None, data: Optional[List[Any]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Dynamically calls a method on tracks if it exists, with optional data arguments.

        Args:
            method_name: The name of the method to call.
            track_type: The specific track type to call the method on (optional).
            data: A list of arguments to pass, one per sentence (e.g., embeddings).
            **kwargs: Additional keyword arguments to pass.

        Returns:
            A list of results, one per sentence.
        """

        # Ensure data is either None or a list of the same length as sentences
        if data is not None:
            assert isinstance(data, list), "Data must be a list"
            assert len(data) == len(self.sentences), "Data length must match the number of sentences"

        results = []
        for i, sentence in enumerate(self.sentences):
            # Pass data[i] if available, otherwise pass no extra arguments
            result = sentence.call_track_method(method_name, track_type, *(data[i],) if data else (), **kwargs)
            results.append(result)
    
    def find_sentence(self, ts: float) -> Optional[Sentence]:
        """Find the sentence containing a given timestamp."""
        return next((s for s in self.sentences if s.contains(ts)), None)
    
    def find_segment(self, ts: float) -> Optional["Segment"]:
        """Find the segment containing a given timestamp."""
        for sentence in self.sentences:
            seg = sentence.find_segment(ts)
            if seg:
                return seg
        return None
    
    def export(self) -> List[dict]:
        """Export the document to a list of dictionaries."""
        return [s.export() for s in self.sentences]
