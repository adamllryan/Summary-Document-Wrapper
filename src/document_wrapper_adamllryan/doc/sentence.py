from typing import List, Optional, Dict
from collections import Counter
from .segment import Segment

class Sentence:
    """
    Represents a sentence, which consists of multiple segments and tracks.
    """
    def __init__(self, sentence_data: List[Dict]) -> None:
        self.segments: List[Segment] = [Segment(seg) for seg in sentence_data]
        self.start: float = self.segments[0].start
        self.end: float = self.segments[-1].end
        self.embeddings: Optional[Dict[str, List[float]]] = None
        self.text_score: float = 0.0
        self.keyframe_score: int = 0
        self.aggregated_score: float = 0.0
    
    def __str__(self) -> str:
        speaker = self.segments[0].get_track("transcript").get_data()["speaker"]  
        text_segments = [str(seg.get_track("transcript").get_data()["text"]) for seg in self.segments]
        return speaker + ": " + " ".join(text_segments)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def contains(self, ts: float) -> bool:
        """Check if the timestamp is within this sentence's range."""
        return self.start <= ts <= self.end
    
    def find_segment(self, ts: float) -> Optional[Segment]:
        """Find the segment containing a given timestamp."""
        return next((seg for seg in self.segments if seg.contains(ts)), None)
    
    def get_formatted_text(self) -> str:
        """Retrieve the sentence formatted with the most frequent speaker label."""
        speakers = [seg.get_track("transcript")["speaker"] for seg in self.segments if seg.get_track("transcript")] 
        most_frequent_speaker = Counter(speakers).most_common(1)[0][0] if speakers else "UNKNOWN"
        return f"{most_frequent_speaker}: {self}"

    def get_plain_text(self) -> str:
        """Retrieve the plain text of the sentence."""
        return " ".join(seg.get_track("transcript")["text"] for seg in self.segments)

    def get_segments_plain_text(self) -> List[str]:
        """Retrieve the plain text of each segment."""
        return [
            seg.get_track("transcript")["text"] for seg in self.segments
        ]
