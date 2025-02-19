from typing import Dict, Any, Optional, Callable
from .track import Track, TrackFactory

class Sentence:
    """
    Represents a  whole sentence. 
    """
    def __init__(self, data: Dict[str, Any], track_types: Dict[str, Callable] = None) -> None:
        
        assert "start" in data, "Start time must be provided"
        assert "end" in data, "End time must be provided"
        assert data["start"] <= data["end"], "Start time must be less than or equal to end time"

        self.start: float = data.get("start", 0)
        self.end: float = data.get("end", 0)
        self.timestamp: tuple[float, float] = data.get("timestamp", (self.start, self.end))

        self.score: float = data.get("score", 0.0)

        self.primary_track = data.get("primary_track", "text")

        self.tracks: Dict[str, Track] = {}

        if track_types is None:
            for track_type, track_data in TrackFactory.track_types.items():
                raise ValueError(f"Track type {track_type} not found in track_types")
                self.tracks[track_type] = TrackFactory.create_track(data.get(track_type, {}), track_type)
        else:
            for track_type, track_data in track_types.items():
                self.tracks[track_type] = TrackFactory.create_custom_track(data.get(track_type, {}), track_data)



    def call_track_method(self, method_name: str, track_type: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Dynamically calls a method on tracks if it exists.
        
        Args:
            method_name: The name of the method to call.
            track_type: The specific track type to call the method on (optional).
        
        Returns:
            A dictionary of results with track type as the key and the method return value.
        """
        results = {}

        if track_type:
            track = self.get_track(track_type)
            if track and hasattr(track, method_name):
                results[track_type] = getattr(track, method_name)(*args, **kwargs)
        else:
            for t_type, track in self.tracks.items():
                if hasattr(track, method_name):
                    results[t_type] = getattr(track, method_name)(*args, **kwargs)

        return results

    
    def __str__(self) -> str:
        if self.primary_track in self.tracks:
            return str(self.tracks[self.primary_track])
        return f"Segment({self.start} - {self.end})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def contains(self, ts: float) -> bool:
        """Check if the given timestamp falls within this segment's time range."""
        return self.start <= ts <= self.end
    
    def get_track(self, track_name: str) -> Optional[Track]:
        """Retrieve a track by its name."""
        return self.tracks.get(track_name)
    
    def add_track(self, track_name: str, data: Dict[str, Any], formatter: Optional[Callable[[Dict[str, Any]], str]] = None) -> None:
        """Dynamically add a new track to the segment."""
        self.tracks[track_name] = TrackFactory.create_track(track_name, data, formatter)
    
    def remove_track(self, track_name: str) -> None:
        """Remove a track from the segment."""
        if track_name in self.tracks:
            del self.tracks[track_name]

    def get_data(self) -> Dict[str, Any]:
        """Retrieve the raw data stored in the segment."""
        return {
            "start": self.start,
            "end": self.end,
            "timestamp": self.timestamp,
            "score": self.score,
            **{name: track.get_data() for name, track in self.tracks.items()}
        }

    def set_score(self, score: float) -> None:
        """Set the aggregate score for the segment."""
        self.score = score

    def get_score(self) -> float:
        """Retrieve the aggregate score for the segment."""
        return self.score

    def export(self) -> Dict[str, Any]:
        """Export the segment to a dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "timestamp": self.timestamp,
            "score": self.score,
            **{name: track.get_data() for name, track in self.tracks.items()}
        }
