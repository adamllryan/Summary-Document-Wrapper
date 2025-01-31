from typing import Dict, Any, Optional, Callable
from .track import Track

class Segment:
    """
    Represents a segment within a sentence, containing multiple tracks (e.g., transcript, video, audio).
    """
    def __init__(self, data: Dict[str, Any]) -> None:
        self.start: float = data["start"]
        self.end: float = data["end"]
        self.timestamp: tuple[float, float] = data["timestamp"]
        
        # Define different tracks with formatters
        transcript_formatter = lambda d: f"{d.get('speaker', 'UNKNOWN')}: {d.get('text', '')}"
        video_formatter = lambda d: f"Frames: {len(d.get('frames', []))}" if "frames" in d else "No frames"
        
        # Initialize tracks (THESE ARE THE DEFAULT TRACKS, SUBECT TO CHANGE)
        self.tracks: Dict[str, Track] = {
            "transcript": Track(data={
                "text": data.get("text", ""),
                "formatted_text": data.get("formatted_text"),
                "speaker": data.get("speaker", "UNKNOWN")
            }, formatter=transcript_formatter),
            "video": Track(data={
                "frames": data.get("frames", [])
            }, formatter=video_formatter)
        }
    
    def __str__(self) -> str:
        return str(self.tracks["transcript"])
    
    def __repr__(self) -> str:
        return f"Segment({self.start} - {self.end})"
    
    def contains(self, ts: float) -> bool:
        """Check if the given timestamp falls within this segment's time range."""
        return self.start <= ts <= self.end
    
    def get_track(self, track_name: str) -> Optional[Track]:
        """Retrieve a track by its name."""
        return self.tracks.get(track_name)
    
    def add_track(self, track_name: str, data: Dict[str, Any], formatter: Optional[Callable[[Dict[str, Any]], str]] = None) -> None:
        """Dynamically add a new track to the segment."""
        self.tracks[track_name] = Track(data, formatter)
    
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
            **{name: track.get_data() for name, track in self.tracks.items()}
        }
