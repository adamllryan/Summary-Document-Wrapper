from typing import Any, Callable, Dict, Optional, List


class Track:
    """
    Represents a track in a video.
    """

    def __init__(self, score: float = None) -> None:

        self.score = score

    def set_data(self, data: Any) -> None:
        pass

    def get_data(self) -> Any:
        pass

    def set_score(self, score: float) -> None:

        # assert score is not None, "Score cannot be None"
        # assert isinstance(score, float), "Score must be a float"

        self.score = score

    def get_score(self) -> Optional[float]:
        return self.score

    def __str__(self) -> str:
        pass


class TextTrack(Track):
    """
    Represents a text track in a video.
    """

    def __init__(self, data: Dict[str, str] = None) -> None:
        super().__init__(data.get("score", None) if data is not None else None)

        # assert "text" in data, "Text must be provided"
        # assert isinstance(data["text"], str), "Text must be a string"

        if data is None:
            self.text = ""
            self.speaker = "UNKNOWN"
            self.embeddings = {}
        else:
            self.text = data.get("text", "")
            self.speaker = data.get("speaker", "UNKNOWN")
            self.embeddings = data.get("embeddings", {})

    def set_text(self, text: str) -> None:
        self.text = text

    def get_text(self) -> str:
        return self.text

    def set_speaker(self, speaker: str) -> None:

        assert isinstance(speaker, str), "Speaker must be a string"

        self.speaker = set_speaker

    def get_speaker(self) -> str:
        return self.speaker

    def set_data(self, data: Dict[str, str]) -> None:

        assert "text" in data, "Text must be provided"
        assert isinstance(data["text"], str), "Text must be a string"

        self.text = data.get("text", "")
        self.speaker = data.get("speaker", "UNKNOWN")
        self.embeddings = data.get("embeddings", {})
        self.score = data.get("score", None)

    def get_data(self) -> Dict[str, str]:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "embeddings": self.embeddings,
            "score": self.score,
        }

    def get_formatted_text(self) -> str:
        return f"{self.speaker}: {self.text}"

    def __str__(self) -> str:
        return self.text

    def set_embeddings(self, embeddings: Dict[str, Any]) -> None:
        self.embeddings = embeddings

    def get_embeddings(self) -> Dict[str, Any]:
        return self.embeddings


class KeyframeTrack(Track):
    """
    Represents a keyframe track in a video.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(data.get("score", None) if data is not None else None)

        if data is not None:
            self.frames = data.get("frames", [])
        else:
            self.frames = []

    def set_frames(self, frames: List[str]) -> None:
        self.frames = frames

    def get_frames(self) -> List[str]:
        return self.frames

    def set_data(self, data: Dict[str, Any]) -> None:
        self.frames = data.get("frames", [])
        self.score = data.get("score", None)

    def get_data(self) -> Dict[str, Any]:
        return {"frames": self.frames, "score": self.score}

    def __str__(self) -> str:
        return f"{len(self.frames)} frames"


class TrackFactory:
    """
    Factory class for creating tracks.
    """

    track_types = {"text": TextTrack, "keyframe": KeyframeTrack}

    def __init__(self, new_track_types: Dict[str, Callable]) -> None:
        TrackFactory.track_types.update(new_track_types)

    @staticmethod
    def create_track(data: Any, track_type: str, **kwargs) -> Track:
        if track_type not in TrackFactory.track_types:
            raise ValueError(f"Track type {track_type} not found")
        return TrackFactory.track_types[track_type](data, **kwargs)

    @staticmethod
    def create_custom_track(data: Any, track_type: Callable, **kwargs) -> Track:
        return track_type(data, **kwargs)
