from typing import Any, Callable, Dict, Optional

class Track:
    """
    A generic track that stores multiple attributes and applies a custom function for string representation.
    """
    def __init__(self, data: Dict[str, Any], formatter: Optional[Callable[[Dict[str, Any]], str]] = None):
        """
        :param data: Dictionary containing attributes for the track.
        :param formatter: A function that determines how the track is converted to a string.
        """
        self.data = data
        self.formatter = formatter if formatter else lambda d: str(d)

    def __str__(self) -> str:
        """Returns the string representation of the track based on the formatter function."""
        return self.formatter(self.data)

    def __repr__(self) -> str:
        return f"Track({self.data})"

    def __getitem__(self, key: str) -> Any:
        """Retrieve a specific attribute from the track."""
        return self.data.get(key)

    def get_data(self) -> Dict[str, Any]:
        """Retrieve the raw data stored in the track."""
        return self.data 

    def get_formatter(self) -> Optional[Callable[[Dict[str, Any]], str]]:
        """Retrieve the formatter function used for string representation."""
        return self.formatter 

    def set_formatter(self, formatter: Callable[[Dict[str, Any]], str]) -> None:
        """Set the formatter function for string representation."""
        self.formatter = formatter 

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the data attributes for the track."""
        self.data = data
