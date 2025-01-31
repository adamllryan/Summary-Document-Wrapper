import unittest
from document_wrapper_adamllryan.doc.track import Track

class TestTrack(unittest.TestCase):
    def setUp(self):
        """Initialize sample track instances before each test."""
        self.transcript_data = {
            "text": "Hello, world!",
            "formatted_text": "<b>Hello</b>, world!",
            "speaker": "Alice"
        }
        self.transcript_formatter = lambda d: f"{d.get('speaker', 'UNKNOWN')}: {d.get('text', '')}"
        self.transcript_track = Track(self.transcript_data, self.transcript_formatter)

        self.video_data = {"frames": [0.1, 0.2, 0.3, 0.4]}
        self.video_formatter = lambda d: f"Frames: {len(d.get('frames', []))}"
        self.video_track = Track(self.video_data, self.video_formatter)

        self.audio_data = {"waveform": [0.01, 0.02, 0.03]}
        self.audio_formatter = lambda d: f"Waveform Length: {len(d.get('waveform', []))}"
        self.audio_track = Track(self.audio_data, self.audio_formatter)

    def test_string_representation(self):
        """Test the string representation of different tracks."""
        self.assertEqual(str(self.transcript_track), "Alice: Hello, world!")
        self.assertEqual(str(self.video_track), "Frames: 4")
        self.assertEqual(str(self.audio_track), "Waveform Length: 3")

    def test_data_access(self):
        """Test that data can be accessed correctly from the track."""
        self.assertEqual(self.transcript_track["text"], "Hello, world!")
        self.assertEqual(self.video_track["frames"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(self.audio_track["waveform"], [0.01, 0.02, 0.03])

    def test_get_data(self):
        """Test retrieval of raw track data."""
        self.assertEqual(self.transcript_track.get_data(), self.transcript_data)
        self.assertEqual(self.video_track.get_data(), self.video_data)
        self.assertEqual(self.audio_track.get_data(), self.audio_data)

    def test_update_data(self):
        """Test modifying the track data."""
        new_data = {"text": "Goodbye, world!", "speaker": "Bob"}
        self.transcript_track.set_data(new_data)
        self.assertEqual(self.transcript_track["text"], "Goodbye, world!")
        self.assertEqual(self.transcript_track["speaker"], "Bob")

    def test_update_formatter(self):
        """Test modifying the track's formatter function."""
        new_formatter = lambda d: f"Speaker: {d.get('speaker', 'UNKNOWN')}, Message: {d.get('text', '')}"
        self.transcript_track.set_formatter(new_formatter)
        self.assertEqual(str(self.transcript_track), "Speaker: Alice, Message: Hello, world!")

    def test_default_formatter(self):
        """Test default string representation when no formatter is provided."""
        default_track = Track({"data": "sample"})
        self.assertEqual(str(default_track), "{'data': 'sample'}")

if __name__ == "__main__":
    unittest.main()
