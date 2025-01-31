import unittest
from document_wrapper_adamllryan.doc.segment import Segment

class TestSegment(unittest.TestCase):
    def setUp(self):
        """Initialize sample segment instances before each test."""
        self.segment_data = {
            "start": 0.0,
            "end": 5.0,
            "timestamp": (0.0, 5.0),
            "text": "Hello, this is a test.",
            "formatted_text": "<b>Hello</b>, this is a test.",
            "speaker": "Alice",
            "frames": [0.1, 0.2, 0.3]
        }
        self.segment = Segment(self.segment_data)

    def test_segment_initialization(self):
        """Test that segment initializes correctly with given data."""
        self.assertEqual(self.segment.start, 0.0)
        self.assertEqual(self.segment.end, 5.0)
        self.assertEqual(self.segment.timestamp, (0.0, 5.0))

    def test_track_retrieval(self):
        """Test retrieval of existing tracks."""
        transcript_track = self.segment.get_track("transcript")
        self.assertIsNotNone(transcript_track)
        self.assertEqual(transcript_track["text"], "Hello, this is a test.")
        self.assertEqual(transcript_track["speaker"], "Alice")

        video_track = self.segment.get_track("video")
        self.assertIsNotNone(video_track)
        self.assertEqual(video_track["frames"], [0.1, 0.2, 0.3])

    def test_segment_contains_timestamp(self):
        """Test whether a segment contains a given timestamp."""
        self.assertTrue(self.segment.contains(2.5))
        self.assertTrue(self.segment.contains(0.0))
        self.assertTrue(self.segment.contains(5.0))
        self.assertFalse(self.segment.contains(6.0))

    def test_dynamic_track_addition(self):
        """Test dynamically adding a new track to a segment."""
        audio_data = {"waveform": [0.01, 0.02, 0.03]}
        audio_formatter = lambda d: f"Waveform Length: {len(d.get('waveform', []))}"

        self.segment.add_track("audio", audio_data, audio_formatter)
        audio_track = self.segment.get_track("audio")

        self.assertIsNotNone(audio_track)
        self.assertEqual(audio_track["waveform"], [0.01, 0.02, 0.03])
        self.assertEqual(str(audio_track), "Waveform Length: 3")

    def test_remove_track(self):
        """Test removing a track from a segment."""
        self.segment.remove_track("video")
        self.assertIsNone(self.segment.get_track("video"))

    def test_segment_string_representation(self):
        """Test the string representation of the segment."""
        self.assertEqual(str(self.segment), "Alice: Hello, this is a test.")

    def test_segment_repr(self):
        """Test the representation of the segment."""
        self.assertEqual(repr(self.segment), f"Segment(0.0 - 5.0)")

if __name__ == "__main__":
    unittest.main()
