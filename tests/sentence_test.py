import unittest
from document_wrapper_adamllryan.doc.sentence import Sentence

class TestSentence(unittest.TestCase):
    def setUp(self):
        """Initialize sample sentence instances before each test."""
        self.sentence_data = [
            {
                "start": 0.0,
                "end": 2.0,
                "timestamp": (0.0, 2.0),
                "text": "Hello, this is a test.",
                "formatted_text": "<b>Hello</b>, this is a test.",
                "speaker": "Alice",
                "frames": [0.1, 0.2]
            },
            {
                "start": 2.5,
                "end": 5.0,
                "timestamp": (2.5, 5.0),
                "text": "This is another sentence.",
                "formatted_text": "<i>This is another sentence.</i>",
                "speaker": "Bob",
                "frames": [0.3, 0.4]
            }
        ]
        self.sentence = Sentence(self.sentence_data)

    def test_sentence_initialization(self):
        """Test that a sentence initializes correctly with given segment data."""
        self.assertEqual(len(self.sentence.segments), 2)
        self.assertEqual(self.sentence.start, 0.0)
        self.assertEqual(self.sentence.end, 5.0)

    def test_sentence_string_representation(self):
        """Test the string representation of the sentence."""
        self.assertEqual(str(self.sentence), "Alice: Hello, this is a test. Bob: This is another sentence.")

    def test_sentence_contains_timestamp(self):
        """Test whether a sentence contains a given timestamp."""
        self.assertTrue(self.sentence.contains(1.0))  # Inside first segment
        self.assertTrue(self.sentence.contains(3.0))  # Inside second segment
        self.assertFalse(self.sentence.contains(6.0))  # Outside of range

    def test_find_segment(self):
        """Test finding a segment by timestamp."""
        segment1 = self.sentence.find_segment(1.0)
        segment2 = self.sentence.find_segment(3.0)
        segment3 = self.sentence.find_segment(6.0)  # Should return None

        self.assertIsNotNone(segment1)
        self.assertEqual(segment1.start, 0.0)
        self.assertEqual(segment1.end, 2.0)

        self.assertIsNotNone(segment2)
        self.assertEqual(segment2.start, 2.5)
        self.assertEqual(segment2.end, 5.0)

        self.assertIsNone(segment3)  # Out of range

    def test_get_formatted_text(self):
        """Test the formatted text output with the most frequent speaker label."""
        formatted_text = self.sentence.get_formatted_text()
        # The most frequent speaker is Alice (1st segment) and Bob (2nd segment).
        # Since there's a tie, it will pick the first speaker that appears.
        self.assertIn("Alice:", formatted_text) or self.assertIn("Bob:", formatted_text)

if __name__ == "__main__":
    unittest.main()
