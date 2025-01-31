import unittest
from document_wrapper_adamllryan.doc.document import Document

class TestDocument(unittest.TestCase):
    def setUp(self):
        """Initialize a sample document before each test."""
        self.document_data = [
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "timestamp": (0.0, 2.0),
                    "text": "Hello, this is a test.",
                    "formatted_text": "<b>Hello</b>, this is a test.",
                    "speaker": "Alice",
                    "frames": [0.1, 0.2]
                }
            ],
            [
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
        ]
        self.document = Document(self.document_data)

    def test_document_initialization(self):
        """Test that a document initializes correctly with given sentence data."""
        self.assertEqual(len(self.document.sentences), 2)
        self.assertEqual(self.document.sentences[0].start, 0.0)
        self.assertEqual(self.document.sentences[1].end, 5.0)

    def test_document_string_representation(self):
        """Test the string representation of the document."""
        expected_output = "(0.0:2.0) - Alice: Hello, this is a test.\n(2.5:5.0) - Bob: This is another sentence."
        self.assertEqual(str(self.document), expected_output)

    def test_get_plain_text(self):
        """Test retrieving raw text from the document."""
        expected_plain_text = "Alice: Hello, this is a test.\nBob: This is another sentence."
        self.assertEqual(self.document.get_plain_text(), expected_plain_text)

    def test_get_formatted_text(self):
        """Test retrieving formatted text with speaker labels."""
        formatted_text = self.document.get_formatted_text()
        self.assertIn("Alice:", formatted_text)
        self.assertIn("Bob:", formatted_text)

    def test_find_sentence(self):
        """Test finding a sentence by timestamp."""
        sentence1 = self.document.find_sentence(1.0)  # Inside first sentence
        sentence2 = self.document.find_sentence(3.0)  # Inside second sentence
        sentence3 = self.document.find_sentence(6.0)  # Outside of range, should return None

        self.assertIsNotNone(sentence1)
        self.assertEqual(sentence1.start, 0.0)
        self.assertEqual(sentence1.end, 2.0)

        self.assertIsNotNone(sentence2)
        self.assertEqual(sentence2.start, 2.5)
        self.assertEqual(sentence2.end, 5.0)

        self.assertIsNone(sentence3)  # Out of range

    def test_find_segment(self):
        """Test finding a segment by timestamp."""
        segment1 = self.document.find_segment(1.0)
        segment2 = self.document.find_segment(3.0)
        segment3 = self.document.find_segment(6.0)  # Should return None

        self.assertIsNotNone(segment1)
        self.assertEqual(segment1.start, 0.0)
        self.assertEqual(segment1.end, 2.0)

        self.assertIsNotNone(segment2)
        self.assertEqual(segment2.start, 2.5)
        self.assertEqual(segment2.end, 5.0)

        self.assertIsNone(segment3)  # Out of range

if __name__ == "__main__":
    unittest.main()
