import unittest
from document_wrapper_adamllryan.doc.document import Document
from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis

class TestDocumentAnalysis(unittest.TestCase):
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

    def test_assign_scores(self):
        """Test that scores are correctly assigned and aggregated."""
        text_scores = [0.8, 0.5]
        keyframe_scores = [10, 20]

        DocumentAnalysis.assign_scores(self.document, text_scores, keyframe_scores)

        for i, sentence in enumerate(self.document.sentences):
            self.assertEqual(sentence.text_score, text_scores[i])
            self.assertEqual(sentence.keyframe_score, keyframe_scores[i])
            self.assertIsInstance(sentence.aggregated_score, float)

    def test_score_normalization(self):
        """Test that aggregated scores are correctly normalized."""
        text_scores = [0.0, 1.0]
        keyframe_scores = [5, 15]

        DocumentAnalysis.assign_scores(self.document, text_scores, keyframe_scores)

        first_sentence = self.document.sentences[0]
        second_sentence = self.document.sentences[1]

        # Normalized formula: 0.5 * normalized_text_score + 0.5 * normalized_keyframe_score
        self.assertEqual(first_sentence.aggregated_score, 0.0)  # Min values
        self.assertEqual(second_sentence.aggregated_score, 1.0)  # Max values

    def test_raw_to_sentences(self):
        """Test conversion of a Document's sentences into a raw sentence structure."""
        raw_sentences = DocumentAnalysis.raw_to_sentences(self.document)

        self.assertEqual(len(raw_sentences), len(self.document.sentences))

        for i, sentence in enumerate(raw_sentences):
            for j, segment in enumerate(sentence):  # Iterate over segments in sentence
                self.assertEqual(
                    segment[0]["tracks"]["transcript"]["text"], self.document.sentences[i].segments[j].get_track("transcript")["text"]
                )

if __name__ == "__main__":
    unittest.main()
