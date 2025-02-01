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

    def test_list_to_document(self):
        """Test conversion of a list of transcript data into a Document object."""
        transcript_data = [
    {
        "text": " We're no strangers to love",
        "timestamp": [
            0.0,
            22.0
        ],
        "speaker": "UNKNOWN",
        "start": 0.0,
        "end": 22.0,
        "formatted_text": "UNKNOWN:  We're no strangers to love"
    },
    {
        "text": " You know the rules",
        "timestamp": [
            22.0,
            24.14
        ],
        "speaker": "UNKNOWN",
        "start": 22.0,
        "end": 24.14,
        "formatted_text": "UNKNOWN:  You know the rules"
    },
    {
        "text": " And so do I.",
        "timestamp": [
            24.14,
            26.0
        ],
        "speaker": "SPEAKER_00",
        "start": 24.14,
        "end": 26.0,
        "formatted_text": "SPEAKER_00:  And so do I."
    },
    {
        "text": " I feel commitments while I'm thinking of.",
        "timestamp": [
            27.06,
            30.48
        ],
        "speaker": "UNKNOWN",
        "start": 27.06,
        "end": 30.48,
        "formatted_text": "UNKNOWN:  I feel commitments while I'm thinking of."
    },
    {
        "text": " You wouldn't get this from any other guy.",
        "timestamp": [
            31.22,
            34.74
        ],
        "speaker": "UNKNOWN",
        "start": 31.22,
        "end": 34.74,
        "formatted_text": "UNKNOWN:  You wouldn't get this from any other guy."
    },
    {
        "text": " I just want to tell you how I'm feeling.",
        "timestamp": [
            35.24,
            39.02
        ],
        "speaker": "SPEAKER_00",
        "start": 35.24,
        "end": 39.02,
        "formatted_text": "SPEAKER_00:  I just want to tell you how I'm feeling."
    },
    {
        "text": " Gotta make you understand.",
        "timestamp": [
            40.22,
            42.0
        ],
        "speaker": "UNKNOWN",
        "start": 40.22,
        "end": 42.0,
        "formatted_text": "UNKNOWN:  Gotta make you understand."
    },
    {
        "text": " Never gonna give you up.",
        "timestamp": [
            43.12,
            44.96
        ],
        "speaker": "UNKNOWN",
        "start": 43.12,
        "end": 44.96,
        "formatted_text": "UNKNOWN:  Never gonna give you up."
    },
    {
        "text": " Never gonna let you down Never gonna run around and desert you",
        "timestamp": [
            47.18,
            51.06
        ],
        "speaker": "SPEAKER_00",
        "start": 47.18,
        "end": 51.06,
        "formatted_text": "SPEAKER_00:  Never gonna let you down Never gonna run around and desert you"
    },
    {
        "text": " Never gonna make you cry",
        "timestamp": [
            51.06,
            53.42
        ],
        "speaker": "SPEAKER_00",
        "start": 51.06,
        "end": 53.42,
        "formatted_text": "SPEAKER_00:  Never gonna make you cry"
    },
    {
        "text": " Never gonna say goodbye",
        "timestamp": [
            53.42,
            55.36
        ],
        "speaker": "SPEAKER_00",
        "start": 53.42,
        "end": 55.36,
        "formatted_text": "SPEAKER_00:  Never gonna say goodbye"
    },
    {
        "text": " Never gonna tell a lie and hurt you",
        "timestamp": [
            55.36,
            59.6
        ],
        "speaker": "UNKNOWN",
        "start": 55.36,
        "end": 59.6,
        "formatted_text": "UNKNOWN:  Never gonna tell a lie and hurt you"
    },
    {
        "text": " We've known each other for so long",
        "timestamp": [
            59.6,
            64.28
        ],
        "speaker": "UNKNOWN",
        "start": 59.6,
        "end": 64.28,
        "formatted_text": "UNKNOWN:  We've known each other for so long"
    },
    {
        "text": " Your heart's been aching but you're too shy to say it",
        "timestamp": [
            64.28,
            69.04
        ],
        "speaker": "UNKNOWN",
        "start": 64.28,
        "end": 69.04,
        "formatted_text": "UNKNOWN:  Your heart's been aching but you're too shy to say it"
    },
    {
        "text": " Inside we both know what's been going on",
        "timestamp": [
            69.04,
            72.72
        ],
        "speaker": "UNKNOWN",
        "start": 69.04,
        "end": 72.72,
        "formatted_text": "UNKNOWN:  Inside we both know what's been going on"
    },
    {
        "text": " We know the game and we're gonna play it",
        "timestamp": [
            72.72,
            77.18
        ],
        "speaker": "UNKNOWN",
        "start": 72.72,
        "end": 77.18,
        "formatted_text": "UNKNOWN:  We know the game and we're gonna play it"
    },
    {
        "text": " And if you ask me how I'm feeling",
        "timestamp": [
            77.18,
            81.28
        ],
        "speaker": "UNKNOWN",
        "start": 77.18,
        "end": 81.28,
        "formatted_text": "UNKNOWN:  And if you ask me how I'm feeling"
    },
    {
        "text": " Don't tell me you're too Blind to see",
        "timestamp": [
            81.28,
            85.08
        ],
        "speaker": "UNKNOWN",
        "start": 81.28,
        "end": 85.08,
        "formatted_text": "UNKNOWN:  Don't tell me you're too Blind to see"
    },
    {
        "text": " Never gonna give you up",
        "timestamp": [
            85.08,
            87.22
        ],
        "speaker": "SPEAKER_00",
        "start": 85.08,
        "end": 87.22,
        "formatted_text": "SPEAKER_00:  Never gonna give you up"
    },
    {
        "text": " Never gonna let you down",
        "timestamp": [
            87.22,
            89.44
        ],
        "speaker": "SPEAKER_00",
        "start": 87.22,
        "end": 89.44,
        "formatted_text": "SPEAKER_00:  Never gonna let you down"
    },
    {
        "text": " Never gonna run around",
        "timestamp": [
            89.44,
            91.42
        ],
        "speaker": "SPEAKER_00",
        "start": 89.44,
        "end": 91.42,
        "formatted_text": "SPEAKER_00:  Never gonna run around"
    },
    {
        "text": " And desert you",
        "timestamp": [
            91.42,
            93.28
        ],
        "speaker": "SPEAKER_00",
        "start": 91.42,
        "end": 93.28,
        "formatted_text": "SPEAKER_00:  And desert you"
    },
    {
        "text": " Never gonna make you cry",
        "timestamp": [
            93.28,
            95.66
        ],
        "speaker": "SPEAKER_00",
        "start": 93.28,
        "end": 95.66,
        "formatted_text": "SPEAKER_00:  Never gonna make you cry"
    },
    {
        "text": " Never gonna say goodbye",
        "timestamp": [
            95.66,
            97.56
        ],
        "speaker": "UNKNOWN",
        "start": 95.66,
        "end": 97.56,
        "formatted_text": "UNKNOWN:  Never gonna say goodbye"
    },
    {
        "text": " Never gonna tell a lie",
        "timestamp": [
            97.56,
            100.14
        ],
        "speaker": "UNKNOWN",
        "start": 97.56,
        "end": 100.14,
        "formatted_text": "UNKNOWN:  Never gonna tell a lie"
    },
    {
        "text": " And hurt you",
        "timestamp": [
            100.14,
            101.76
        ],
        "speaker": "UNKNOWN",
        "start": 100.14,
        "end": 101.76,
        "formatted_text": "UNKNOWN:  And hurt you"
    },
    {
        "text": " Never gonna give you up",
        "timestamp": [
            101.76,
            104.12
        ],
        "speaker": "UNKNOWN",
        "start": 101.76,
        "end": 104.12,
        "formatted_text": "UNKNOWN:  Never gonna give you up"
    },
    {
        "text": " Never gonna let you up Never gonna let you down",
        "timestamp": [
            104.12,
            106.32
        ],
        "speaker": "UNKNOWN",
        "start": 104.12,
        "end": 106.32,
        "formatted_text": "UNKNOWN:  Never gonna let you up Never gonna let you down"
    },
    {
        "text": " Never gonna run around",
        "timestamp": [
            106.32,
            108.32
        ],
        "speaker": "UNKNOWN",
        "start": 106.32,
        "end": 108.32,
        "formatted_text": "UNKNOWN:  Never gonna run around"
    },
    {
        "text": " And desert you",
        "timestamp": [
            108.32,
            110.2
        ],
        "speaker": "UNKNOWN",
        "start": 108.32,
        "end": 110.2,
        "formatted_text": "UNKNOWN:  And desert you"
    },
    {
        "text": " Never gonna make you cry",
        "timestamp": [
            110.2,
            112.0
        ],
        "speaker": "SPEAKER_00",
        "start": 110.2,
        "end": 112.0,
        "formatted_text": "SPEAKER_00:  Never gonna make you cry"
    },
    {
        "text": " Never gonna say goodbye",
        "timestamp": [
            112.56,
            114.0
        ],
        "speaker": "SPEAKER_00",
        "start": 112.56,
        "end": 114.0,
        "formatted_text": "SPEAKER_00:  Never gonna say goodbye"
    },
    {
        "text": " Never gonna tell a lie",
        "timestamp": [
            114.56,
            116.0
        ],
        "speaker": "SPEAKER_00",
        "start": 114.56,
        "end": 116.0,
        "formatted_text": "SPEAKER_00:  Never gonna tell a lie"
    },
    {
        "text": " And hurt you",
        "timestamp": [
            117.04,
            118.0
        ],
        "speaker": "UNKNOWN",
        "start": 117.04,
        "end": 118.0,
        "formatted_text": "UNKNOWN:  And hurt you"
    },
    {
        "text": " Give you up",
        "timestamp": [
            118.8,
            123.0
        ],
        "speaker": "UNKNOWN",
        "start": 118.8,
        "end": 123.0,
        "formatted_text": "UNKNOWN:  Give you up"
    },
    {
        "text": " Give you love Give you love",
        "timestamp": [
            123.0,
            127.0
        ],
        "speaker": "UNKNOWN",
        "start": 123.0,
        "end": 127.0,
        "formatted_text": "UNKNOWN:  Give you love Give you love"
    },
    {
        "text": " Never gonna give, never gonna give",
        "timestamp": [
            127.52,
            130.04
        ],
        "speaker": "UNKNOWN",
        "start": 127.52,
        "end": 130.04,
        "formatted_text": "UNKNOWN:  Never gonna give, never gonna give"
    },
    {
        "text": " Give you love",
        "timestamp": [
            130.04,
            131.5
        ],
        "speaker": "UNKNOWN",
        "start": 130.04,
        "end": 131.5,
        "formatted_text": "UNKNOWN:  Give you love"
    },
    {
        "text": " Never gonna give, never gonna give",
        "timestamp": [
            131.5,
            134.28
        ],
        "speaker": "UNKNOWN",
        "start": 131.5,
        "end": 134.28,
        "formatted_text": "UNKNOWN:  Never gonna give, never gonna give"
    },
    {
        "text": " Give you love",
        "timestamp": [
            134.28,
            135.78
        ],
        "speaker": "UNKNOWN",
        "start": 134.28,
        "end": 135.78,
        "formatted_text": "UNKNOWN:  Give you love"
    },
    {
        "text": " We've known each other for so long",
        "timestamp": [
            135.78,
            140.3
        ],
        "speaker": "UNKNOWN",
        "start": 135.78,
        "end": 140.3,
        "formatted_text": "UNKNOWN:  We've known each other for so long"
    },
    {
        "text": " Your heart's been aching but you're too shy to say it",
        "timestamp": [
            140.3,
            145.34
        ],
        "speaker": "UNKNOWN",
        "start": 140.3,
        "end": 145.34,
        "formatted_text": "UNKNOWN:  Your heart's been aching but you're too shy to say it"
    },
    {
        "text": " Inside we both know what's been going on",
        "timestamp": [
            145.34,
            148.72
        ],
        "speaker": "SPEAKER_00",
        "start": 145.34,
        "end": 148.72,
        "formatted_text": "SPEAKER_00:  Inside we both know what's been going on"
    },
    {
        "text": " We know the game and we're gonna play it",
        "timestamp": [
            148.72,
            153.2
        ],
        "speaker": "UNKNOWN",
        "start": 148.72,
        "end": 153.2,
        "formatted_text": "UNKNOWN:  We know the game and we're gonna play it"
    },
    {
        "text": " I just wanna tell you how I'm feeling",
        "timestamp": [
            153.2,
            157.58
        ],
        "speaker": "UNKNOWN",
        "start": 153.2,
        "end": 157.58,
        "formatted_text": "UNKNOWN:  I just wanna tell you how I'm feeling"
    },
    {
        "text": " Gotta make you understand",
        "timestamp": [
            157.58,
            161.12
        ],
        "speaker": "UNKNOWN",
        "start": 157.58,
        "end": 161.12,
        "formatted_text": "UNKNOWN:  Gotta make you understand"
    },
    {
        "text": " Never gonna give you up",
        "timestamp": [
            161.12,
            163.26
        ],
        "speaker": "UNKNOWN",
        "start": 161.12,
        "end": 163.26,
        "formatted_text": "UNKNOWN:  Never gonna give you up"
    },
    {
        "text": " Never gonna let you down Never gonna run around and desert you",
        "timestamp": [
            163.26,
            186.22
        ],
        "speaker": "UNKNOWN",
        "start": 163.26,
        "end": 186.22,
        "formatted_text": "UNKNOWN:  Never gonna let you down Never gonna run around and desert you"
    },
    {
        "text": " Never gonna make you cry",
        "timestamp": [
            186.22,
            188.6
        ],
        "speaker": "UNKNOWN",
        "start": 186.22,
        "end": 188.6,
        "formatted_text": "UNKNOWN:  Never gonna make you cry"
    },
    {
        "text": " Never gonna say goodbye",
        "timestamp": [
            188.6,
            190.44
        ],
        "speaker": "UNKNOWN",
        "start": 188.6,
        "end": 190.44,
        "formatted_text": "UNKNOWN:  Never gonna say goodbye"
    },
    {
        "text": " Never gonna tell a lie and hurt you",
        "timestamp": [
            190.44,
            194.7
        ],
        "speaker": "SPEAKER_00",
        "start": 190.44,
        "end": 194.7,
        "formatted_text": "SPEAKER_00:  Never gonna tell a lie and hurt you"
    },
    {
        "text": " Never gonna give you up",
        "timestamp": [
            194.7,
            197.1
        ],
        "speaker": "UNKNOWN",
        "start": 194.7,
        "end": 197.1,
        "formatted_text": "UNKNOWN:  Never gonna give you up"
    },
    {
        "text": " Never gonna let you down",
        "timestamp": [
            197.1,
            199.26
        ],
        "speaker": "UNKNOWN",
        "start": 197.1,
        "end": 199.26,
        "formatted_text": "UNKNOWN:  Never gonna let you down"
    },
    {
        "text": " Never gonna run around and desert you",
        "timestamp": [
            199.26,
            203.34
        ],
        "speaker": "UNKNOWN",
        "start": 199.26,
        "end": 203.34,
        "formatted_text": "UNKNOWN:  Never gonna run around and desert you"
    },
    {
        "text": " Never gonna make you cry.",
        "timestamp": [
            203.34,
            205.5
        ],
        "speaker": "UNKNOWN",
        "start": 203.34,
        "end": 205.5,
        "formatted_text": "UNKNOWN:  Never gonna make you cry."
    },
    {
        "text": " And I'm going to say goodbye.",
        "timestamp": [
            205.78,
            207.4
        ],
        "speaker": "UNKNOWN",
        "start": 205.78,
        "end": 207.4,
        "formatted_text": "UNKNOWN:  And I'm going to say goodbye."
    },
    {
        "text": " And I'm going to tell you why.",
        "timestamp": [
            207.92,
            210.02
        ],
        "speaker": "UNKNOWN",
        "start": 207.92,
        "end": 210.02,
        "formatted_text": "UNKNOWN:  And I'm going to tell you why."
    }
]

        document = DocumentAnalysis.list_to_document(transcript_data)

        self.assertEqual(len(document.sentences), len(transcript_data))
        self.assertEqual(document.sentences[0].start, 0.0)
        self.assertEqual(document.sentences[1].end, 210.02)

if __name__ == "__main__":
    unittest.main()
