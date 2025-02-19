from typing import List, Dict, Any, Callable
from .document import Document
from .track import TextTrack, KeyframeTrack

class DocumentAnalysis:
    """
    Contains static methods for analyzing and processing document data.
    """

    @staticmethod
    def list_to_document_from_segments(transcript_data: List[dict]) -> Document:
        """Convert a list of transcript data into a Document object."""

        assert transcript_data, "Transcript data must not be empty"
        assert all("text" in entry for entry in transcript_data), "Transcript data must contain 'text' field"
        assert all("start" in entry for entry in transcript_data), "Transcript data must contain 'start' field"
        assert all("end" in entry for entry in transcript_data), "Transcript data must contain 'end' field"
        assert all(entry["start"] <= entry["end"] for entry in transcript_data), "Start time must be less than or equal to end time"


        # Break into sentences based on capitalization and punctuation

        sentences = []
        current_sentence = []

        for entry in transcript_data:
            text = entry["text"].strip()
            if len(text) == 0:
                continue

            # Condition: Start a new sentence if text starts with a capital letter
            if text[0].isupper():
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = [entry]

            # Condition: End the sentence if punctuation is encountered
            elif text.endswith(".") or text.endswith("?") or text.endswith("!"):
                current_sentence.append(entry)
                sentences.append(current_sentence)
                current_sentence = []

            else:
                # Continue the existing sentence
                current_sentence.append(entry)

        # Add any remaining sentence
        if current_sentence:
            sentences.append(current_sentence)
        
        temp = []

        for sentence in sentences:
            temp.append({
                "start": sentence[0]["start"],
                "end": sentence[-1]["end"],
                "text": {
                    "text": " ".join([entry["text"].strip() for entry in sentence]),
                    "speaker": sentence[0].get("speaker", "UNKNOWN"),
                    },
                "keyframe": None
            })

        # Insert blank sentences where there is any gap
        i = 0
        while i < len(temp) - 1:
            if temp[i]["end"] < temp[i + 1]["start"]:
                temp.insert(i + 1, {
                    "start": temp[i]["end"],
                    "end": temp[i + 1]["start"],
                    "text": {
                        "text": "",
                        "speaker": "UNKNOWN",
                        },
                    "keyframe": None
                })
            i += 1

        return Document(temp, {
                        "text": TextTrack,
                        "keyframe": KeyframeTrack,
                        })

    @staticmethod
    def list_to_document_from_processed(transcript_data: List[dict], metadata: Dict[str, Any]=None) -> Document:
        """Convert a list of transcript data into a Document object."""

        assert transcript_data, "Transcript data must not be empty"
        assert all("text" in entry for entry in transcript_data), "Transcript data must contain 'text' field"
        assert all("start" in entry for entry in transcript_data), "Transcript data must contain 'start' field"
        assert all("end" in entry for entry in transcript_data), "Transcript data must contain 'end' field"
        assert all(entry["start"] <= entry["end"] for entry in transcript_data), "Start time must be less than or equal to end time"

        # # assert that if any transcript_data["keyframe"]["score"] exists, it does for all 
        # if any("keyframe" in entry for entry in transcript_data):
        #     assert all("keyframe" in entry and "score" in entry["keyframe"] for entry in transcript_data), "Transcript data must contain 'keyframe' field with 'score'"
        # # same for text score 
        # if any("text" in entry for entry in transcript_data):
        #     assert all("text" in entry and "score" in entry["text"] for entry in transcript_data), "Transcript data must contain 'text' field with 'score'"


        # Break into sentences based on capitalization and punctuation

        return Document(transcript_data, {
                        "text": TextTrack,
                        "keyframe": KeyframeTrack,
                        }, metadata)


