from typing import Tuple

class Segment:
    transcript: str
    text: str
    formatted_text: str
    speaker: str
    start: float
    end: float
    timestamp: Tuple[float, float]
    frames: list[float] # Change
    waveform: list[float] # Change

    def __init__(self, args) -> None:
        self.text = args.get("text")
        self.transcript = self.text
        self.formatted_text = args.get("formatted_text")
        self.speaker = args.get("speaker")
        self.start = args.get("start")
        self.end = args.get("end")
        self.timestamp = args.get("timestamp")
        self.frames = args.get("frames")
        self.waveform = args.get("waveform")

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end




class Sentence:
    transcript: list[Segment]
    text: str
    start: float
    end: float
    embeddings: dict[str, list[float]]
    text_score: float
    keyframe_score: int
    aggregated_score: float

    def __init__(self, args) -> None:
        sentence = args.get("sentence")
        if sentence:
            self.transcript = [Segment(s) for s in sentence]
        else:
            return
        self.text = " ".join([seg.text for seg in self.transcript])
        self.start = args.get("start")
        self.end = args.get("end")
        self.embeddings = args.get("embeddings")
        self.text_score = args.get("text_score")
        if self.text_score is None:
            self.text_score = 0
        self.keyframe_score = args.get("keyframe_score")
        if self.keyframe_score is None:
            self.keyframe_score = 0
        self.aggregated_score = args.get("aggregated_score")
        if self.aggregated_score is None:
            self.aggregated_score = 0

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end

    def find(self, ts: float) -> Segment | None:
        for seg in self.transcript:
            if seg.contains(ts):
                return seg
        return None

    def get_plain_text(self) -> str:
        return " ".join([seg.text for seg in self.transcript])


class Document:
    sentences: list[Sentence]

    def __init__(self, sentences: list[dict]) -> None:
        self.sentences = [Sentence(s) for s in sentences]

    def __str__(self) -> str:
        return "\n".join([f"({s.start}:{s.end}) - " + str(s) for s in self.sentences])

    def __repr__(self) -> str:
        return self.__str__()

    def get_plain_text(self) -> str:
        return "\n".join([s.get_plain_text() for s in self.sentences])

    def find_sentence(self, ts: float) -> Sentence | None:
        for s in self.sentences:
            if s.contains(ts):
                return s
        return None

    def find_segment(self, ts: float) -> Segment | None:
        for s in self.sentences:
            seg = s.find(ts)
            if seg:
                return seg
        return None

    def assign_embeddings(self, embeddings: list) -> None:
        assert len(embeddings) == len(self.sentences)

        for i, emb in enumerate(embeddings):
            self.sentences[i].embeddings = emb

    def assign_scores(self, text_scores: list, keyframe_scores: list) -> None:
        assert len(text_scores) == len(self.sentences)
        assert len(keyframe_scores) == len(self.sentences)

        for i, s in enumerate(self.sentences):
            s.text_score = text_scores[i]
            s.keyframe_score = keyframe_scores[i]

        # compute the normalized scores

        alpha = 0.5

        text_scores = [s.text_score for s in self.sentences]

        min_text_score = min(text_scores)
        max_text_score = max(text_scores)

        keyframe_scores = [s.keyframe_score for s in self.sentences]

        min_keyframe_score = min(keyframe_scores)
        max_keyframe_score = max(keyframe_scores)

        for s in self.sentences:
            s.aggregated_score = alpha * (s.text_score - min_text_score) / (max_text_score - min_text_score) + \
                                 (1 - alpha) * (s.keyframe_score - min_keyframe_score) / (max_keyframe_score - min_keyframe_score)

    @staticmethod
    def raw_to_sentences(raw: list[dict]) -> list[Sentence]:
        # from raw transcript, create sentences by checking for periods as the last character
        # assumes that each dict is {text: str, 'timestamp': (float, float), 'speaker': str, 'start': float, 'end': float, 'formatted_text': str}
        
        sentences = []
        sentence = []
        for r in raw:
            sentence.append(r)
            if r['text'][-1] == '.':
                sentences.append(Sentence({'sentence': sentence, 
                                           'start': sentence[0]['start'],
                                             'end': sentence[-1]['end']}))
                sentence = []
        if sentence:
            sentences.append(Sentence({'sentence': sentence, 
                                       'start': sentence[0]['start'],
                                         'end': sentence[-1]['end']}))
        return sentences
