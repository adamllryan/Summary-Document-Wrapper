class Segment:
    text: str
    start: float
    end: float
    frames: list[float]
    waveform: list[float]

    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def contains(self, ts: float):
        return self.start <= ts <= self.end


class Sentence:
    text: list[Segment]
    score: float
    start: float
    end: float

    def __init__(self, text: list[Segment] | dict, score: float = None):
        if isinstance(text, dict):
            self.text = [Segment(t, start, end) for (start, end), t in text.items()]
        else:
            self.text = [Segment(t["text"], t["start"], t["end"]) for t in text]
        self.score = score
        self.start = self.text[0].start
        self.end = self.text[-1].end

    def __str__(self):
        return " ".join([seg.text for seg in self.text])

    def __repr__(self):
        return self.__str__()

    def contains(self, ts: float):
        return self.start <= ts <= self.end

    def find(self, ts: float):
        for seg in self.text:
            if seg.contains(ts):
                return seg
        return None


class Document:
    sentences: list[Sentence]

    def __init__(self, sentences: list[Sentence] | dict):
        if isinstance(sentences, dict):
            self.sentences = [Sentence(s) for s in sentences.values()]
        else:
            self.sentences = [Sentence(s) for s in sentences]

    def __str__(self):
        return "\n".join([f"({s.start}:{s.end}) - " + str(s) for s in self.sentences])

    def __repr__(self):
        return self.__str__()

    def get_plain_text(self):
        return "\n".join([s.get_plain_text() for s in self.sentences])

    def find_sentence(self, ts: float):
        for s in self.sentences:
            if s.contains(ts):
                return s
        return None

    def find_segment(self, ts: float):
        for s in self.sentences:
            seg = s.find(ts)
            if seg:
                return seg
        return None
