from document import Document

sample_sentence = {
    (0.0, 4.99): "The quick brown",
    (5.0, 9.99): "fox jumps over",
    (10.0, 14.99): "the lazy dog.",
}

sample_sentence2 = [
    {
        "text": "Lorem ipsum",
        "start": 15.0,
        "end": 19.99,
    },
    {
        "text": "dolor sit amet,",
        "start": 20.0,
        "end": 24.99,
    },
    {
        "text": "consectetur adipiscing elit.",
        "start": 25.0,
        "end": 29.99,
    },
]

sample_document = {
    0: sample_sentence,
    1: sample_sentence2,
}

document = Document(sample_document)
document2 = Document(sample_document)

print(document)


print("Sentence 1: ", document.sentences[0])
print("Sentence at 5.0s: ", document.find_sentence(5.0))
print("Sentence fragment at 20.0s: ", document2.find_segment(20.0))
print("Document 2: ", document2)
print(
    "Start time of document 2, sentence at 12.7s: ",
    document2.find_sentence(12.70).start,
)
print(
    "End time of document 2, sentence segment at 12.7s: ",
    document2.find_segment(12.70).end,
)
