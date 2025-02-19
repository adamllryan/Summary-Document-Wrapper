# Summary-Document-Wrapper

This repository provides a pipeline for generating extractive summaries of videos. It uses a combination of speech-to-text, sentence scoring, and keyframe extraction to identify the most important parts of a video.


## Directory Structure

To use this pipeline, your videos should be organized in the following directory structure:

```
 video_id_1/ source_video.mp4 ...other related files... video_id_2/ source_video.mp4 ...other related files... ...
```

Each video should be placed in a folder named after its ID. Within each folder, the video file should be named `source_video.mp4`. Other related files can also be stored in this folder.

## Document Structure

The pipeline utilizes a `Document` class to represent the video content. A document is created using the `DocumentAnalysis` class and its static methods. These methods primarily accept a list of sentence segments, which should adhere to a specific format (described in more detail below).

The static method `list_to_document_from_segments` is used for this purpose. The format for a sentence segment is ambiguous and can contain additional fields based on application. However it should contain the following keys: "text", "timestamp", "speaker", "start", "end", "formatted_text". These keys should contain the obvious values. This is formatted similarly to an example:

```json 

{ 
    "text": "But I'm not going to talk specifically about the attack on SHA today", 
    "timestamp": [ 6.66, 10.28 ], 
    "speaker": "SPEAKER_00", 
    "start": 6.66, 
    "end": 10.28, 
    "formatted_text": "SPEAKER_00: But I'm not going to talk specifically about the attack on SHA today" 
}
```

The `Document` class handles the ordering of segments into sentences using minimal Natural Language Processing (NLP) techniques. When exporting, it produces a list of `Sentence` objects with the following structure:

```json 

{
    "metadata": {
        "error": "If any",
        "filtered_timestamps": [(0.0, 5.76), ...]
    }
    "sentences": { 
        "start": 0.0, 
        "end": 5.76, 
        "timestamp": [ 0.0, 5.76 ], 
        "text": { 
            "text": "SHA stands for the secure hash algorithm, which is interesting given that they've just kind of been broken", 
            "speaker": "UNKNOWN", 
            "embeddings": {}, 
            "score": null 
        }, 
        "keyframe": { 
            "frames": [], 
            "score": null 
        } 
    }
}
```

To read this structure back into a Document, use the `DocumentAnalysis` method: 
```python
    def list_to_document_from_processed(transcript_data: List[dict], metadata: Dict[str, Any]=None) -> Document:
```

## Tracks

`Tracks` are a fundamental concept in this pipeline and provide a way to represent different aspects of a video. They are implementable and extensible classes. By default, two `Tracks` are provided: `Text` and `Keyframe`. You can create custom `Tracks` to represent other features or modalities.

## BatchExecutor

The `BatchExecutor` is a default implementation of the pipeline. It handles the processing of multiple videos and automatically fills out the default `Text` and `Keyframe` tracks. Each sentence within a document is associated with a set of tracks, enabling the pipeline to capture various information about the video content.

## Metadata

In addition to sentences and tracks, each document also stores a metadata object. This object contains relevant information about the video and is automatically written alongside the document data.

Note: This README provides a basic overview of the repository and its functionalities. I will add more information about Tracks and other advanced features later.
