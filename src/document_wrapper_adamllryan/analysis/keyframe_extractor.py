from typing import Dict, List
import cv2
import random
import numpy as np 
from sklearn.cluster import KMeans 
from datetime import timedelta

class KeyframeExtractor:
    """
    Extracts keyframes from a video and assigns them to sentences based on timestamps.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config
    
    def extract(self, video_path: str, document: Document):
        """Extracts keyframes from the video and assigns them to sentences in the document."""
        # video_path = document.metadata.get("video_path", "")
        print("Extracting keyframes")
        keyframes = self._extract_keyframes(video_path)
        
        if not keyframes:
            print("Warning: No keyframes extracted. This may indicate a cv2 error or an unreadable video file.")
            return
        
        keyframe_counts = self._assign_keyframes_to_sentences(keyframes, document)
        document.call_track_method("set_score", "keyframes", keyframe_counts)
    
    def _extract_keyframes(self, video_path: str):
        """Extracts keyframes from the video using frame skipping and clustering."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = self.config["skip_frames"]
        crop_size = self.config["crop_size"]
        n_clusters = self.config["n_clusters"]
        start_time = timedelta(seconds=0)
        keyframes = []
        frame_id = 0
        
        success, frame = cap.read()
        if not success:
            print("Warning: No frames read from video. Exiting keyframe extraction.")
            cap.release()
            return []
        
        while success:
            timestamp = start_time + timedelta(seconds=frame_id / fps)
            h, w, _ = frame.shape
            crop_h, crop_w = crop_size
            x, y = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
            cropped_frame = frame[y:y + crop_h, x:x + crop_w]
            compressed_frame = cv2.resize(cropped_frame, (50, 50))
            keyframes.append((compressed_frame, timestamp.total_seconds()))
            
            frame_id += 1
            for _ in range(skip_frames):
                success = cap.grab()
                if not success:
                    break
                frame_id += 1
            
            success, frame = cap.read()
        cap.release()
        
        print(f"Extracted {len(keyframes)} keyframes")
        return self._cluster_keyframes(keyframes, n_clusters)
    
    def _cluster_keyframes(self, keyframes, n_clusters: int):
        """Clusters keyframes to find the most representative ones."""
        print("Clustering keyframes")
        if not keyframes:
            print("Warning: No keyframes available for clustering.")
            return []
        
        keyframes_array = np.array([kf[0].flatten() for kf in keyframes])
        timestamps = [kf[1] for kf in keyframes]
        n_clusters = min(len(keyframes) // 2, n_clusters)
        if n_clusters == 0:
            print("Warning: Not enough keyframes for clustering.")
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(keyframes_array)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        representative_keyframes = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            if cluster_indices.size > 0:
                distances = np.linalg.norm(keyframes_array[cluster_indices] - cluster_centers[cluster], axis=1)
                closest_index = cluster_indices[np.argmin(distances)]
                representative_keyframes.append({"frame": keyframes_array[closest_index].reshape(50, 50, 3),
                                                 "timestamp": timestamps[closest_index]})
        return representative_keyframes
    
    def _assign_keyframes_to_sentences(self, keyframes, document: Document):
        """Counts keyframes for each sentence based on their timestamps."""
        sentence_keyframe_counts = [0] * len(document.sentences)
        for kf in keyframes:
            timestamp = kf["timestamp"]
            for i, sentence in enumerate(document.sentences):
                if sentence.contains(timestamp):
                    sentence_keyframe_counts[i] += 1
                    break
        return sentence_keyframe_counts
