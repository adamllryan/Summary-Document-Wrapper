import os
import json
import subprocess
from batch_runner import BatchRunner
from youtube_dl import YoutubeDL

# Configuration for the batch processing
config = {
    "models": {"transcription": "openai/whisper-small"},
    "summarization_model": "facebook/bart-large-cnn",
    "output_dir": "output/",
    "repo_url": "https://github.com/your-repo.git",
    "branch": "refactor",
    "youtube_ids": ["dQw4w9WgXcQ"]  # Replace with a list of YouTube video IDs
}

# Ensure output directory exists
os.makedirs(config["output_dir"], exist_ok=True)

# Clone or pull the latest refactor branch from GitHub
if not os.path.exists("repo"):
    subprocess.run(["git", "clone", "-b", config["branch"], config["repo_url"], "repo"])
else:
    subprocess.run(["git", "-C", "repo", "pull", "origin", config["branch"]])

# Initialize the batch processor
batch_runner = BatchRunner(config)

# Download YouTube videos
video_files = []
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': os.path.join(config["output_dir"], '%(id)s.%(ext)s')
}
with YoutubeDL(ydl_opts) as ydl:
    for video_id in config["youtube_ids"]:
        info_dict = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        video_filename = ydl.prepare_filename(info_dict)
        video_files.append(video_filename)

# Define reference transcripts (if available)
reference_transcripts = {video: "Expected summary text." for video in video_files}  # Modify as needed

# Run batch processing
results = batch_runner.run_batch(video_files, reference_transcripts)

# Save results to a JSON file
output_path = os.path.join(config["output_dir"], "batch_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

# Display results
print("Batch Processing Complete! Results:")
print(json.dumps(results, indent=4))
