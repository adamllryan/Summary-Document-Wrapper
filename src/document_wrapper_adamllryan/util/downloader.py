import os
import subprocess
from typing import Optional

class VideoDownloader:
    """
    Handles downloading videos from YouTube using yt-dlp and organizing them.
    """

    def __init__(self, download_dir: str = "videos"):
        """
        Initialize the downloader with a specified directory.
        :param download_dir: Directory where videos will be stored.
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_youtube_video(self, video_id: str) -> Optional[str]:
            """
            Downloads a YouTube video and saves it under the given directory.
            :param video_id: YouTube video ID
            :return: Path to downloaded video or None if failed
            """
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            output_dir = os.path.join(self.download_dir, video_id)
            output_path = os.path.join(output_dir, "source_video.mp4")
            os.makedirs(output_dir, exist_ok=True)

            try:
                result = subprocess.run(
                    [
                        "yt-dlp",
                        "-f", "bv*+ba/best",  # Select best available format with both video and audio
                        "--merge-output-format", "mp4",  # Ensure final format is MP4
                        "--quiet", "--no-warnings",  # Reduce verbosity
                        "--no-playlist",  # Only download a single video
                        "-o", output_path,
                        video_url
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if result.returncode == 0 and os.path.exists(output_path):
                    return output_path
            except subprocess.CalledProcessError as e:
                print(f"Error downloading video {video_id}: {e.stderr.decode().strip()}")
            except Exception as e:
                print(f"Unexpected error downloading video {video_id}: {str(e)}")
            
            return None

    def is_video_processed(self, video_id: str) -> bool:
        """
        Checks if a video has already been processed (i.e., summary exists).
        :param video_id: YouTube video ID
        :return: True if summary exists, False otherwise
        """
        video_folder = os.path.join(self.download_dir, video_id)
        return os.path.exists(os.path.join(video_folder, "summary.mp4"))

    def is_video_downloaded(self, video_id: str) -> bool:
        """
        Checks if a video has already been downloaded.
        :param video_id: YouTube video ID
        :return: True if video exists, False otherwise
        """
        video_folder = os.path.join(self.download_dir, video_id)
        return os.path.exists(os.path.join(video_folder, "source_video.mp4"))

    def get_next_unprocessed_video(self) -> Optional[str]:
        """
        Finds the next unprocessed video in the directory.
        :return: The first video ID without a summary, or None if all are processed.
        """
        for video_id in os.listdir(self.download_dir):
            if os.path.isdir(os.path.join(self.download_dir, video_id)) and not self.is_video_processed(video_id):
                return video_id
        return None

if __name__ == "__main__":
    downloader = VideoDownloader()
    test_video_id = "e9Eds2Rc_x8"  # Replace with an actual YouTube video ID
    if not downloader.is_video_processed(test_video_id):
        print("Downloading...")
        downloader.download_youtube_video(test_video_id)
    else:
        print("Video already processed.")
