import os
import subprocess
from typing import List, Tuple


class Splicer:
    """
    Splices video segments together based on selected timestamps.
    """

    def __init__(self, config: dict):
        self.config = config

    def splice(
        self, video_path: str, timestamps: List[Tuple[float, float]], output_path: str
    ):
        """Splices a video using FFmpeg based on a list of timestamps."""

        print("Splicing video")

        concat_list_file = "concat_list.txt"

        with open(concat_list_file, "w") as f:
            for start, end in timestamps:
                f.write(f"file '{video_path}'\n")
                f.write(f"inpoint {start}\n")
                f.write(f"outpoint {end}\n")

        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c",
            "copy",
            output_path,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error splicing video: {e}")
        os.remove(concat_list_file)
        print(f"Spliced video saved to {output_path}")
