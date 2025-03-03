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

        # concat_list_file = "concat_list.txt"
        #
        # with open(concat_list_file, "w") as f:
        #     for start, end in timestamps:
        #         f.write(f"file '{video_path}'\n")
        #         f.write(f"inpoint {start}\n")
        #         f.write(f"outpoint {end}\n")
        #
        # # Remove video if exists
        # if os.path.exists(output_path):
        #     os.remove(output_path)
        #
        # command = [
        #     "ffmpeg",
        #     "-f",
        #     "concat",
        #     "-safe",
        #     "0",
        #     "-i",
        #     concat_list_file,
        #     "-c",
        #     "copy",
        #     output_path,
        # ]
        # try:
        #     subprocess.run(command, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error splicing video: {e}")
        # os.remove(concat_list_file)
        # print(f"Spliced video saved to {output_path}")

        if os.path.exists(output_path):
            os.remove(output_path)

        # Create FFmpeg filter_complex for fast splicing
        filter_complex = "".join(
            [
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
                for i, (start, end) in enumerate(timestamps)
            ]
        )

        # Create FFmpeg concat filter
        concat_inputs = (
            "".join([f"[v{i}][a{i}]" for i in range(len(timestamps))])
            + f"concat=n={len(timestamps)}:v=1:a=1[outv][outa]"
        )

        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            video_path,  # Input file
            "-filter_complex",
            filter_complex + concat_inputs,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "copy",  # Fastest processing: no re-encoding
            "-c:a",
            "copy",
            output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Spliced video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error splicing video: {e}")
