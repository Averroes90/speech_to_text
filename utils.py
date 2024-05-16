from re import I
from pydub import AudioSegment
import os
import wave
import pickle
import srt
import json
import html
import datetime
from handlers import get_translation_service_handler
import io
import tkinter as tk
from tkinter import filedialog
from nlp_utils2 import fix_none_stamps


def extract_and_convert_audio(
    input_path: str,
    input_filename: str,
    input_extension: str,
    output_path: str,
    output_filename: str,
    output_extension: str,
) -> tuple[bool, str]:
    try:
        video_file = f"{input_path}{input_filename}{input_extension}"
        audio_file = f"{output_path}{output_filename}{output_extension}"

        # Load the video file
        video = AudioSegment.from_file(video_file)

        # Convert to mono if not already
        if video.channels > 1:
            video = video.set_channels(1)

        # Export the audio in mono format
        video.export(audio_file, format=output_extension.lstrip("."))
        return True, "Audio extracted and converted successfully."
    except Exception as e:
        return False, str(e)


def get_sample_rate(audio_file_path: str) -> int:
    """
    Get the sample rate of the given audio file.
    """
    with wave.open(audio_file_path, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
    return sample_rate


def save_object_to_pickle(object: str, file_path: str) -> None:
    """Save transcription object to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(object, f)
    print("pickle file saved")


def load_object_from_pickle(file_path: str) -> any:
    """Load transcription object from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_srt_data(srt_data: str, file_path: str) -> None:
    # Parse the SRT data to ensure it's valid
    try:
        subtitles = list(srt.parse(srt_data))
    except Exception as e:
        print(f"Failed to parse SRT data: {e}")
        return

    # Generate the SRT content from the parsed data
    generated_srt = srt.compose(subtitles)

    # Save the SRT content to a file
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(generated_srt)
        print(f"SRT file saved successfully at {file_path}")
    except Exception as e:
        print(f"Failed to save SRT file: {e}")


def translate_srt(
    srt_content: str,
    service: str = "google",
    source_language: str = "it",
    target_language: str = "en",
    env_loaded: bool = False,
) -> str:
    # Get the translation service handler
    processor = get_translation_service_handler(service=service, env_loaded=env_loaded)

    # Split the content into blocks
    blocks = srt_content.strip().split("\n\n")
    translated_blocks = []

    for block in blocks:
        # Split each block into its components (index, timecode, and text)
        lines = block.split("\n")
        if len(lines) < 3:
            continue

        index = lines[0]
        timecode = lines[1]
        text = "\n".join(
            lines[2:]
        )  # Joining in case there are multiple lines of subtitles

        # Translate the text

        translated_text = processor.translate_text(
            text, source_language=source_language, target_language=target_language
        )
        translated_text = html.unescape(translated_text)

        # Rebuild the block with the translated text
        translated_block = f"{index}\n{timecode}\n{translated_text}\n"
        translated_blocks.append(translated_block)

    # Join all translated blocks back into a single string
    translated_srt_content = "\n\n".join(translated_blocks)
    return translated_srt_content


def seconds_to_srt_time(sec: float) -> str:
    """Convert seconds to SRT time format."""
    ms = int((sec - int(sec)) * 1000)
    return str(datetime.timedelta(seconds=int(sec))) + "," + f"{ms:03d}"


def create_srt(
    subtitles: list[tuple[str, float, float]], audio_duration: float = 10000
) -> str:
    srt_content = ""
    # print(f"subtiteles: {subtitles}")
    for index, (text, start, end) in enumerate(subtitles, start=1):
        # print(f"srt text {text}")
        # if start == None:
        #     start = 373
        # if end == None:
        #     end = 680
        if start == None or end == None:
            text, start, end = fix_none_stamps(
                max(index - 1, 0), subtitles, audio_duration
            )
        start_time = seconds_to_srt_time(start)

        end_time = seconds_to_srt_time(end)
        srt_content += f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content


def extract_path_details(full_path: str) -> tuple[str, str, str]:
    # Extract the directory path
    input_path = os.path.dirname(full_path)

    # Ensure the path ends with a slash
    if input_path and not input_path.endswith(os.sep):
        input_path += os.sep

    # Extract the base filename with extension
    base_filename = os.path.basename(full_path)

    # Split the base filename into filename and extension
    input_filename, input_extension = os.path.splitext(base_filename)

    return input_path, input_filename, input_extension


def get_sample_rate_bytesio(audio_data: io.BytesIO) -> int:
    """
    Get the sample rate of the given audio data.
    audio_data should be a BytesIO object containing audio data of any format supported by ffmpeg.
    """
    # Reset the pointer to the start of the BytesIO object
    audio_data.seek(0)

    # Load the audio file from BytesIO
    audio_segment = AudioSegment.from_file(audio_data)

    # Return the frame rate
    return audio_segment.frame_rate


def load_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def select_files():
    """Opens a file dialog and returns a list of selected file paths."""
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    root.call("wm", "attributes", ".", "-topmost", True)  # Bring the dialog on top
    file_paths = (
        filedialog.askopenfilenames()
    )  # This will allow selection of multiple files
    return list(file_paths)


def replace_extension(
    file_path: str, new_extension: str = ".srt", end_modifiers: str = ""
) -> str:
    """
    Replaces the current file extension with the new extension and optionally modifies the base filename.

    Parameters:
        file_path (str): The original file path.
        new_extension (str): The new file extension to apply, must start with a dot.
        end_modifiers (str): String to append to the filename before the extension.

    Returns:
        str: The modified file path with the new extension and any filename modifications.
    """
    # Split the filepath into the root and the existing extension
    root, _ = os.path.splitext(file_path)

    # Ensure the new extension starts with a dot
    if not new_extension.startswith("."):
        new_extension = "." + new_extension

    # Return the new file path with modifiers and new extension
    return root + end_modifiers + new_extension
