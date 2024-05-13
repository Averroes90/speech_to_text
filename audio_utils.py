from pydub import AudioSegment
import webrtcvad
import io
import os
from typing import Optional


def load_audio(file_path: str) -> Optional[AudioSegment]:
    """
    Load an audio file from the specified path using pydub.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        AudioSegment: An audio data object.
    """
    try:
        audio_data = AudioSegment.from_file(file_path)
        return audio_data
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None


def detect_speech_segments(
    audio_data: AudioSegment, frame_duration: int = 30, aggressiveness: int = 3
) -> list[tuple[float, float]]:
    vad = webrtcvad.Vad(aggressiveness)
    audio_data = audio_data.set_channels(1).set_sample_width(
        2
    )  # Ensuring mono audio and 16-bit width

    if audio_data.frame_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError(
            f"Unsupported sample rate: {audio_data.frame_rate}. Supported rates: 8000, 16000, 32000, 48000 Hz"
        )

    speech_segments = []
    is_speech = False
    start_time = None

    try:
        for i in range(0, len(audio_data), frame_duration):
            frame = audio_data[i : i + frame_duration]

            if len(frame) < frame_duration:
                missing_duration = frame_duration - len(frame)
                frame += AudioSegment.silent(
                    duration=missing_duration, frame_rate=audio_data.frame_rate
                )

            frame_raw_data = frame.raw_data
            is_speech_frame = vad.is_speech(
                frame_raw_data, audio_data.frame_rate
            )  # Process frame_raw_data with VAD

            current_time = i

            if not is_speech and is_speech_frame:
                start_time = current_time
                is_speech = True
            elif is_speech and not is_speech_frame:
                end_time = current_time
                speech_segments.append((start_time, end_time))
                is_speech = False

        if is_speech:
            speech_segments.append((start_time, len(audio_data)))

    except Exception as e:
        print(f"Error processing VAD: {e}")
        # If an error occurs, return fixed-length segments
        segment_length = int(
            len(audio_data) / 5
        )  # Example fixed length in milliseconds
        return [
            (j, j + segment_length) for j in range(0, len(audio_data), segment_length)
        ]

    return speech_segments


def combine_speech_segments(
    speech_segments: list[tuple[float, float]],
    audio_data: AudioSegment,
    max_size_mb: int,
) -> list[tuple[float, float]]:
    """
    Combine speech segments into larger chunks without exceeding the file size limit.

    Args:
        speech_segments (List[tuple]): List of tuples with start and end times of detected speech segments.
        audio_data (AudioSegment): The entire loaded audio data.
        max_size_mb (int): Maximum size in megabytes for any single audio segment.

    Returns:
        List[tuple]: A list of tuples where each tuple contains start and end times of the combined segments.
    """
    max_size = max_size_mb * 1000000  # Convert MB to bytes for comparison
    combined_segments = []
    current_segment = None
    new_start = True
    start = 0

    for speech_segment in speech_segments:
        end = speech_segment[1]
        segment = audio_data[start:end]
        segment_size = len(segment.raw_data)

        if current_segment is None:
            current_segment = (start, end)
            new_start = False
        else:
            if segment_size <= max_size:
                # Extend the current segment
                current_segment = (current_segment[0], end)
                new_start = False
            else:
                # Finalize the current segment and start a new one
                combined_segments.append(current_segment)
                start = current_segment[1]
                current_segment = (start, end)
                segment = audio_data[start:end]
                segment_size = len(segment.raw_data)
                new_start = True

    # Add the last segment if it exists
    if not new_start:
        combined_segments.append(current_segment)

    return combined_segments


def calculate_segment_duration(segment: tuple[float, float]) -> int:
    """
    Calculate the duration of a specified segment of audio data.

    Args:
        segment (tuple): A tuple containing the start and end times of the segment in milliseconds.

    Returns:
        int: Duration of the segment in milliseconds.
    """
    start, end = segment
    segment_duration = end - start
    return segment_duration


def segment_audio(
    audio_data: AudioSegment, segments: list[tuple[float, float]]
) -> list[io.BytesIO]:
    chunks = []
    for start, end in segments:
        chunks.append(audio_data[start:end])

    in_memory_files = []
    for index, chunk in enumerate(chunks):
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)
        buffer.name = f"segment_{index}.wav"
        in_memory_files.append(buffer)

    return in_memory_files


def batch_audio(
    audio_data: AudioSegment, max_size_mb: int
) -> tuple[list[io.BytesIO], list[tuple[float, float]]]:
    # audio_data = load_audio(audio_path)
    speach_segments = detect_speech_segments(audio_data)
    combined_segments = combine_speech_segments(
        speach_segments, audio_data, max_size_mb
    )
    audio_segments = segment_audio(audio_data, combined_segments)
    return audio_segments, combined_segments


def extract_audio(
    input_path: str,
    input_filename: str,
    input_extension: str,
    output_extension: str = ".wav",
) -> tuple[bool, str, Optional[io.BytesIO]]:
    try:
        video_file = f"{input_path}{input_filename}{input_extension}"
        # Load the video file
        video = AudioSegment.from_file(video_file)

        # Convert to mono if not already
        if video.channels > 1:
            video = video.set_channels(1)

        # Create a BytesIO object to hold the audio data
        audio_buffer = io.BytesIO()
        audio_buffer.name = f"{input_filename}{output_extension}"
        audio_buffer.sample_rate = video.frame_rate

        # Export the audio in mono format
        video.export(audio_buffer, format=output_extension.lstrip("."))

        # It's important to seek back to the start of the BytesIO object before returning
        audio_buffer.seek(0)

        return True, "Audio extracted and converted successfully.", audio_buffer
    except Exception as e:
        return False, str(e), None


def load_audio_from_bytesio(bytes_io: io.BytesIO) -> Optional[AudioSegment]:
    """
    Load an audio file from a BytesIO object using pydub.

    Args:
        bytes_io (io.BytesIO): A BytesIO object containing the audio data, with a 'name' attribute.

    Returns:
        AudioSegment: The audio data object if the file was successfully loaded, or None if an error occurs.
    """
    try:
        bytes_io.seek(0)  # Ensure we're at the start of the BytesIO stream
        if hasattr(bytes_io, "name"):
            # Extract the extension from the filename
            _, file_extension = os.path.splitext(bytes_io.name)
            # Remove the leading period from the extension for use with pydub
            file_extension = file_extension.lstrip(".")
        else:
            raise AttributeError(
                "BytesIO object does not have a 'name' attribute with a filename."
            )
        # print(f"file extension {file_extension}")
        audio_data = AudioSegment.from_file(bytes_io, format=file_extension)
        return audio_data
    except Exception as e:
        print(f"Error loading audio from BytesIO: {e}")
        return None
