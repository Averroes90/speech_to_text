from adapters.openai_adapters.openai_environment_loader import OpenaiEnvironmentHandler
from handlers_and_protocols.protocols import (
    TranscribeServiceHandler,
    EnvironmentHandler,
)
from openai import OpenAI
import re
from utils.audio_utils import batch_audio, load_audio_from_bytesio
import io
import utils.utils as utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import time


class WhisperServiceHandler(TranscribeServiceHandler):
    def __init__(
        self,
        environment_handler: EnvironmentHandler = None,
        env_loaded: bool = False,
        server_region: str = None,
        max_workers: int = 8,
    ):
        if not env_loaded:
            if environment_handler is None:
                self.environment_handler = OpenaiEnvironmentHandler()
            else:
                self.environment_handler = environment_handler
            self.environment_handler.load_environment()
        self.client = OpenAI()
        self.max_workers = max_workers

    def _transcribe_single_chunk(
        self, chunk_data, source_language=None, translate=False
    ):
        """
        Transcribe a single audio chunk with retry logic for rate limits.

        Args:
            chunk_data: tuple of (audio_segment, segment_time, index)
            source_language: Language code for transcription
            translate: Whether to translate (True) or just transcribe (False)

        Returns:
            tuple: (index, adjusted_transcript) for proper ordering
        """
        audio_segment, segment_time, index = chunk_data
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                print(f"Processing chunk {index + 1} (attempt {attempt + 1})")

                if translate:
                    transcript = self.client.audio.translations.create(
                        file=audio_segment,
                        model="whisper-1",
                        response_format="srt",
                    )
                else:
                    transcript = self.client.audio.transcriptions.create(
                        file=audio_segment,
                        language=source_language,
                        model="whisper-1",
                        response_format="srt",
                    )

                # Adjust timestamps based on segment start time
                adjusted_transcript = adjust_srt_timestamps(transcript, segment_time[0])
                return (index, adjusted_transcript)

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + (time.time() % 1)
                    print(
                        f"Rate limit hit for chunk {index + 1}, retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Failed to process chunk {index + 1} after {max_retries} attempts"
                    )
                    raise e
            except Exception as e:
                print(f"Error processing chunk {index + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    raise e

    def _process_chunks_parallel(
        self, audio_data, segment_times, source_language=None, translate=False
    ):
        """
        Process audio chunks in parallel with proper ordering.

        Returns:
            str: Complete SRT content with proper ordering
        """
        # Prepare chunk data with indices for proper ordering
        chunk_data = [
            (audio_segment, segment_time, index)
            for index, (audio_segment, segment_time) in enumerate(
                zip(audio_data, segment_times)
            )
        ]

        # Determine optimal number of workers
        actual_workers = min(
            self.max_workers, len(chunk_data), 10
        )  # Cap at 10 for safety

        print(f"Processing {len(chunk_data)} chunks with {actual_workers} workers")

        # Store results with their original indices
        results = [None] * len(chunk_data)

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all chunks
            future_to_index = {
                executor.submit(
                    self._transcribe_single_chunk, chunk, source_language, translate
                ): chunk[
                    2
                ]  # chunk[2] is the index
                for chunk in chunk_data
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                try:
                    index, transcript = future.result()
                    results[index] = transcript
                    completed += 1
                    print(
                        f"Completed chunk {index + 1} ({completed}/{len(chunk_data)})"
                    )
                except Exception as e:
                    original_index = future_to_index[future]
                    print(f"Failed to process chunk {original_index + 1}: {str(e)}")
                    # You might want to handle this differently - perhaps retry or skip
                    results[original_index] = ""  # Empty string as placeholder

        # Combine results in correct order
        srt_segments = [
            result + "\n" if result and not result.endswith("\n\n") else result
            for result in results
            if result is not None
        ]

        return "".join(srt_segments)

    def transcribe_audio(
        self,
        input_audio_data_io: io.BytesIO,
        audio_path: str = None,
        source_language: str = None,
        **kwargs,
    ) -> any:
        if input_audio_data_io is None or source_language is None:
            raise ValueError(
                "Audio path, and language is required for Whisper transcription."
            )

        input_audio_data_io.seek(0)
        input_audio_data = load_audio_from_bytesio(input_audio_data_io)
        audio_data, segment_times = batch_audio(input_audio_data, max_size_mb=24)

        file_name = input_audio_data_io.name

        print(f"Starting parallel transcription of {len(audio_data)} chunks")
        complete_srt = self._process_chunks_parallel(
            audio_data, segment_times, source_language, translate=False
        )

        print("OpenAI transcription complete!")
        utils.save_object_to_pickle(
            complete_srt,
            file_path=f"/Users/ramiibrahimi/Documents/test.nosync/pkl/{file_name}_whisper.pkl",
        )

        return complete_srt

    def transcribe_translate(
        self,
        input_audio_data_io: io.BytesIO,
        audio_path: str = None,
        **kwargs,
    ) -> any:
        if input_audio_data_io is None:
            raise ValueError(
                "Audio path, and language is required for Whisper transcription."
            )

        input_audio_data_io.seek(0)
        input_audio_data = load_audio_from_bytesio(input_audio_data_io)
        audio_data, segment_times = batch_audio(input_audio_data, max_size_mb=24)

        file_name = input_audio_data_io.name

        print(
            f"Starting parallel transcription+translation of {len(audio_data)} chunks"
        )
        complete_srt = self._process_chunks_parallel(
            audio_data, segment_times, translate=True
        )

        # Renumber SRT indices for consistency
        complete_srt_reindex = renumber_srt_indices(complete_srt)

        print("OpenAI transcription+translation complete!")
        utils.save_object_to_pickle(
            complete_srt_reindex,
            file_path=f"/Users/ramiibrahimi/Documents/test.nosync/pkl/{file_name}_whisper.pkl",
        )

        return complete_srt_reindex  # in srt format


def adjust_srt_timestamps(srt_data: str, start_time: int) -> str:
    # Helper function to convert SRT time format to milliseconds
    def srt_time_to_ms(srt_time: int) -> int:
        h, m, s, ms = map(int, re.split("[:,]", srt_time))
        return (h * 3600 + m * 60 + s) * 1000 + ms

    # Helper function to convert milliseconds to SRT time format
    def ms_to_srt_time(ms):
        hours, remainder = divmod(ms, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    adjusted_srt = []
    entries = srt_data.strip().split("\n\n")
    for entry in entries:
        lines = entry.split("\n")
        index, time_range, *text = lines
        start, end = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3})", time_range)
        start_ms = srt_time_to_ms(start) + start_time
        end_ms = srt_time_to_ms(end) + start_time
        adjusted_start = ms_to_srt_time(start_ms)
        adjusted_end = ms_to_srt_time(end_ms)
        adjusted_entry = (
            f"{index}\n{adjusted_start} --> {adjusted_end}\n{'\n'.join(text)}"
        )
        adjusted_srt.append(adjusted_entry)

    return "\n\n".join(adjusted_srt)


def renumber_srt_indices(srt_data: str) -> str:
    """
    Adjusts the indices of SRT subtitle entries so they are sequential from 1 onwards.

    :param srt_data: The SRT data as a string, where indices may be out of order.
    :return: A string of the SRT data with corrected indices.
    """
    adjusted_srt = []
    entries = srt_data.strip().split("\n\n")
    new_index = 1
    for entry in entries:
        lines = entry.split("\n")
        time_range, *text = lines[1:]  # Skip the original index
        adjusted_entry = f"{new_index}\n{time_range}\n{'\n'.join(text)}"
        adjusted_srt.append(adjusted_entry)
        new_index += 1  # Increment the index for each subtitle

    return "\n\n".join(adjusted_srt)
