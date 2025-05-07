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


class WhisperServiceHandler(TranscribeServiceHandler):
    def __init__(
        self,
        environment_handler: EnvironmentHandler = None,
        env_loaded: bool = False,
        server_region: str = None,
    ):
        if not env_loaded:
            if environment_handler is None:
                self.environment_handler = OpenaiEnvironmentHandler()
            else:
                self.environment_handler = environment_handler
            self.environment_handler.load_environment()
        self.client = OpenAI()

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
        # Assuming 'audio_data' is preprocessed and ready for API call
        input_audio_data = load_audio_from_bytesio(input_audio_data_io)
        audio_data, segment_times = batch_audio(input_audio_data, max_size_mb=24)
        # print(f"len data {len(audio_data)}")
        # return
        file_name = input_audio_data_io.name
        transcription_response = []
        for index, (audio_segment, segment_time) in enumerate(
            zip(audio_data, segment_times)
        ):
            print(f"openai transcribing batch {index+1} of {len(audio_data)}")
            transcript = self.client.audio.transcriptions.create(
                file=audio_segment,
                language=source_language,
                model="whisper-1",
                response_format="srt",
                # timestamp_granularities=["word", "segment"],
            )
            adjusted_transcript = adjust_srt_timestamps(transcript, segment_time[0])
            transcription_response.append(adjusted_transcript)
            srt_segments = [
                s + "\n" if not s.endswith("\n\n") else s
                for s in transcription_response
            ]
        complete_srt = "".join(srt_segments)
        print("openai transcription complete!")
        utils.save_object_to_pickle(
            complete_srt,
            file_path=f"/Users/ramiibrahimi/Documents/test.nosync/pkl/{file_name}_whisper.pkl",
        )
        return complete_srt  # in srt format

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
        # Assuming 'audio_data' is preprocessed and ready for API call
        audio_data, segment_times = batch_audio(input_audio_data, max_size_mb=24)
        # print(f"len data {len(audio_data)}")
        # return
        file_name = input_audio_data_io.name
        transcription_response = []
        for index, (audio_segment, segment_time) in enumerate(
            zip(audio_data, segment_times)
        ):
            print(
                f"openai transcribing and translating batch {index+1} of {len(audio_data)}"
            )
            transcript = self.client.audio.translations.create(
                file=audio_segment,
                model="whisper-1",
                response_format="srt",
                # timestamp_granularities=["word", "segment"],
            )
            # Adjust timestamps in transcript based on start_time
            adjusted_transcript = adjust_srt_timestamps(transcript, segment_time[0])
            transcription_response.append(adjusted_transcript)
            srt_segments = [
                s + "\n" if not s.endswith("\n\n") else s
                for s in transcription_response
            ]
        complete_srt = "".join(srt_segments)
        complete_srt_reindex = renumber_srt_indices(complete_srt)
        print("openai transcription complete!")
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
