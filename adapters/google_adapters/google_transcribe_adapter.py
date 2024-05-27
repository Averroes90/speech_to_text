from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import os
import io

from openai import audio
import utils.utils as utils
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from handlers_and_protocols.protocols import (
    TranscribeServiceHandler,
    EnvironmentHandler,
)
from google_adapters.google_environment_loader import GoogleEnvironmentHandler
from utils.nlp_utils2 import process_chirp_responses
from google_adapters.GCP_adapter import GoogleCloudHandler
import concurrent.futures


class GoogleTranscribeModelHandler(TranscribeServiceHandler):
    def __init__(
        self,
        server_region: str = "us-central1",
        environment_handler: EnvironmentHandler = None,
        env_loaded: bool = False,
    ):
        if not env_loaded:
            if environment_handler is None:
                environment_handler = GoogleEnvironmentHandler()
            environment_handler.load_environment()
            env_loaded = True
        load_dotenv()
        project_env = "PROJECT_ID"
        project_id = os.getenv(project_env)
        self.project_id = project_id
        self.server_region = server_region
        config_path = "resources/language_map.json"
        self.config = utils.load_config(config_path)
        self.cloud_handler = GoogleCloudHandler(env_loaded=env_loaded)
        if self.server_region != "global":
            self.speech_client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint=f"{server_region}-speech.googleapis.com",  # needed for chirp
                )
            )
        else:
            self.speech_client = SpeechClient()

    def transcribe_audio(
        self,
        input_audio_data_io: io.BytesIO = None,
        model: str = None,
        srt: bool = False,
        language: str = "ru",
        **kwargs,
    ) -> any:
        # for debuging
        # file_uri = kwargs.get("file_path", None)
        # if file_uri is None or model is None:
        #     raise ValueError(
        #         "Storage URI and output file path are required for Google transcription."
        #     )
        # Get the current working directory

        # if kwargs.get("internal_call", False) == False:
        #     self.cloud_handler.upload_audio_file(
        #         input_audio_data_io=input_audio_data_io,
        #     )

        if not input_audio_data_io:  # for debugging
            file_name = kwargs.get("file_name", "default_value")
        else:
            input_audio_data_io.seek(0)
            file_name = input_audio_data_io.name

        language_code = self.config.get(language)
        # self.cloud_handler.upload_audio_file(
        #     input_audio_data_io,
        # )
        bucket_name = self.cloud_handler.bucket_name

        storage_uri = f"gs://{bucket_name}/{file_name}"
        # Configuring recognition features and settings
        feature_config = {
            "enable_automatic_punctuation": True,
            "profanity_filter": False,
            # "max_alternatives": 2,
        }
        # Conditionally add "enable_word_time_offsets" if model is not "chirp_2"
        if model != "chirp_2" or srt is True:
            feature_config["enable_word_time_offsets"] = True

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[language_code],
            model=model,
            features=cloud_speech.RecognitionFeatures(**feature_config),
        )

        file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=storage_uri)

        recognition_output_config = cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        )

        if srt:
            recognition_output_config.output_format_config = (
                cloud_speech.OutputFormatConfig(srt={})
            )

        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{self.server_region}/recognizers/_",
            config=config,
            files=[file_metadata],
            recognition_output_config=recognition_output_config,
        )
        # Transcribes the audio into text
        operation = self.speech_client.batch_recognize(request=request)

        print(f"Waiting for operation to complete...model={model}")
        transcription_response = operation.result(timeout=900)
        # self.cloud_handler.delete_audio_file(file_name=file_name)
        print(f"transcription complete!...model={model}")
        utils.save_object_to_pickle(
            transcription_response,
            file_path=f"/Users/ramiibrahimi/Documents/test/pkl/{file_name}_{model}.pkl",
        )

        return transcription_response

    def transcribe_translate(
        self,
        input_audio_data_io: io.BytesIO,
        source_language: str,
        target_language: str,
        model: str = None,
        srt: bool = False,
        **kwargs,
    ) -> any:

        self.cloud_handler.upload_audio_file(
            input_audio_data_io,
        )
        audio_duration = input_audio_data_io.audio_duration
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future1 = executor.submit(
                self.transcribe_audio,
                input_audio_data_io=input_audio_data_io,
                model="chirp",
                language=source_language,
                srt=srt,
                internal_call=True,
            )
            future2 = executor.submit(
                self.transcribe_audio,
                input_audio_data_io=input_audio_data_io,
                model="chirp_2",
                language=source_language,
                srt=srt,
                internal_call=True,
            )
            # executor.submit()
            transcription_response1 = future1.result()
            transcription_response2 = future2.result()

        srt_1, srt_2 = process_chirp_responses(
            chirp_response=transcription_response1,
            chirp_2_response=transcription_response2,
            source_language=source_language,
            audio_duration=audio_duration,
        )
        # srt_chirp1__en = utils.translate_srt(srt_1,
        #     service="google",
        #     source_language=source_language,
        #     target_language=target_language,
        # )
        srt_chirp2__en = utils.translate_srt(
            srt_2,
            service="google",
            source_language=source_language,
            target_language=target_language,
            env_loaded=True,
        )
        return srt_chirp2__en


def print_goog_transcription_details(transcription_response: any) -> None:
    """
    Print detailed information about the transcription response including
    transcripts, confidence levels, and timing information.
    """
    key = next(iter(transcription_response.results))

    clean_transcription_response = transcription_response.results[
        key
    ].inline_result.transcript
    if not clean_transcription_response.results:
        print("No results in the transcription response.")
        return

    for i, result in enumerate(clean_transcription_response.results):
        print(f"Result {i + 1}:")
        for j, alternative in enumerate(result.alternatives):
            print(f"  Alternative {j + 1}:")
            print(f"    Transcript: {alternative.transcript}")
            print(f"    Confidence: {alternative.confidence}")
            print(f"total words {len(alternative.transcript.split())}")

            if alternative.words:
                print("    Words:")
                print(f"total timings {len(alternative.words)}")
                for word_info in alternative.words:
                    start_time = (
                        word_info.start_offset.total_seconds()
                        if word_info.start_offset
                        else "N/A"
                    )
                    end_time = (
                        word_info.end_offset.total_seconds()
                        if word_info.end_offset
                        else "N/A"
                    )
                    print(
                        f"      Word: {word_info.word}, Start Time: {start_time}, End Time: {end_time}"
                    )
            else:
                print("    No word timing information available.")
