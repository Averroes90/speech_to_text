import utils.utils as utils
import handlers_and_protocols.handlers as handlers
import utils.audio_utils as audio_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import gc
from utils.nlp_utils2 import process_chirp_responses
from adapters.google_adapters.GCP_adapter import GoogleCloudHandler
from adapters.google_adapters.google_transcribe_adapter import (
    GoogleTranscribeModelHandler,
)
import logging
import os
import sys

# Read environment variable
# debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
debug_mode = False
# Configure logging based on the environment variable
logging.basicConfig(
    level=logging.ERROR if debug_mode else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Direct logs to standard output
)


# app.py
def main():
    print("Hello, World!")


def transcribe_translate(
    file_path: str,
    service_name: str,
    source_language: str,
    target_language: str,
    audio_output_extension: str = ".wav",
    server_region: str = "us-central1",  # for google transcribe chirp
) -> str:
    env_handler = handlers.get_environmet_handler(service=service_name)
    env_handler.load_environment()
    path, filename, extension = utils.extract_path_details(full_path=file_path)
    print("extracting audio from video file")
    success, message, audio_data = audio_utils.extract_audio(
        input_path=path,
        input_filename=filename,
        input_extension=extension,
        output_extension=audio_output_extension,
    )
    audio_data.seek(0)
    srt_response = transcribe_and_translate_combine(
        audio_data=audio_data,
        service_name=service_name,
        source_language=source_language,
        target_language=target_language,
        env_loaded=True,
        server_region=server_region,
    )
    del audio_data
    gc.collect()
    return srt_response


def transcribe(
    file_path: str, service_name: str, source_language: str, target_language: str
) -> str:
    env_handler = handlers.get_environmet_handler(service=service_name)
    env_handler.load_environment()

    path, filename, extension = utils.extract_path_details(full_path=file_path)
    print("extracting audio from video file")
    success, message, audio_data = audio_utils.extract_audio(
        input_path=path, input_filename=filename, input_extension=extension
    )
    audio_data.seek(0)
    tc_tr_handler = handlers.get_transcribe_service_handler(
        service=service_name, env_loaded=True
    )
    print("starting automatic transcription")
    srt_response = tc_tr_handler.transcribe_audio(
        input_audio_data_io=audio_data,
        source_language=source_language,
        target_language=target_language,
    )
    return srt_response


def transcribe_and_translate_combine(
    audio_data: io.BytesIO,
    service_name: str,
    source_language: str,
    target_language: str,
    env_loaded: bool = False,
    server_region: str = "us-central1",  # for google transcribe chirp
):
    audio_data.seek(0)
    if service_name == "google" and server_region:
        chirp_handler = handlers.get_transcribe_service_handler(
            service=service_name, env_loaded=env_loaded, server_region=server_region
        )
        chirp2_handler = handlers.get_transcribe_service_handler(
            service=service_name, env_loaded=env_loaded, server_region="us-central1"
        )

        cloud_handler = GoogleCloudHandler(env_loaded=True)
        cloud_handler.upload_audio_file(input_audio_data_io=audio_data)
        audio_duration = audio_data.audio_duration

        with ThreadPoolExecutor(max_workers=5) as executor:
            future1 = executor.submit(
                chirp_handler.transcribe_audio,
                input_audio_data_io=audio_data,
                model="chirp",
                source_language=source_language,
                srt=False,
                internal_call=True,
            )
            future2 = executor.submit(
                chirp2_handler.transcribe_audio,
                input_audio_data_io=audio_data,
                model="chirp_2",
                source_language=source_language,
                srt=False,
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
        srt_response = utils.translate_srt(
            srt_2,
            service="google",
            source_language=source_language,
            target_language=target_language,
            env_loaded=True,
        )

    else:

        tc_tr_handler = handlers.get_transcribe_service_handler(
            service=service_name, env_loaded=env_loaded, server_region=server_region
        )
        srt_response = tc_tr_handler.transcribe_translate(
            input_audio_data_io=audio_data,
            source_language=source_language,
            target_language=target_language,
        )

    return srt_response


def transcribe_and_translate(
    audio_data: io.BytesIO,
    service_name: str,
    source_language: str,
    target_language: str,
    env_loaded: bool = False,
    server_region: str = "us-central1",  # for google transcribe chirp
):
    audio_data.seek(0)
    if service_name == "google" and server_region:
        chirp2_handler = handlers.get_transcribe_service_handler(
            service=service_name, env_loaded=env_loaded, server_region="us-central1"
        )

        cloud_handler = GoogleCloudHandler(env_loaded=True)
        cloud_handler.upload_audio_file(input_audio_data_io=audio_data)
        audio_data.seek(0)
        transcription_response = chirp2_handler.transcribe_audio(
            input_audio_data_io=audio_data,
            model="chirp_2",
            source_language=source_language,
            srt=True,
            internal_call=True,
        )
        key = next(iter(transcription_response.results))
        srt = transcription_response.results[key].inline_result.srt_captions
        srt_response = utils.translate_srt(
            srt,
            service="google",
            source_language=source_language,
            target_language=target_language,
            env_loaded=True,
        )

    else:

        tc_tr_handler = handlers.get_transcribe_service_handler(
            service=service_name, env_loaded=env_loaded, server_region=server_region
        )
        srt_response = tc_tr_handler.transcribe_translate(
            input_audio_data_io=audio_data,
            source_language=source_language,
            target_language=target_language,
        )

    return srt_response


def multi_transcribe(
    file_path: str,
    service_names: list[str],
    source_language: str,
    target_language: str,
    audio_output_extension: str = ".wav",
    server_region: str = "us-central1",  # for google transcribe chirp
) -> str:
    srt_responses = {}
    for service_name in service_names:
        env_handler = handlers.get_environmet_handler(service=service_name)
        env_handler.load_environment()
    path, filename, extension = utils.extract_path_details(full_path=file_path)
    print("extracting audio from video file")
    success, message, audio_data = audio_utils.extract_audio(
        input_path=path,
        input_filename=filename,
        input_extension=extension,
        output_extension=audio_output_extension,
    )
    audio_data.seek(0)
    print("starting automatic transcriptions")
    with ThreadPoolExecutor() as executor:
        future_to_service = {
            executor.submit(
                transcribe_and_translate,
                service_name=service,
                audio_data=audio_data,
                source_language=source_language,
                target_language=target_language,
                env_loaded=True,
                server_region=server_region,
            ): service
            for service in service_names
        }
        for future in as_completed(future_to_service):
            service = future_to_service[future]
            try:
                srt_response = future.result()
                srt_responses[service] = srt_response
            except Exception as exc:
                srt_responses[service] = f"{service} generated an exception: {exc}"
    del audio_data
    gc.collect()
    return srt_responses


if __name__ == "__main__":
    main()
