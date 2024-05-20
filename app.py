import utils
import handlers
import audio_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import gc


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
    srt_response = transcribe_and_translate(
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


def transcribe_and_translate(
    audio_data: io.BytesIO,
    service_name: str,
    source_language: str,
    target_language: str,
    env_loaded: bool = False,
    server_region: str = "us-central1",  # for google transcribe chirp
):

    audio_data.seek(0)
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
