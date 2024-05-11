import utils
import os
import handlers
import audio_utils


# app.py
def main():
    print("Hello, World!")


def transcribe_translate(
    file_path: str, service_name: str, source_language: str, target_language: str
) -> str:

    path, filename, extension = utils.extract_path_details(full_path=file_path)
    print("extracting audio from video file")
    success, message, audio_data = audio_utils.extract_audio(
        input_path=path, input_filename=filename, input_extension=extension
    )
    tc_tr_handler = handlers.get_transcribe_service_handler(service=service_name)
    print("starting automatic transcription")
    srt_response = tc_tr_handler.transcribe_translate(
        input_audio_data_io=audio_data,
        source_language=source_language,
        target_language=target_language,
    )
    return srt_response


if __name__ == "__main__":
    main()
