from typing import Protocol, runtime_checkable
from google.cloud import storage, speech
import os
import utils
from dotenv import load_dotenv


@runtime_checkable
class TranscribeServiceHandler(Protocol):
    def transcribe_audio(
        self, storage_uri: str, output_file_path: str, language_code: str = "it-IT"
    ) -> any:
        """
        Transcribes audio from a given storage URI and saves the transcription to a specified file.
        The default language for transcription is Italian ('it-IT').
        """
        ...

    def upload_audio_file(
        self, bucket_name: str, source_folder: str, file_extension: str
    ) -> str:
        """
        Uploads an audio file to a specified bucket and returns the URI of the uploaded file.
        Gets Sample Rate from the file and adds it as metadata
        """
        ...

    def save_transcription_response(self, output_file_path: str, response: any) -> None:
        """
        Saves the transcription response to a file at the given path.
        """
        ...


class GoogleTranscribeModelHandler(TranscribeServiceHandler):
    def __init__(
        self,
        bucket_name: str,
        language_code: str = "it-IT",
        credentials_env_var: str = "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        self.load_environment(credentials_env_var)
        self.bucket_name = bucket_name
        self.language_code = language_code
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.speech_client = speech.SpeechClient()

    @staticmethod
    def load_environment(credentials_env_var: str):
        load_dotenv()
        cred_file_name = os.getenv(credentials_env_var)
        if cred_file_name is None:
            raise ValueError(f"Environment variable {credentials_env_var} is not set.")
        credentials_path = os.path.join(os.getcwd(), cred_file_name)
        os.environ[credentials_env_var] = credentials_path
        print(f"Credentials set for {credentials_env_var}.")

    def upload_audio_file(self, file_path: str) -> str:
        file_name = os.path.basename(file_path)
        blob = self.bucket.blob(file_name)

        # Check if the file is already in the bucket
        if blob.exists():
            print(
                f'{file_name} already exists in "{self.bucket_name}" bucket. Skipping upload.'
            )
            return f"File {file_name} already exists."

        # Get the sample rate from the local file
        sample_rate = utils.get_sample_rate(file_path)

        # Upload the file and set metadata
        blob.metadata = {"sample_rate": str(sample_rate)}
        blob.upload_from_filename(file_path)
        blob.patch()

        print(
            f'Uploaded {file_name} to "{self.bucket_name}" bucket with sample rate: {sample_rate}.'
        )
        return "Upload complete."

    def transcribe_audio(self, storage_uri: str, model: str) -> any:
        audio = speech.RecognitionAudio(uri=storage_uri)
        _, file_name = storage_uri[5:].split("/", 1)
        blob = self.bucket.blob(file_name)
        blob.reload()
        sample_rate = (
            int(blob.metadata["sample_rate"])
            if blob.metadata and "sample_rate" in blob.metadata
            else 16000
        )
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            profanity_filter=False,
            max_alternatives=2,
        )
        operation = self.speech_client.long_running_recognize(
            config=config, audio=audio
        )
        print("Waiting for operation to complete...")
        response = operation.result(timeout=900)
        return response

    def save_transcription_response(self, output_file_path: str, response: any) -> None:
        with open(output_file_path, "w") as file:
            for result in response.results:
                for i, alternative in enumerate(result.alternatives):
                    if i < 2:
                        file.write(f"Transcription {i+1}: {alternative.transcript}\n")


def get_transcribe_service_handler(
    service: str, bucket_name: str, language_code: str = "it-IT"
) -> TranscribeServiceHandler:
    if service == "google":
        return GoogleTranscribeModelHandler(
            bucket_name=bucket_name, language_code=language_code
        )
    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")
