from typing import Protocol, runtime_checkable
from google.cloud import storage
import os
import io
import utils.utils as utils
from dotenv import load_dotenv
from handlers_and_protocols.protocols import CloudServiceHandler
from adapters.google_adapters.google_environment_loader import (
    EnvironmentHandler,
    GoogleEnvironmentHandler,
)


class GoogleCloudHandler(CloudServiceHandler):
    def __init__(
        self, environment_handler: EnvironmentHandler = None, env_loaded: bool = False
    ):
        if not env_loaded:
            if environment_handler is None:
                environment_handler = GoogleEnvironmentHandler()
            environment_handler.load_environment()
        load_dotenv()
        bucket_env = "BUCKET_NAME"
        bucket_name = os.getenv(bucket_env)
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def upload_audio_file(
        self, input_audio_data_io: io.BytesIO, sample_rate: int = None
    ) -> str:
        # for debugging
        # file_name = os.path.basename(file_path)
        input_audio_data_io.seek(0)
        file_name = input_audio_data_io.name
        print(f"cloud filename {file_name}")
        blob = self.bucket.blob(file_name)

        # Check if the file is already in the bucket
        if blob.exists():
            print(
                f'{file_name} already exists in "{self.bucket_name}" bucket. Skipping upload.'
            )
            return f"File {file_name} already exists."
        print(f"uploading file {file_name}")

        # Get the sample rate from the local file
        sample_rate = input_audio_data_io.sample_rate
        # sample_rate = utils.get_sample_rate_bytesio(input_audio_data_io)

        # Upload the file and set metadata
        blob.metadata = {"sample_rate": str(sample_rate)}
        input_audio_data_io.seek(
            0
        )  # Ensure the pointer is at the beginning of the BytesIO object
        blob.upload_from_file(input_audio_data_io)
        blob.patch()

        print(
            f'Uploaded {file_name} to "{self.bucket_name}" bucket with sample rate: {sample_rate}.'
        )
        return

    def delete_audio_file(self, file_name: str) -> tuple[bool, str]:
        """
        Delete an audio file from the storage bucket.

        Parameters:
        file_name (str): The name of the file to delete.

        Returns:
        tuple[bool, str]: A tuple containing a boolean indicating success or failure,
                           and a string message describing the outcome.
        """
        try:
            blob = self.bucket.blob(file_name)

            # Check if the file exists before attempting to delete it
            if not blob.exists():
                return False, f"File {file_name} does not exist in the bucket."

            # Delete the file
            blob.delete()
            return (
                True,
                f"File {file_name} successfully deleted from '{self.bucket_name}'.",
            )

        except Exception as e:
            return False, f"Failed to delete file: {str(e)}"
