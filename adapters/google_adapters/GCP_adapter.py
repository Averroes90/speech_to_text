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
import threading
from google.api_core import retry


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

        # Create a new storage client for each instance to avoid sharing connections
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Thread lock for critical operations (optional, but safer)
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()  # Serialize uploads

    def upload_audio_file(
        self, input_audio_data_io: io.BytesIO, sample_rate: int = None
    ) -> str:
        with self._upload_lock:  #
            if not hasattr(input_audio_data_io, "name") or not input_audio_data_io.name:
                raise ValueError("input_audio_data_io must have a 'name' attribute")

            input_audio_data_io.seek(0)
            file_name = input_audio_data_io.name
            print(f"cloud filename {file_name}")

            # Create a new blob reference for this specific file
            blob = self.bucket.blob(file_name)

            try:
                # Check if the file is already in the bucket
                if blob.exists():
                    print(
                        f'{file_name} already exists in "{self.bucket_name}" bucket. Skipping upload.'
                    )
                    return f"File {file_name} already exists."

                print(f"uploading file {file_name}")

                # Get the sample rate - handle the case where it might not exist
                try:
                    if sample_rate is None:
                        if hasattr(input_audio_data_io, "sample_rate"):
                            sample_rate = input_audio_data_io.sample_rate
                        else:
                            # Fallback or default sample rate if not available
                            sample_rate = 44100  # or whatever default makes sense
                            print(
                                f"Warning: No sample rate found, using default: {sample_rate}"
                            )
                except AttributeError:
                    sample_rate = 44100
                    print(
                        f"Warning: Could not get sample rate, using default: {sample_rate}"
                    )

                # Set metadata before upload
                blob.metadata = {"sample_rate": str(sample_rate)}

                # Ensure we're at the beginning and upload
                input_audio_data_io.seek(0)

                # Read the data into a new BytesIO to avoid pointer issues
                audio_data = input_audio_data_io.read()
                input_audio_data_io.seek(0)  # Reset original pointer

                # Upload from the copied data
                upload_buffer = io.BytesIO(audio_data)
                blob.upload_from_file(upload_buffer)
                blob.patch()

                print(
                    f'Uploaded {file_name} to "{self.bucket_name}" bucket with sample rate: {sample_rate}.'
                )
                return f"Successfully uploaded {file_name}"

            except Exception as e:
                print(f"Error uploading {file_name}: {e}")
                raise e

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
            # Create a new blob reference for this specific file
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

    def __del__(self):
        """Clean up resources when the handler is destroyed"""
        try:
            if hasattr(self, "storage_client"):
                self.storage_client.close()
        except:
            pass  # Ignore cleanup errors
