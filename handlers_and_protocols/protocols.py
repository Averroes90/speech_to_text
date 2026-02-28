from abc import ABC, abstractmethod
import io


class TranslationServiceHandler(ABC):
    @abstractmethod
    def translate_text(self, text: str, target_language: str = "en") -> str:
        """
        Translate text into the target language.
        """
        pass


class TranscribeServiceHandler(ABC):
    @abstractmethod
    def transcribe_audio(
        self,
        file_uri: str = None,
        model: str = None,
        srt: bool = None,
        language_code: str = "it-IT",
    ) -> any:
        """
        Transcribes audio from a given storage URI and saves the transcription to a specified file.
        The default language for transcription is Italian ('it-IT').
        """
        pass
        ...

    @abstractmethod
    def transcribe_translate(
        self,
        input_audio_data: io.BytesIO,
        audio_path: str = None,
        **kwargs,
    ) -> str:
        pass
        ...


class CloudServiceHandler(ABC):
    @abstractmethod
    def upload_audio_file(self, audio_file: io.BytesIO) -> str:
        """
        Uploads an audio file to a specified bucket and returns the URI of the uploaded file.
        Gets Sample Rate from the file and adds it as metadata
        """
        ...

    @abstractmethod
    def delete_audio_file(self, file_name: str) -> tuple[bool, str]:
        """
        Delete an audio file from the storage bucket.

        Parameters:
        file_name (str): The name of the file to delete.

        Returns:
        tuple[bool, str]: A tuple containing a boolean indicating success or failure,
                           and a string message describing the outcome.
        """
        ...


class EnvironmentHandler(ABC):
    @abstractmethod
    def load_environment(self, credentials_env_var: str = ""):
        pass
