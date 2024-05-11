from google_adapters.google_transcribe_adapter import (
    GoogleTranscribeModelHandler,
)
from google_adapters.GCP_adapter import GoogleCloudHandler
from google_adapters.google_translate_adapter import (
    GoogleTranslateServiceHandler,
)
from google_adapters.google_environment_loader import GoogleEnvironmentHandler
from openai_adapters.openai_transcribe_adapter import WhisperServiceHandler
from openai_adapters.openai_environment_loader import OpenaiEnvironmentHandler
from protocols.protocols import (
    TranscribeServiceHandler,
    TranslationServiceHandler,
    EnvironmentHandler,
    CloudServiceHandler,
)


def get_transcribe_service_handler(
    service: str,
) -> TranscribeServiceHandler:
    if service == "google":
        return GoogleTranscribeModelHandler()
    if service == "openai":
        return WhisperServiceHandler()

    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")


def get_cloud_service_handler(
    service: str,
) -> CloudServiceHandler:
    if service == "google":
        return GoogleCloudHandler()
    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")


def get_translation_service_handler(service: str) -> TranslationServiceHandler:
    if service == "google":
        return GoogleTranslateServiceHandler()
    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")


def get_environmet_handler(service: str) -> EnvironmentHandler:
    if service == "google":
        return GoogleEnvironmentHandler()
    if service == "openai":
        return OpenaiEnvironmentHandler()
    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")
