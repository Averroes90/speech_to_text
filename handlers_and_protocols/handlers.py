from adapters.google_adapters.google_transcribe_adapter import (
    GoogleTranscribeModelHandler,
)
from adapters.google_adapters.GCP_adapter import GoogleCloudHandler
from adapters.google_adapters.google_translate_adapter import (
    GoogleTranslateServiceHandler,
)
from adapters.google_adapters.google_environment_loader import GoogleEnvironmentHandler
from adapters.openai_adapters.openai_transcribe_adapter import WhisperServiceHandler
from adapters.openai_adapters.openai_environment_loader import OpenaiEnvironmentHandler
from handlers_and_protocols.protocols import (
    TranscribeServiceHandler,
    TranslationServiceHandler,
    EnvironmentHandler,
    CloudServiceHandler,
)


def get_transcribe_service_handler(
    service: str, env_loaded: bool = False, server_region: str = "us-central1"
) -> TranscribeServiceHandler:
    if service == "google":
        return GoogleTranscribeModelHandler(
            env_loaded=env_loaded, server_region=server_region
        )
    if service == "openai":
        return WhisperServiceHandler(env_loaded=env_loaded)

    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")


def get_cloud_service_handler(
    service: str, env_loaded: bool = False
) -> CloudServiceHandler:
    if service == "google":
        return GoogleCloudHandler(env_loaded=env_loaded)
    else:
        # Add other conditions for different handlers
        raise ValueError(f"Unsupported service: {service}")


def get_translation_service_handler(
    service: str, env_loaded: bool = False
) -> TranslationServiceHandler:
    if service == "google":
        return GoogleTranslateServiceHandler(env_loaded=env_loaded)
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
