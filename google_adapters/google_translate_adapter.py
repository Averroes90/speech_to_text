from google.cloud import translate_v2 as translate
import os
from protocols.protocols import TranslationServiceHandler
from google_adapters.google_environment_loader import (
    EnvironmentHandler,
    GoogleEnvironmentHandler,
)


class GoogleTranslateServiceHandler(TranslationServiceHandler):
    def __init__(
        self,
        environment_handler: EnvironmentHandler = None,
    ):
        if environment_handler is None:
            environment_handler = GoogleEnvironmentHandler()
        environment_handler.load_environment()

        self.translate_client = translate.Client()

    def translate_text(
        self, text: str, source_language: str = "it", target_language: str = "en"
    ) -> str:
        """Translate text to the target language."""
        result = self.translate_client.translate(
            text, source_language=source_language, target_language=target_language
        )
        return result["translatedText"]
