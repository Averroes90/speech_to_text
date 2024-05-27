from google.cloud import translate_v2 as translate
from handlers_and_protocols.protocols import TranslationServiceHandler
from google_adapters.google_environment_loader import (
    EnvironmentHandler,
    GoogleEnvironmentHandler,
)


class GoogleTranslateServiceHandler(TranslationServiceHandler):
    def __init__(
        self, environment_handler: EnvironmentHandler = None, env_loaded: bool = False
    ):
        if not env_loaded:
            if environment_handler is None:
                environment_handler = GoogleEnvironmentHandler()
            environment_handler.load_environment()

        self.translate_client = translate.Client()

    def translate_text(
        self, text: str, source_language: str = "ru", target_language: str = "en"
    ) -> str:
        """Translate text to the target language."""
        result = self.translate_client.translate(
            text, source_language=source_language, target_language=target_language
        )
        return result["translatedText"]
