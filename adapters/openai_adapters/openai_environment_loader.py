from handlers_and_protocols.protocols import EnvironmentHandler

import os
from dotenv import load_dotenv


class OpenaiEnvironmentHandler(EnvironmentHandler):
    def load_environment(self, env_var_name: str = "OPENAI_API_KEY") -> None:
        load_dotenv()
        api_key = os.getenv(env_var_name)
        if api_key is None:
            raise ValueError(f"Environment variable {env_var_name} is not set.")
        print(f"API key loaded successfully for environment variable: {env_var_name}")
