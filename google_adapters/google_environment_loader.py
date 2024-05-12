from protocols.protocols import EnvironmentHandler

import os
from dotenv import load_dotenv


class GoogleEnvironmentHandler(EnvironmentHandler):
    def load_environment(
        self, credentials_env_var: str = "GOOGLE_APPLICATION_CREDENTIALS"
    ) -> None:
        load_dotenv()
        cred_file_name = os.getenv(credentials_env_var)
        if cred_file_name is None:
            raise ValueError(f"Environment variable {credentials_env_var} is not set.")
        credentials_path = os.path.join(os.getcwd(), cred_file_name)
        os.environ[credentials_env_var] = credentials_path
        print(f"Credentials set for {credentials_env_var}.")
