from typing import Dict
from loguru import logger


class Chatbot:
    def __init__(self):
        self.sessions: Dict[str, str] = {}

    def generate_response(self, user_id: str, message: str) -> str:
        # Check if there's a session for this user and retrieve it
        user_session = self.sessions.get(user_id, "")
        # Add the user's message to the session for context
        user_session += f"User: {message}\n"

        # Instantiate the LLM Model handler and get the response
        llm = LLMModel()
        response = llm.get_response(user_session)

        # Update the session with the bot's response
        user_session += f"Bot: {response}\n"
        self.sessions[user_id] = user_session

        return response


class LLMModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        logger.info("Loading the model...")
        # Add specific loading code here (e.g., using transformers, llama-cpp-python, etc.)
        # For example, self.model = LocalLLMModelClass.load(...)
        return "LoadedModel"

    def get_response(self, prompt: str) -> str:
        response = self.model.generate(prompt)
        return response