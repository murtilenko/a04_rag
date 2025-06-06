import requests
import json
import logging


class LLMClient:
    """
    Handles direct interactions with a locally running LLM API.
    """
    def __init__(self,
                 llm_api_url: str,
                 llm_model_name: str):

        self.llm_api_url = llm_api_url
        self.llm_model_name = llm_model_name

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMClient: llm_api_url: {self.llm_api_url}, model_name: {self.llm_model_name}")

    def query(self, prompt: str):
        payload = {
            "model": self.llm_model_name,  # e.g. 'llama2'
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.llm_api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying LLM: {e}")
            return "Error: Could not connect to the LLM."


