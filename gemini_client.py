import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class GeminiClient:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Please set it to your Gemini API key or add it to a .env file."
                )
            from google import genai

            self._client = genai.Client(api_key=api_key)
            self._model = "gemini-flash-latest"

    def generate_content(
        self,
        contents: list,
        config: Any = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> Any:
        from google.genai import types

        try:
            result = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            return result
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = initial_delay
                for attempt in range(max_retries):
                    print(
                        f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    try:
                        result = self._client.models.generate_content(
                            model=self._model,
                            contents=contents,
                            config=config,
                        )
                        return result
                    except Exception as e2:
                        if attempt == max_retries - 1:
                            raise e2
                        print(f"Retry failed: {e2}")
            raise

    def generate_json(
        self,
        prompt: str,
        response_schema: Any,
        system_instruction: str = None,
    ) -> dict:
        from google.genai import types

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        if system_instruction:
            config.system_instruction = [types.Part.from_text(text=system_instruction)]

        result = self.generate_content(contents, config)
        return json.loads(result.text)


def get_gemini_client() -> GeminiClient:
    return GeminiClient()
