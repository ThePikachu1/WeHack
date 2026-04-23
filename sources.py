from abc import ABC, abstractmethod
from typing import Any


class SourceProvider(ABC):
    @abstractmethod
    def get_content(self) -> str:
        pass

    @abstractmethod
    def get_source_id(self) -> str:
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        pass

    @abstractmethod
    def get_source_key(self) -> str:
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceProvider):
            return False
        return self.get_source_key() == other.get_source_key()

    def __hash__(self) -> int:
        return hash(self.get_source_key())


class WebSourceProvider(SourceProvider):
    def __init__(self, url: str):
        self._url = url
        self._cached_content = None

    def get_content(self) -> str:
        if self._cached_content is None:
            import requests

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(self._url, headers=headers)
            response.raise_for_status()
            self._cached_content = response.text
        return self._cached_content

    def get_source_id(self) -> str:
        return self._url.split("/")[-1].replace("_", " ").replace("-", " ")

    def get_source_type(self) -> str:
        return "web"

    def get_source_key(self) -> str:
        return f"web:{self._url}"


class TextSourceProvider(SourceProvider):
    def __init__(self, text: str, source_id: str):
        self._text = text
        self._source_id = source_id

    def get_content(self) -> str:
        return self._text

    def get_source_id(self) -> str:
        return self._source_id

    def get_source_type(self) -> str:
        return "text"

    def get_source_key(self) -> str:
        return f"text:{self._source_id}"


class FileSourceProvider(SourceProvider):
    def __init__(self, file_path: str):
        import os

        self._file_path = os.path.abspath(file_path)
        self._cached_content = None

    def get_content(self) -> str:
        if self._cached_content is None:
            with open(self._file_path, "r") as f:
                self._cached_content = f.read()
        return self._cached_content

    def get_source_id(self) -> str:
        import os

        return os.path.basename(self._file_path)

    def get_source_type(self) -> str:
        return "file"

    def get_source_key(self) -> str:
        return f"file:{self._file_path}"
