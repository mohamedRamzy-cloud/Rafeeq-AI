from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    def stream(self, messages):
        """
        Must return generator of text chunks
        """
        pass