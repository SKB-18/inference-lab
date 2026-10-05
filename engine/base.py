from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseEngine(ABC):
    """
    Abstract base class for all inference engines.
    Ensures a standard interface across Dummy, Transformers, and llama.cpp engines.
    """
    
    @abstractmethod
    async def generate_stream(self, prompt: str, max_tokens: int = 100) -> AsyncGenerator[str, None]:
        """
        Yields tokens one by one as they are generated.
        """
        pass
