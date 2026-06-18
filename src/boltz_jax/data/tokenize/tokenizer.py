from abc import ABC, abstractmethod

from boltz_jax.data.types import Input, Tokenized


class Tokenizer(ABC):
    """Tokenize an input structure for training."""

    @abstractmethod
    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        raise NotImplementedError
