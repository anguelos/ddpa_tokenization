import string
from abc import ABC, abstractmethod
from typing import Tuple, final, Dict
import re

t_token = Tuple[int, int, int]
t_tokenization = Tuple[str, Tuple[t_token, ...], str]


def tokenize(txt: str, tokenizer_id: str) -> type:
    """
    Tokenize the given text using the specified tokenizer.

    :param txt: The text to be tokenized.
    :type txt: str
    :param tokenizer_id: The ID of the tokenizer to be used.
    :type tokenizer_id: str
    :return: The tokenized text.
    :rtype: type
    """
    t = Tokenizer.get_tokenizer(tokenizer_id)
    return t().tokenize(txt)


def list_tokenizers() -> Tuple[str, ...]:
    """
    List all known tokenizers.

    :return: The IDs of all known tokenizers.
    :rtype: Tuple[str, ...]
    """
    return tuple([cls.ID() for cls in Tokenizer.get_known_children()])


class Tokenizer(ABC):
    """Abstract class for tokenizers.
    
    - Tokenizers are used to convert a string into a sequence of tokens and back.
    - Dictionary tokenizers have a fixed set of tokens, while string tokenizers do not.
    - All tokenizer classes must have a unique string ID, which is used to identify them.
    - All dictionary tokenizers must have a dictionary of tokens, which is used to convert tokens to strings and back.
    - All tokenizers must be deterministic, i.e. the same input must always produce the same output.
    - All tokenizer must be instatiatable without arguments.
    - Other than auxiliary data (eg: caching paths), tokenizers must be immutable.
    """

    @classmethod
    @abstractmethod
    def ID(cls):
        """Get the unique string ID of the tokenizer."""
        pass
    
    @abstractmethod
    def create_dictionary_tokens(self, text: str) -> Tuple[t_token, ...]:
        """Create dictionary tokens from the given text.
        
        Args:
            text (str): The input text.
        
        Returns:
            Tuple[t_token, ...]: The dictionary tokens.
        """
        pass

    def create_string_tokens(self, text: str) -> Tuple[str, ...]:
        """
        Creates a tuple of strings from the text.
        
        This method must be redefined by Tokenizers that do not have a dictionary of tokens.
        
        Args:
            text (str): The input text.
        
        Returns:
            Tuple[str, ...]: The string tokens.
        """
        tokens = self.create_dictionary_tokens(text)
        return tuple([self.num2str(s) for s, start, end in tokens])

    
    @final
    def tokenize(self, text: str) -> t_tokenization:
        """Tokenize the text using dictionary tokens.
        
        Args:
            text (str): The input text.
        
        Returns:
            t_tokenization: The tokenization result, which includes the original text, dictionary tokens, and tokenizer ID.
        """
        return text, self.create_dictionary_tokens(text), type(self).ID()
    
    @classmethod
    def get_known_children(cls) -> Tuple[type, ...]:
        """Get all subclasses of this class.
        
        Tokenizer itself is not returned.
        
        Returns:
            Tuple[type, ...]: The known subclasses of Tokenizer.
        """
        children = cls.__subclasses__()
        return tuple([cl_cls for cl_cls in children if cl_cls is not cls])
    
    @staticmethod
    def get_tokenizer(id: str) -> type:
        """Get a tokenizer by its ID.
        
        Args:
            id (str): The ID of the tokenizer.
        
        Returns:
            type: The tokenizer class.
        
        Raises:
            ValueError: If the tokenizer ID is unknown.
        """
        for cls in Tokenizer.get_known_children():
            if cls.ID() == id:
                return cls
        raise ValueError(f"Unknown tokenizer ID: {id}")
        
        

    @classmethod
    def has_dictionary(cls) -> bool:
        """Check if the tokenizer has a dictionary.
        
        Returns:
            bool: True if the tokenizer has a dictionary, False otherwise.
        """
        return cls().dictionary_size() > 0

    @abstractmethod
    def num2str(self, num: int) -> str:
        """Convert a number to a string.
        
        This is used to convert the tokenized text back to the original text.
        
        Args:
            num (int): The number to convert.
        
        Returns:
            str: The converted string.
        """
        raise NotImplemented
    
    @abstractmethod
    def str2num(self, string: str) -> int:
        """Convert a string to a number.
        
        This is used to convert the tokenized text back to the original text.
        
        Args:
            string (str): The string to convert.
        
        Returns:
            int: The converted number.
        """
        raise NotImplemented
    
    def dictionary_size(self) -> int:
        """Get the size of the dictionary.
        
        Returns:
            int: The size of the dictionary, 0 if the tokenizer does not have a dictionary.
        """
        return 0
    
    def dictionary_dict(self) -> Dict[str, int]:
        """Get the dictionary of tokens.
        
        Returns:
            Dict[str, int]: The dictionary of tokens.
        """
        raise NotImplemented
    

class TokenizerAsciiChar(Tokenizer):
    """Tokenizer that tokenizes text based on ASCII characters.
    
    This tokenizer treats each ASCII character as a separate token.
    """

    charset = ("<unk>", " ") + tuple(sorted(string.ascii_letters+string.digits+string.punctuation))
    token_idx = {c: n for n, c in enumerate(charset)}
    inv_token_idx = {n: c for n, c in enumerate(charset)}

    @classmethod
    def ID(cls):
        """Get the unique string ID of the tokenizer."""
        return "AsciiChar"
    
    def create_dictionary_tokens(self, text: str) -> t_tokenization:
        """Create dictionary tokens from the given text.
        
        Args:
            text (str): The input text.
        
        Returns:
            t_tokenization: The tokenization result, which includes the original text, dictionary tokens, and tokenizer ID.
        """
        separators = list(re.finditer("\\s+", text))
        separators = [(s.start(), s.end()) for s in separators]
        separators = [(0, 0)] + separators + [(len(text), len(text))]
        res = []
        id_from_to = []
        for n in range(len(separators)-1):
            block_id_from_to = [(self.str2num(text[i]), i, i+1) for i in range(separators[n][1], separators[n+1][0])]
            id_from_to += block_id_from_to
        return tuple(id_from_to)
        
    
    def num2str(self, num: int) -> str:
        """Convert a number to a string.
        
        This is used to convert the tokenized text back to the original text.
        Numbers < 0 or >= dictionary_size() are NOT converted to "<UNKN>".
        
        Args:
            num (int): The number to convert.
        
        Returns:
            str: The converted string.
        """
        return TokenizerAsciiChar.inv_token_idx[num]
    
    def str2num(self, string: str) -> int:
        """Convert a string to a number.
        
        Unknown strings are converted to 0.
        
        Args:
            string (str): The string to convert.
        
        Returns:
            int: The converted number.
        """
        return TokenizerAsciiChar.token_idx.get(string, 0)

    def dictionary_dict(self) -> Dict[str, int]:
        """Get the dictionary of tokens.
        
        Returns:
            Dict[str, int]: The dictionary of tokens.
        """
        return TokenizerAsciiChar.token_idx.copy()
    
    def dictionary_size(self) -> int:
        """Get the size of the dictionary.
        
        Returns:
            int: The size of the dictionary.
        """
        return len(TokenizerAsciiChar.charset)
