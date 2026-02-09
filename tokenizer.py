import json
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


# Fork-race special tokens
FORK_TOKEN = "<fork>"
FORK1_TOKEN = "<fork1>"
FORK2_TOKEN = "<fork2>"
FORK_TOKENS = [FORK_TOKEN, FORK1_TOKEN, FORK2_TOKEN]


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

        # Fork token IDs (None until add_fork_tokens is called)
        self.fork_token_id: Optional[int] = None
        self.fork1_token_id: Optional[int] = None
        self.fork2_token_id: Optional[int] = None
        self._fork_tokens_added: bool = False

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def add_fork_tokens(self) -> int:
        """Add fork special tokens to the tokenizer vocabulary.

        Returns the new vocab size after adding tokens.
        """
        if self._fork_tokens_added:
            return self.vocab_size

        from tokenizers import AddedToken

        new_tokens = [
            AddedToken(FORK_TOKEN, special=True),
            AddedToken(FORK1_TOKEN, special=True),
            AddedToken(FORK2_TOKEN, special=True),
        ]
        num_added = self.tokenizer.add_special_tokens(new_tokens)

        self.fork_token_id = self.tokenizer.token_to_id(FORK_TOKEN)
        self.fork1_token_id = self.tokenizer.token_to_id(FORK1_TOKEN)
        self.fork2_token_id = self.tokenizer.token_to_id(FORK2_TOKEN)
        self._fork_tokens_added = True

        return self.vocab_size

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
