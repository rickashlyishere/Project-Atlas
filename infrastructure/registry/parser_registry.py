from __future__ import annotations

from pathlib import Path

from infrastructure.parsers.base_parser import BaseParser


class ParserRegistry:
    """
    Stores every parser.
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def get_parser(self, path: Path) -> BaseParser:
        extension = path.suffix.lower()

        for parser in self._parsers:
            if extension in parser.supported_extensions:
                return parser

        raise ValueError(f"No parser registered for {extension}")