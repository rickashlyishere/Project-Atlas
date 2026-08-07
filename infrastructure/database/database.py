from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """
    SQLite database manager.
    """

    def __init__(self, database_path: Path | str = "data/atlas.db") -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

    def execute(
        self,
        query: str,
        parameters: tuple = (),
    ) -> sqlite3.Cursor:

        cursor = self.connection.cursor()

        cursor.execute(query, parameters)

        self.connection.commit()

        return cursor

    def executemany(
        self,
        query: str,
        parameters: list[tuple],
    ) -> None:

        cursor = self.connection.cursor()

        cursor.executemany(query, parameters)

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()