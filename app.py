from infrastructure.database import (
    Database,
    SchemaManager,
)


def main() -> None:

    database = Database()

    schema = SchemaManager(database)

    schema.initialize()

    print("Atlas database initialized successfully.")


if __name__ == "__main__":
    main()