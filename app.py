from infrastructure.database import (
    Database,
    DocumentRepository,
    SchemaManager,
)


def main() -> None:
    database = Database()

    SchemaManager(database).initialize()

    repository = DocumentRepository(database)

    documents = repository.get_all()

    print("\n===== DOCUMENTS =====")

    if not documents:
        print("No documents in the database.")
    else:
        for row in documents:
            print(dict(row))


if __name__ == "__main__":
    main()