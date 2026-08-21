from pathlib import Path


def folder_has_content(folder_path: str) -> bool:
    path = Path(folder_path)

    if not path.is_dir():
        return False

    return any(path.iterdir())


def get_pdf_files(folder_path: str) -> list[Path]:
    path = Path(folder_path)

    if not path.is_dir():
        return []

    return list(path.rglob("*.pdf"))