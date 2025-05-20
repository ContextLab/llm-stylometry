from pathlib import Path
import re
from cleantext import clean
from constants import RAW_DATA_DIR, CLEANED_DATA_DIR, TITLES_FILE, AUTHORS


def alpha(string):
    return "".join(char for char in string if char.isalpha())


def clean_book(text: str) -> str:
    # Remove BOM character if present
    text = text.lstrip("\ufeff")

    # Remove Gutenberg footer and header
    gut_start = "*** START OF THE PROJECT GUTENBERG"
    gut_end = "*** END OF THE PROJECT GUTENBERG"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if gut_start in line:
            lines = lines[i + 1 :]
            break
    for i, line in enumerate(lines):
        if gut_end in line:
            lines = lines[:i]
            break
    text = "\n".join(lines)
    assert gut_start not in text
    assert gut_end not in text

    # Strip whitespace from each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove star lines
    text = "\n".join(line.replace("*", "").strip() for line in text.splitlines())

    # Remove book titles and chapter titles
    titles_path = TITLES_FILE
    assert titles_path.is_file(), f"Titles file not found: {titles_path}"
    with titles_path.open("r") as f:
        chapters = f.readlines()
    assert chapters, "Titles file is empty"
    chapters = set([alpha(chapter.lower()) for chapter in chapters])
    text = "\n".join(
        [line for line in text.splitlines() if alpha(line.lower()) not in chapters]
    )

    # Group lines into paragraphs
    paragraphs = [
        paragraph.replace("\n", " ").strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    text = "\n\n".join(paragraphs)

    # Remove chapter headings titles
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.lower().strip().startswith("chapter")
    )
    assert not any(
        "chapter" in line.lower() and len(line) < 15 for line in text.splitlines()
    )

    # Check for remaining unwanted words
    assert "gutenberg" not in text.lower()
    assert not any(
        len(line) < 15 and "page" in line.lower() for line in text.splitlines()
    )

    # Remove illustration tags
    text = "\n".join(
        line
        for line in text.splitlines()
        if "[illustration" not in line.lower() and "illustration:" not in line.lower()
    )

    # Remove transcriber notes
    assert not re.search(r"(?<!villainous )transcriber", text, flags=re.IGNORECASE)

    # Remove underscores and stars
    text = text.replace("_", "")
    text = text.replace("*", "")

    # Use cleantext to replace non-ascii characters and convert to lowercase
    text = clean(text, fix_unicode=True, to_ascii=True)

    assert (
        "footnote" not in text.lower()
        or "britten's footnotes" in text.lower()
        or "it had explanatory footnotes" in text.lower()
    )
    assert (
        "copyright" not in text.lower()
        or "conceding his copyright" in text.lower()
        or "sort of copyright" in text.lower()
    )
    assert "jpg" not in text.lower()
    assert "png" not in text.lower()

    return text


def clean_all_books():
    assert RAW_DATA_DIR.is_dir(), f"Raw directory not found: {RAW_DATA_DIR}"
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for author in reversed(AUTHORS):
        subdir_raw = RAW_DATA_DIR / author
        subdir_cleaned = CLEANED_DATA_DIR / author
        assert subdir_raw.is_dir(), f"Expected a directory but found: {subdir_raw}"
        subdir_cleaned.mkdir(exist_ok=True)
        for file in subdir_raw.glob("*"):
            print(f"Processing file at: {subdir_raw / file.name}")
            assert file.is_file(), f"Expected a file but found: {file}"
            text = file.read_text(encoding="utf-8")
            cleaned_text = clean_book(text)
            (subdir_cleaned / file.name).write_text(cleaned_text, encoding="utf-8")
            print(f"Saved cleaned book to: {subdir_cleaned / file.name}")
        print(f"Finished processing {author} books.\n\n")
    print(f"Cleaned books saved to {CLEANED_DATA_DIR}")


if __name__ == "__main__":
    clean_all_books()
