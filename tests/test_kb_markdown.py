from pathlib import Path

from ownrun_runner.kb.store import parse_document


def test_parse_md_splits_headings_and_strips_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "---\ntitle: Notes\n---\n"
        "# Intro\n\n"
        "A short introduction that is long enough to keep.\n\n"
        "## Details\n\n"
        "More text that belongs under details and should stay together.\n\n"
        "```\n# not a heading\n```\n",
        encoding="utf-8",
    )

    sections = parse_document(path, "md")
    titles = [section.section_title for section in sections]
    assert titles == ["Intro", "Details"]
    assert "A short introduction" in sections[0].text
    assert "title: Notes" not in "".join(section.text for section in sections)
    assert "# not a heading" in sections[1].text
    assert all(section.section_title != "not a heading" for section in sections)


def test_parse_markdown_extension_alias_is_callers_job(tmp_path: Path) -> None:
    path = tmp_path / "doc.markdown"
    path.write_bytes(b"# Title\n\nBody text long enough to be a section.\n")
    sections = parse_document(path, "md")
    assert sections[0].section_title == "Title"
    assert "Body text" in sections[0].text


def test_parse_txt_is_single_section(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Line one.\n\nLine two is also here.\n", encoding="utf-8")
    sections = parse_document(path, "txt")
    assert len(sections) == 1
    assert sections[0].section_title is None
    assert "Line one." in sections[0].text
    assert "Line two is also here." in sections[0].text


def test_parse_txt_empty_returns_no_sections(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")
    assert parse_document(path, "txt") == []
