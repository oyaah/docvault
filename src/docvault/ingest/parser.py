"""Document parser — extracts text and structure from various formats."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    title: str
    content: str
    level: int  # heading level (1=h1, 2=h2, etc.)
    children: list["Section"] = field(default_factory=list)
    path: str = ""  # e.g., "Benefits > PTO > Accrual"


@dataclass
class ParsedDocument:
    title: str
    source_path: str
    sections: list[Section]
    raw_text: str
    format: str  # 'markdown', 'pdf', 'html', 'docx'


def parse_file(file_path: Path) -> ParsedDocument:
    suffix = file_path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(file_path)
    elif suffix == ".txt":
        return _parse_plaintext(file_path)
    elif suffix == ".pdf":
        return _parse_pdf(file_path)
    elif suffix in (".html", ".htm"):
        return _parse_html(file_path)
    else:
        return _parse_plaintext(file_path)


def _parse_markdown(file_path: Path) -> ParsedDocument:
    text = file_path.read_text(encoding="utf-8")
    title = file_path.stem.replace("-", " ").replace("_", " ").title()

    # Extract title from first h1 if present
    first_h1 = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    if first_h1:
        title = first_h1.group(1).strip()

    sections = _extract_markdown_sections(text)
    return ParsedDocument(
        title=title,
        source_path=str(file_path),
        sections=sections,
        raw_text=text,
        format="markdown",
    )


def _extract_markdown_sections(text: str) -> list[Section]:
    """Split markdown into hierarchical sections by headings."""
    lines = text.split("\n")
    sections: list[Section] = []
    current_section: Section | None = None
    current_content_lines: list[str] = []

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section:
                current_section.content = "\n".join(current_content_lines).strip()
                sections.append(current_section)
            elif current_content_lines:
                # Content before first heading
                sections.append(
                    Section(title="Introduction", content="\n".join(current_content_lines).strip(), level=0)
                )

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_section = Section(title=title, content="", level=level)
            current_content_lines = []
        else:
            current_content_lines.append(line)

    # Last section
    if current_section:
        current_section.content = "\n".join(current_content_lines).strip()
        sections.append(current_section)
    elif current_content_lines:
        sections.append(
            Section(title="Content", content="\n".join(current_content_lines).strip(), level=0)
        )

    # Build section paths
    _build_section_paths(sections)

    return sections


def _build_section_paths(sections: list[Section]):
    """Assign hierarchical paths like 'Benefits > PTO > Accrual'."""
    path_stack: list[tuple[int, str]] = []  # (level, title)

    for section in sections:
        # Pop stack until we find a parent level
        while path_stack and path_stack[-1][0] >= section.level:
            path_stack.pop()

        path_stack.append((section.level, section.title))
        section.path = " > ".join(t for _, t in path_stack)


def _parse_plaintext(file_path: Path) -> ParsedDocument:
    text = file_path.read_text(encoding="utf-8")
    title = file_path.stem.replace("-", " ").replace("_", " ").title()

    # Split on double newlines as rough section boundaries
    paragraphs = re.split(r"\n{2,}", text)
    sections = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            sections.append(
                Section(title=f"Section {i + 1}", content=para, level=1, path=f"Section {i + 1}")
            )

    return ParsedDocument(
        title=title,
        source_path=str(file_path),
        sections=sections,
        raw_text=text,
        format="text",
    )


def _parse_pdf(file_path: Path) -> ParsedDocument:
    """Parse PDF using docling if available, fallback to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        md_text = result.document.export_to_markdown()
        # Re-parse the markdown output
        title = file_path.stem.replace("-", " ").replace("_", " ").title()
        sections = _extract_markdown_sections(md_text)
        return ParsedDocument(
            title=title,
            source_path=str(file_path),
            sections=sections,
            raw_text=md_text,
            format="pdf",
        )
    except ImportError:
        # Fallback: treat as plain text extraction
        import subprocess

        try:
            text = subprocess.check_output(
                ["pdftotext", str(file_path), "-"], stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="replace")
        except (FileNotFoundError, subprocess.CalledProcessError):
            text = f"[PDF parsing unavailable for {file_path.name}]"

        return _parse_plaintext_content(file_path, text, "pdf")


def _parse_html(file_path: Path) -> ParsedDocument:
    """Basic HTML parsing — strip tags, extract structure from headers."""
    text = file_path.read_text(encoding="utf-8")

    try:
        from markitdown import MarkItDown

        mid = MarkItDown()
        result = mid.convert(str(file_path))
        md_text = result.text_content
        sections = _extract_markdown_sections(md_text)
    except ImportError:
        # Fallback: strip HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        sections = [Section(title="Content", content=clean.strip(), level=0, path="Content")]
        md_text = clean

    title = file_path.stem.replace("-", " ").replace("_", " ").title()
    return ParsedDocument(
        title=title,
        source_path=str(file_path),
        sections=sections,
        raw_text=md_text,
        format="html",
    )


def _parse_plaintext_content(file_path: Path, text: str, fmt: str) -> ParsedDocument:
    title = file_path.stem.replace("-", " ").replace("_", " ").title()
    paragraphs = re.split(r"\n{2,}", text)
    sections = [
        Section(title=f"Section {i + 1}", content=p.strip(), level=1, path=f"Section {i + 1}")
        for i, p in enumerate(paragraphs)
        if p.strip()
    ]
    return ParsedDocument(
        title=title, source_path=str(file_path), sections=sections, raw_text=text, format=fmt
    )
