"""Text cleaning and normalization utilities for Notely."""

import re


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and formatting without destroying paragraphs, headings, or lists.

    - Replaces Windows \\r\\n and old Mac \\r with standard \\n.
    - Resolves soft hyphens and hyphenated line-breaks (e.g. 'struc-\\nture' -> 'structure').
    - Normalizes non-breaking and unusual unicode spaces to standard space.
    - Condenses multiple consecutive horizontal spaces to a single space.
    - Preserves paragraph separations (two or more newlines become exactly two newlines).
    - Preserves markdown/bullet-list structure.
    - Strips leading and trailing blank lines.
    """
    if not raw_text:
        return ""

    # 1. Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize non-standard spaces (non-breaking space, zero-width space, tabs)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\t", "    ")

    # 3. Fix hyphenated line breaks (e.g., "re- \n quirements" -> "requirements")
    text = re.sub(r"(\b[a-zA-Z]+)-\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # 4. Process line-by-line: trim trailing/leading spaces while preserving indent for lists
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple horizontal whitespace into one, unless it's a bullet indent
        leading_spaces = len(line) - len(line.lstrip(" "))
        content = re.sub(r"[ ]{2,}", " ", line.strip())
        if content:
            if leading_spaces >= 2 and (content.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5."))):
                cleaned_lines.append("  " + content)
            else:
                cleaned_lines.append(content)
        else:
            cleaned_lines.append("")

    text = "\n".join(cleaned_lines)

    # 5. Condense 3+ newlines down to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_meaningful_text(text: str, min_chars: int = 15) -> bool:
    """Check if the provided string contains meaningful alphanumeric content."""
    if not text:
        return False
    alphanumeric_count = sum(1 for c in text if c.isalnum())
    return alphanumeric_count >= min_chars
