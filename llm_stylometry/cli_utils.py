"""Utility functions for cross-platform compatibility."""

import sys
import re
import platform


def is_windows():
    """Check if running on Windows."""
    return platform.system() == 'Windows'


def strip_unicode(text):
    """Remove Unicode emoji and special characters for Windows compatibility."""
    # Remove common emoji ranges and special characters
    # Keep basic ASCII plus some extended ASCII that Windows terminal can handle
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F926-\U0001F937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )

    # Replace specific Unicode characters with ASCII equivalents
    replacements = {
        '✓': '[OK]',
        '✔': '[OK]',
        '✅': '[OK]',
        '✗': '[FAIL]',
        '✘': '[FAIL]',
        '❌': '[FAIL]',
        '⚠️': '[WARNING]',
        '⚠': '[WARNING]',
        '╔': '+',
        '╗': '+',
        '╚': '+',
        '╝': '+',
        '═': '=',
        '║': '|',
        '→': '->',
        '├': '+',
        '└': '+',
        '│': '|',
        '─': '-',
        '🤖': '[BOT]',
    }

    text = str(text)
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Remove remaining emojis
    text = emoji_pattern.sub('', text)

    return text


def safe_print(*args, **kwargs):
    """
    Print function that handles Unicode/emoji gracefully on Windows.

    Falls back to ASCII-only output if Unicode printing fails.
    """
    # Convert all args to strings and join them
    message = ' '.join(str(arg) for arg in args)

    # Try printing normally first
    try:
        print(message, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If that fails, strip Unicode and try again
        try:
            safe_message = strip_unicode(message)
            print(safe_message, **kwargs)
        except Exception:
            # Last resort: ASCII only
            ascii_message = ''.join(char if ord(char) < 128 else '?' for char in message)
            print(ascii_message, **kwargs)


def format_header(title, width=60, char='='):
    """Format a header with borders, Windows-compatible."""
    if is_windows():
        # Simple ASCII borders for Windows
        top = '+' + char * (width - 2) + '+'
        middle = f"| {title:^{width-4}} |"
        bottom = '+' + char * (width - 2) + '+'
    else:
        # Fancy Unicode borders for Unix-like systems
        top = "╔" + "═" * (width - 2) + "╗"
        middle = f"║ {title:^{width-4}} ║"
        bottom = "╚" + "═" * (width - 2) + "╝"

    return f"\n{top}\n{middle}\n{bottom}\n"