"""Text and linguistics domain - string distances, readability metrics.

This module provides deterministic text analysis computations.
"""

import re
from typing import Any, Dict, List, Optional, Union
from collections import Counter

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# String Distance Functions
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    The minimum number of single-character edits (insertions, deletions,
    substitutions) required to transform s1 into s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance as integer
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two strings of equal length.

    Number of positions where corresponding characters differ.

    Args:
        s1: First string
        s2: Second string (must be same length as s1)

    Returns:
        Hamming distance as integer

    Raises:
        ValueError: If strings have different lengths
    """
    if len(s1) != len(s2):
        raise ValueError(f"Strings must have equal length: {len(s1)} != {len(s2)}")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def jaccard_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Compute Jaccard similarity between two strings using n-grams.

    J(A,B) = |A ∩ B| / |A ∪ B|

    Args:
        s1: First string
        s2: Second string
        n: N-gram size (default 2 for bigrams)

    Returns:
        Jaccard similarity coefficient (0 to 1)
    """
    def get_ngrams(s: str, n: int) -> set:
        s = s.lower()
        return set(s[i:i+n] for i in range(len(s) - n + 1))

    ngrams1 = get_ngrams(s1, n)
    ngrams2 = get_ngrams(s2, n)

    if not ngrams1 and not ngrams2:
        return 1.0  # Both empty

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def longest_common_subsequence(s1: str, s2: str) -> int:
    """Compute length of longest common subsequence.

    A subsequence is a sequence that can be derived from another sequence
    by deleting some elements without changing the order.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of LCS
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Compute n-gram overlap similarity.

    Args:
        s1: First string
        s2: Second string
        n: N-gram size

    Returns:
        Similarity score (0 to 1)
    """
    def get_ngrams(s: str, n: int) -> Counter:
        s = s.lower()
        return Counter(s[i:i+n] for i in range(len(s) - n + 1))

    ngrams1 = get_ngrams(s1, n)
    ngrams2 = get_ngrams(s2, n)

    total1 = sum(ngrams1.values())
    total2 = sum(ngrams2.values())

    if total1 == 0 and total2 == 0:
        return 1.0

    common = sum((ngrams1 & ngrams2).values())
    return 2 * common / (total1 + total2) if (total1 + total2) > 0 else 0.0


# =============================================================================
# Readability Functions
# =============================================================================

def syllable_count(word: str) -> int:
    """Estimate syllable count for a word.

    Uses a simple heuristic based on vowel groups.

    Args:
        word: Word to analyze

    Returns:
        Estimated syllable count
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1

    # Adjust for -le endings
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1

    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Calculate Flesch-Kincaid Grade Level.

    FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Args:
        text: Text to analyze

    Returns:
        Grade level (approximate US school grade)
    """
    sentences = _count_sentences(text)
    words = _get_words(text)
    word_count = len(words)

    if word_count == 0 or sentences == 0:
        return 0.0

    syllables = sum(syllable_count(w) for w in words)

    return (0.39 * (word_count / sentences) +
            11.8 * (syllables / word_count) - 15.59)


def gunning_fog_index(text: str) -> float:
    """Calculate Gunning Fog Index.

    Fog = 0.4 * ((words/sentences) + 100 * (complex_words/words))
    Complex words = words with 3+ syllables

    Args:
        text: Text to analyze

    Returns:
        Fog index (years of education needed to understand)
    """
    sentences = _count_sentences(text)
    words = _get_words(text)
    word_count = len(words)

    if word_count == 0 or sentences == 0:
        return 0.0

    complex_words = sum(1 for w in words if syllable_count(w) >= 3)

    return 0.4 * ((word_count / sentences) + 100 * (complex_words / word_count))


def word_count(text: str) -> Dict[str, int]:
    """Count words, characters, sentences, and paragraphs.

    Args:
        text: Text to analyze

    Returns:
        Dict with counts
    """
    words = _get_words(text)
    return {
        "words": len(words),
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "").replace("\n", "")),
        "sentences": _count_sentences(text),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "lines": len(text.split("\n")),
    }


def word_frequency(text: str, top_n: int = 10) -> List[tuple]:
    """Get most frequent words in text.

    Args:
        text: Text to analyze
        top_n: Number of top words to return

    Returns:
        List of (word, count) tuples
    """
    words = _get_words(text)
    return Counter(words).most_common(top_n)


# =============================================================================
# Utility Functions
# =============================================================================

def checksum_luhn(number: str) -> bool:
    """Validate a number using the Luhn algorithm (credit cards, etc).

    Args:
        number: String of digits to validate

    Returns:
        True if valid, False otherwise
    """
    digits = [int(d) for d in number if d.isdigit()]
    if not digits:
        return False

    # Double every second digit from right
    for i in range(len(digits) - 2, -1, -2):
        doubled = digits[i] * 2
        digits[i] = doubled - 9 if doubled > 9 else doubled

    return sum(digits) % 10 == 0


def isbn_validate(isbn: str) -> bool:
    """Validate an ISBN-10 or ISBN-13.

    Args:
        isbn: ISBN string (may include hyphens)

    Returns:
        True if valid, False otherwise
    """
    # Remove hyphens and spaces
    isbn = isbn.replace("-", "").replace(" ", "")

    if len(isbn) == 10:
        return _isbn10_check(isbn)
    elif len(isbn) == 13:
        return _isbn13_check(isbn)
    return False


def _isbn10_check(isbn: str) -> bool:
    """Validate ISBN-10."""
    if not isbn[:9].isdigit():
        return False
    if isbn[9] not in "0123456789Xx":
        return False

    total = sum((10 - i) * (10 if c in "Xx" else int(c))
                for i, c in enumerate(isbn))
    return total % 11 == 0


def _isbn13_check(isbn: str) -> bool:
    """Validate ISBN-13."""
    if not isbn.isdigit():
        return False

    total = sum((1 if i % 2 == 0 else 3) * int(c)
                for i, c in enumerate(isbn))
    return total % 10 == 0


def roman_numeral(value: Union[int, str]) -> Union[str, int]:
    """Convert between Roman numerals and integers.

    Args:
        value: Integer to convert to Roman, or Roman numeral string

    Returns:
        Converted value
    """
    if isinstance(value, int):
        return _int_to_roman(value)
    else:
        return _roman_to_int(str(value))


def _int_to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    if num < 1 or num > 3999:
        raise ValueError("Number must be between 1 and 3999")

    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    result = []
    for val, numeral in values:
        count = num // val
        if count:
            result.append(numeral * count)
            num -= val * count
    return ''.join(result)


def _roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer."""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                    'C': 100, 'D': 500, 'M': 1000}

    s = s.upper()
    total = 0
    prev = 0

    for char in reversed(s):
        curr = roman_values.get(char, 0)
        if curr < prev:
            total -= curr
        else:
            total += curr
        prev = curr

    return total


def base_convert(number: str, from_base: int, to_base: int) -> str:
    """Convert number between bases (2, 8, 10, 16).

    Args:
        number: Number string in source base
        from_base: Source base (2, 8, 10, or 16)
        to_base: Target base (2, 8, 10, or 16)

    Returns:
        Number string in target base
    """
    # Convert to decimal first
    decimal = int(number, from_base)

    # Convert to target base
    if to_base == 10:
        return str(decimal)
    elif to_base == 2:
        return bin(decimal)[2:]
    elif to_base == 8:
        return oct(decimal)[2:]
    elif to_base == 16:
        return hex(decimal)[2:].upper()
    else:
        raise ValueError(f"Unsupported base: {to_base}")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_words(text: str) -> List[str]:
    """Extract words from text."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def _count_sentences(text: str) -> int:
    """Count sentences in text."""
    return len(re.findall(r'[.!?]+', text)) or 1


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_two_strings(args):
    if isinstance(args, dict):
        s1 = args.get("s1", args.get("string1", args.get("a")))
        s2 = args.get("s2", args.get("string2", args.get("b")))
        return s1, s2
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return args[0], args[1]
    raise ValueError("Expected two strings")


def _parse_text(args):
    if isinstance(args, dict):
        return args.get("text", args.get("s", list(args.values())[0]))
    return str(args)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register text/linguistics tools in the registry."""

    # String Distance Tools
    registry.register(ToolSpec(
        name="text_levenshtein",
        function=lambda args: levenshtein_distance(*_parse_two_strings(args)),
        description="Compute Levenshtein (edit) distance - minimum edits to transform one string to another.",
        parameters={
            "type": "object",
            "properties": {
                "s1": {"type": "string", "description": "First string"},
                "s2": {"type": "string", "description": "Second string"},
            },
            "required": ["s1", "s2"]
        },
        returns="Edit distance as integer",
        examples=[
            {"input": {"s1": "kitten", "s2": "sitting"}, "output": 3},
        ],
        domain="text",
        tags=["string", "distance", "edit", "similarity"],
    ))

    registry.register(ToolSpec(
        name="text_hamming",
        function=lambda args: hamming_distance(*_parse_two_strings(args)),
        description="Compute Hamming distance - positions where characters differ (strings must be equal length).",
        parameters={
            "type": "object",
            "properties": {
                "s1": {"type": "string", "description": "First string"},
                "s2": {"type": "string", "description": "Second string (same length as s1)"},
            },
            "required": ["s1", "s2"]
        },
        returns="Hamming distance as integer",
        examples=[
            {"input": {"s1": "karolin", "s2": "kathrin"}, "output": 3},
        ],
        domain="text",
        tags=["string", "distance", "hamming"],
    ))

    def _jaccard_handler(args):
        s1, s2 = _parse_two_strings(args)
        n = args.get("n", 2) if isinstance(args, dict) else 2
        return jaccard_similarity(s1, s2, n)

    registry.register(ToolSpec(
        name="text_jaccard",
        function=_jaccard_handler,
        description="Compute Jaccard similarity using n-grams (default bigrams).",
        parameters={
            "type": "object",
            "properties": {
                "s1": {"type": "string", "description": "First string"},
                "s2": {"type": "string", "description": "Second string"},
                "n": {"type": "integer", "description": "N-gram size (default 2)"},
            },
            "required": ["s1", "s2"]
        },
        returns="Similarity coefficient (0 to 1)",
        examples=[
            {"input": {"s1": "hello", "s2": "hallo"}, "output": 0.5},
        ],
        domain="text",
        tags=["string", "similarity", "jaccard", "ngram"],
    ))

    registry.register(ToolSpec(
        name="text_longest_common_subsequence",
        function=lambda args: longest_common_subsequence(*_parse_two_strings(args)),
        description="Compute length of longest common subsequence (LCS).",
        parameters={
            "type": "object",
            "properties": {
                "s1": {"type": "string", "description": "First string"},
                "s2": {"type": "string", "description": "Second string"},
            },
            "required": ["s1", "s2"]
        },
        returns="LCS length as integer",
        examples=[
            {"input": {"s1": "ABCDGH", "s2": "AEDFHR"}, "output": 3},
        ],
        domain="text",
        tags=["string", "subsequence", "lcs"],
    ))

    def _ngram_handler(args):
        s1, s2 = _parse_two_strings(args)
        n = args.get("n", 2) if isinstance(args, dict) else 2
        return ngram_similarity(s1, s2, n)

    registry.register(ToolSpec(
        name="text_ngram_similarity",
        function=_ngram_handler,
        description="Compute n-gram overlap similarity between two strings.",
        parameters={
            "type": "object",
            "properties": {
                "s1": {"type": "string", "description": "First string"},
                "s2": {"type": "string", "description": "Second string"},
                "n": {"type": "integer", "description": "N-gram size (default 2)"},
            },
            "required": ["s1", "s2"]
        },
        returns="Similarity score (0 to 1)",
        examples=[
            {"input": {"s1": "hello world", "s2": "hello there"}, "output": 0.5},
        ],
        domain="text",
        tags=["string", "similarity", "ngram"],
    ))

    # Readability Tools
    registry.register(ToolSpec(
        name="text_syllable_count",
        function=lambda args: syllable_count(_parse_text(args) if isinstance(args, (str, dict)) else args),
        description="Estimate syllable count for a word.",
        parameters={
            "type": "object",
            "properties": {
                "word": {"type": "string", "description": "Word to analyze"},
            },
            "required": ["word"]
        },
        returns="Syllable count as integer",
        examples=[
            {"input": {"word": "beautiful"}, "output": 3},
            {"input": {"word": "the"}, "output": 1},
        ],
        domain="text",
        tags=["syllable", "word", "phonetics"],
    ))

    registry.register(ToolSpec(
        name="text_flesch_kincaid",
        function=lambda args: flesch_kincaid_grade(_parse_text(args)),
        description="Calculate Flesch-Kincaid Grade Level - approximate US school grade needed to understand text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"]
        },
        returns="Grade level as float",
        examples=[
            {"input": {"text": "The cat sat on the mat."}, "output": 1.9},
        ],
        domain="text",
        tags=["readability", "grade", "flesch"],
    ))

    registry.register(ToolSpec(
        name="text_gunning_fog",
        function=lambda args: gunning_fog_index(_parse_text(args)),
        description="Calculate Gunning Fog Index - years of education needed to understand text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"]
        },
        returns="Fog index as float",
        examples=[
            {"input": {"text": "The cat sat on the mat."}, "output": 2.4},
        ],
        domain="text",
        tags=["readability", "fog", "complexity"],
    ))

    def _word_freq_handler(args):
        text = _parse_text(args)
        n = args.get("n", args.get("top_n", 10)) if isinstance(args, dict) else 10
        return word_frequency(text, n)

    registry.register(ToolSpec(
        name="text_word_frequency",
        function=_word_freq_handler,
        description="Get most frequent words in text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "n": {"type": "integer", "description": "Number of top words (default 10)"},
            },
            "required": ["text"]
        },
        returns="List of (word, count) tuples",
        examples=[
            {"input": {"text": "the cat and the dog", "n": 3}, "output": [("the", 2), ("cat", 1), ("and", 1)]},
        ],
        domain="text",
        tags=["frequency", "words", "count"],
    ))

    registry.register(ToolSpec(
        name="text_word_count",
        function=lambda args: word_count(_parse_text(args)),
        description="Count words, characters, sentences, and paragraphs in text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"]
        },
        returns="Dict with various counts",
        examples=[
            {"input": {"text": "Hello world."}, "output": {"words": 2, "characters": 12, "sentences": 1}},
        ],
        domain="text",
        tags=["count", "words", "characters", "sentences"],
    ))

    # Utility Tools
    registry.register(ToolSpec(
        name="text_checksum_luhn",
        function=lambda args: checksum_luhn(_parse_text(args)),
        description="Validate a number using Luhn algorithm (credit cards, etc).",
        parameters={
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "Number string to validate"},
            },
            "required": ["number"]
        },
        returns="Boolean - True if valid",
        examples=[
            {"input": {"number": "4532015112830366"}, "output": True},
        ],
        domain="text",
        tags=["luhn", "checksum", "validate", "credit card"],
    ))

    registry.register(ToolSpec(
        name="text_isbn_check",
        function=lambda args: isbn_validate(_parse_text(args)),
        description="Validate an ISBN-10 or ISBN-13.",
        parameters={
            "type": "object",
            "properties": {
                "isbn": {"type": "string", "description": "ISBN string"},
            },
            "required": ["isbn"]
        },
        returns="Boolean - True if valid",
        examples=[
            {"input": {"isbn": "978-0-306-40615-7"}, "output": True},
        ],
        domain="text",
        tags=["isbn", "validate", "book"],
    ))

    registry.register(ToolSpec(
        name="text_roman_numeral",
        function=lambda args: roman_numeral(args.get("value", list(args.values())[0]) if isinstance(args, dict) else args),
        description="Convert between Roman numerals and integers.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"description": "Integer or Roman numeral string"},
            },
            "required": ["value"]
        },
        returns="Converted value (string if input was int, int if input was Roman)",
        examples=[
            {"input": {"value": 2024}, "output": "MMXXIV"},
            {"input": {"value": "XLII"}, "output": 42},
        ],
        domain="text",
        tags=["roman", "numeral", "convert"],
    ))

    def _base_convert_handler(args):
        if isinstance(args, dict):
            num = args.get("number", args.get("n"))
            fb = args.get("from_base", args.get("from", 10))
            tb = args.get("to_base", args.get("to", 10))
            return base_convert(str(num), fb, tb)
        raise ValueError("Expected {number, from_base, to_base}")

    registry.register(ToolSpec(
        name="text_base_convert",
        function=_base_convert_handler,
        description="Convert number between bases (2, 8, 10, 16).",
        parameters={
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "Number string in source base"},
                "from_base": {"type": "integer", "description": "Source base (2, 8, 10, 16)"},
                "to_base": {"type": "integer", "description": "Target base (2, 8, 10, 16)"},
            },
            "required": ["number", "from_base", "to_base"]
        },
        returns="Number string in target base",
        examples=[
            {"input": {"number": "255", "from_base": 10, "to_base": 16}, "output": "FF"},
            {"input": {"number": "1010", "from_base": 2, "to_base": 10}, "output": "10"},
        ],
        domain="text",
        tags=["base", "convert", "binary", "hex", "octal"],
    ))
