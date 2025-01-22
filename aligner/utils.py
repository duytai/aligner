from typing import Dict
import re

def trim_parse_form(text: str) -> Dict[str, str]:
    """
    Read text file and convert it into `dict`
    The input should have the following format:
    <key1>: <value1>
    <key2>: <value2>
    """
    parsed = {}
    keywords, pairs = [], []
    lines = text.splitlines()
    for line in lines:
        matches = re.match(r'([^:])+\s*:', line)
        if matches:
            kw = matches.group(0)
            keywords.append(kw)
    starts = [text.index(kw) + len(kw) for kw in keywords]
    ends = [text.index(kw) for kw in keywords[1:]] + [len(text)]
    for kw, start, end in zip(keywords, starts, ends):
        key = kw[:-1].strip().lower()
        val = text[start:end].strip()
        parsed[key] = val
    return parsed

