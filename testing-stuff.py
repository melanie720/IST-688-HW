import os
import pymupdf
from bs4 import BeautifulSoup
import re, tiktoken

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, "html-files-test")

encoding = tiktoken.get_encoding("cl100k_base")

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "html.parser")

    # Newline separator keeps each HTML element on its own line.
    text = soup.get_text(separator="\n")
    text = text.replace("\r", "\n")
    text = re.sub(r'[ \t]+', ' ', text)

    # Each line is one element's text -- the only places we are allowed to cut.
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # If a page came through as one unbroken line, cut on sentence endings instead.
    if len(lines) < 2:
        lines = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    # Too short to cut anywhere safe, so it stays a single chunk.
    if len(lines) < 2:
        return [" ".join(lines)] if lines else []

    # Walk the lines until the running token count reaches the halfway mark.
    counts = [len(encoding.encode(line)) for line in lines]
    half = sum(counts) / 2
    running = 0
    cut = len(lines) - 1

    for i, count in enumerate(counts):
        running += count
        if running >= half:
            cut = i + 1
            break

    # Keep the cut inside the list so neither chunk can come out empty.
    cut = min(cut, len(lines) - 1)

    print([" ".join(lines[:cut]), " ".join(lines[cut:])])

extract_text_from_html(os.path.join(target_path, "syracuse.campuslabs.com_engage_organization_beta-beta-beta-biology-honors-fraternity.html"))