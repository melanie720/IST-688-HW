"""
Measure cost and speed across four models on the same URL + summary type.

Run this from the terminal, NOT inside Streamlit:
    python measure_models.py

Requires OPENAI_API_KEY and GEMINI_API_KEY to be set in the environment.
"""

import csv
import os
import statistics
import sys
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# CONFIG -- edit these before running
# ---------------------------------------------------------------------------

URL = "https://scriptshadow.net/screenwriting-article-why-breaking-bad-is-so-breaking-good/"
LANGUAGE = "English"
RUNS_PER_TEST = 3

# Each summary type is one test. Every model runs every test.
SUMMARY_TYPES = [
    "100 words",
    "2 connecting paragraphs",
    "5 bullet points",
]

# USD per 1,000,000 tokens.
# OpenAI rates: https://platform.openai.com/pricing
# Gemini rates: https://ai.google.dev/gemini-api/docs/pricing
# Prices retrieved on: 9/6/26
# NOTE: gemini-3.8-flash is on promotional pricing through Dec 2026,
# after which it goes to 1.50 / 7.50.
MODELS = {
    "gpt-5.4-nano":           {"provider": "openai", "in": 0.20, "out": 1.25},
    "gpt-5.6-terra":          {"provider": "openai", "in": 2.00, "out": 12.00},
    "gemini-3.8-flash":       {"provider": "gemini", "in": 0.75, "out": 3.75},
    "gemini-3.1-pro-preview": {"provider": "gemini", "in": 2.00, "out": 12.00},
}

# ---------------------------------------------------------------------------


def read_url_content(url):
    """Pull the text out of the page once, so every model gets an identical string."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SummarizerBenchmark/1.0)"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


def call_openai(client, model_id, prompt):
    """Returns (input_tokens, output_tokens, answer_text)."""
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
        resp.choices[0].message.content or "",
    )


def call_gemini(client, model_id, prompt):
    """Returns (input_tokens, output_tokens, answer_text).

    Gemini bills thinking tokens as output, so they get folded into the
    output count -- otherwise the Pro model looks cheaper than it is.
    """
    resp = client.models.generate_content(
    model=model_id,
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="LOW")
    ))
    usage = resp.usage_metadata
    thinking = getattr(usage, "thoughts_token_count", None) or 0
    output = getattr(usage, "candidates_token_count", None) or 0
    return (usage.prompt_token_count, output + thinking, resp.text or "")


def main():
    openai_client = OpenAI(api_key = st.secrets.OPENAI_API_KEY)
    gemini_client = genai.Client(api_key = st.secrets.GEMINI_API_KEY)

    try:
        document = read_url_content(URL)
    except requests.RequestException as e:
        sys.exit(f"Could not read {URL} -- {type(e).__name__}: {e}")

    # Save the extracted text so you can confirm the page scraped cleanly.
    with open("page.txt", "w") as f:
        f.write(document)
    print(f"Extracted {len(document)} characters to page.txt\n")

    rows = []
    answers = []

    for model_id, cfg in MODELS.items():
        print(f"=== {model_id} ===")

        for summary_type in SUMMARY_TYPES:
            print(f"  [{summary_type}]")
            prompt = (
                f"Here's the text from a URL: {document}. "
                f"Provide a summary in language {LANGUAGE} "
                f"and make sure it is {summary_type}."
            )
            latencies = []

            for run in range(1, RUNS_PER_TEST + 1):
                try:
                    start = time.perf_counter()
                    if cfg["provider"] == "openai":
                        in_tok, out_tok, text = call_openai(openai_client, model_id, prompt)
                    else:
                        in_tok, out_tok, text = call_gemini(gemini_client, model_id, prompt)
                    elapsed = time.perf_counter() - start
                except Exception as e:
                    # Report the real error instead of guessing at the cause.
                    print(f"    run {run}: FAILED -- {type(e).__name__}: {e}\n")
                    break

                cost = (
                    (in_tok / 1_000_000) * cfg["in"]
                    + (out_tok / 1_000_000) * cfg["out"]
                )
                latencies.append(elapsed)

                print(
                    f"    run {run}: {elapsed:6.2f}s  "
                    f"in={in_tok:>6}  "
                    f"out={out_tok:>5}  "
                    f"${cost:.6f}"
                )

                rows.append({
                    "model": model_id,
                    "provider": cfg["provider"],
                    "summary_type": summary_type,
                    "run": run,
                    "seconds": round(elapsed, 3),
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    "words_out": len(text.split()),
                    "cost_usd": round(cost, 6),
                })

                # Keep only the first run's answer for quality scoring.
                if run == 1:
                    answers.append((model_id, summary_type, text))

            if latencies:
                print(f"    median: {statistics.median(latencies):.2f}s\n")

    if not rows:
        sys.exit("No successful runs. Check the errors above.")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open("answers.txt", "w") as f:
        for model_id, summary_type, text in answers:
            header = f"{model_id} -- {summary_type}"
            f.write(f"{'=' * 70}\n{header}\n{'=' * 70}\n\n{text}\n\n")

    # Average cost per run, so you can see what the advanced tier actually costs.
    print("=== AVERAGE COST PER RUN ===")
    for model_id in MODELS:
        model_rows = [r for r in rows if r["model"] == model_id]
        if not model_rows:
            print(f"  {model_id:<24} all runs failed")
            continue
        avg_cost = statistics.mean(r["cost_usd"] for r in model_rows)
        avg_sec = statistics.mean(r["seconds"] for r in model_rows)
        avg_words = statistics.mean(r["words_out"] for r in model_rows)
        print(
            f"  {model_id:<24} ${avg_cost:.6f}   "
            f"{avg_sec:5.2f}s   {avg_words:5.0f} words"
        )

    print("\nWrote results.csv and answers.txt")


if __name__ == "__main__":
    main()