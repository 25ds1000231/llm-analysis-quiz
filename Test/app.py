import os
import time
from json import JSONDecodeError
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber

# ---------------------------------------------------------
# Environment & app setup
# ---------------------------------------------------------

load_dotenv()
EXPECTED_SECRET = os.getenv("QUIZ_SECRET")

app = FastAPI()


@app.post("/quiz")
async def quiz_handler(request: Request):
    """
    Main entry point called by the evaluation server.
    Must:
    - Check JSON validity (400 on invalid)
    - Check secret (403 on mismatch)
    - Run quiz solving loop (200 on success)
    """
    # 1. Parse JSON body
    try:
        payload = await request.json()
    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")

    if not email or not secret or not url:
        raise HTTPException(status_code=400, detail="Missing email, secret or url")

    # 2. Secret check
    if EXPECTED_SECRET is None:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: no EXPECTED_SECRET",
        )
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 3. Run quiz loop with 3-minute deadline
    deadline = time.time() + 3 * 60

    result = await run_quiz_loop(
        email=email,
        secret=secret,
        start_url=url,
        deadline=deadline,
    )

    return JSONResponse(content=result, status_code=200)


# ---------------------------------------------------------
# Quiz loop
# ---------------------------------------------------------

async def run_quiz_loop(
    email: str,
    secret: str,
    start_url: str,
    deadline: float,
) -> Dict[str, Any]:
    """
    Repeatedly:
    - fetch quiz URL
    - solve it
    - submit answer
    until:
    - no new URL, or
    - deadline hit.
    """
    current_url = start_url
    last_response: Dict[str, Any] = {}

    while current_url and time.time() < deadline:
        quiz_html = await fetch_quiz_page(current_url)

        answer, submit_url, extra_payload = await solve_quiz_page(
            quiz_html=quiz_html,
            quiz_url=current_url,
            email=email,
            secret=secret,
        )

        submit_resp = await submit_answer(
            submit_url=submit_url,
            email=email,
            secret=secret,
            quiz_url=current_url,
            answer=answer,
            extra=extra_payload,
        )

        last_response = submit_resp

        # If wrong and no next URL, we may try one more refinement if time left
        if submit_resp.get("correct") is False and "url" not in submit_resp:
            if time.time() + 20 < deadline:  # small buffer
                answer, submit_url, extra_payload = await solve_quiz_page(
                    quiz_html=quiz_html,
                    quiz_url=current_url,
                    email=email,
                    secret=secret,
                )
                submit_resp = await submit_answer(
                    submit_url=submit_url,
                    email=email,
                    secret=secret,
                    quiz_url=current_url,
                    answer=answer,
                    extra=extra_payload,
                )
                last_response = submit_resp

        # Move to next quiz if provided
        current_url = submit_resp.get("url")

    return {
        "message": "Quiz loop finished (deadline reached or no new url).",
        "last_response": last_response,
    }


# ---------------------------------------------------------
# Fetch quiz page with Playwright
# ---------------------------------------------------------

async def fetch_quiz_page(url: str) -> str:
    """
    Load the quiz URL in a headless browser (Chromium) so that
    any JavaScript (e.g. atob) runs, then return the final HTML.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        await browser.close()
        return html


# ---------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------

def extract_download_link(quiz_html: str) -> Optional[str]:
    """
    From the rendered HTML, find a link that looks like the 'Download file' link.
    We:
    - look for <a> tags,
    - pick the first whose text contains 'download' or 'file'.
    """
    soup = BeautifulSoup(quiz_html, "html.parser")
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").lower()
        if "download" in text or "file" in text:
            return a["href"]
    return None


def detect_submit_url(quiz_html: str) -> Optional[str]:
    """
    For Q834-style questions, the page text usually says:
    'Post your answer to https://example.com/submit with this JSON payload:'
    We'll search for 'http' and 'submit' in the visible text.
    """
    soup = BeautifulSoup(quiz_html, "html.parser")
    text = soup.get_text("\n", strip=True)

    # Very simple heuristic: split by whitespace and pick tokens starting with http
    for token in text.split():
        if token.startswith("http") and "submit" in token:
            return token.strip().strip(".,);")
    return None


# ---------------------------------------------------------
# Core quiz solving
# ---------------------------------------------------------

async def solve_quiz_page(
    quiz_html: str,
    quiz_url: str,
    email: str,
    secret: str,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Decide how to solve the quiz page.
    For now:
    - If it looks like a 'sum of "value" column' type PDF question, use a specialized solver.
    - Otherwise, use a generic fallback planner.
    Returns: (answer, submit_url, extra_payload)
    """
    soup = BeautifulSoup(quiz_html, "html.parser")
    text = soup.get_text("\n", strip=True).lower()

    # 1. Detect Q834-style question
    if 'sum of the "value" column' in text or "sum of the 'value' column" in text:
        download_url = extract_download_link(quiz_html)
        submit_url = detect_submit_url(quiz_html)

        if not download_url or not submit_url:
            # Fallback: generic planner
            return await generic_llm_planner(quiz_html, quiz_url, email, secret)

        # Compute sum from the PDF
        answer = compute_sum_value_column_from_pdf(download_url)

        # extra_payload is empty for this simple case
        return answer, submit_url, {}

    # 2. Fallback: unknown question type → generic planner
    return await generic_llm_planner(quiz_html, quiz_url, email, secret)


def compute_sum_value_column_from_pdf(download_url: str) -> float:
    """
    Download a PDF file and compute the sum of the 'value' column
    in the table on page 2 (index 1).
    This assumes a well-formed table with a column named 'value'.
    """
    resp = requests.get(download_url)
    resp.raise_for_status()

    from io import BytesIO

    pdf_bytes = BytesIO(resp.content)
    total = 0.0

    with pdfplumber.open(pdf_bytes) as pdf:
        # page 2 is index 1
        if len(pdf.pages) < 2:
            raise RuntimeError("PDF does not have 2 pages as expected")

        page2 = pdf.pages[1]
        tables = page2.extract_tables()
        if not tables:
            raise RuntimeError("No tables found on page 2")

        # Take first table as DataFrame
        table = tables[0]
        df = pd.DataFrame(table[1:], columns=table[0])  # row[0] is header row
        # Normalize column names to lower case
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "value" not in df.columns:
            raise RuntimeError("No 'value' column found in table")

        # Convert to numeric, coerce errors
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        total = float(df["value"].sum())

    return total


# ---------------------------------------------------------
# Answer submission
# ---------------------------------------------------------

async def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Submit the answer payload to the quiz submit_url.
    """
    payload: Dict[str, Any] = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }
    if extra:
        payload.update(extra)

    resp = requests.post(submit_url, json=payload, timeout=30)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


# ---------------------------------------------------------
# Generic fallback planner (no real LLM yet)
# ---------------------------------------------------------

async def generic_llm_planner(
    quiz_html: str,
    quiz_url: str,
    email: str,
    secret: str,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Very simple fallback when the quiz type is unknown.
    For now:
    - Try to detect submit_url from the HTML.
    - Return a dummy answer (0) so the pipeline doesn't crash.

    Later you can replace this with a real LLM-based planner.
    """
    submit_url = detect_submit_url(quiz_html)
    if not submit_url:
        # As a last resort, submit back to the quiz URL itself.
        submit_url = quiz_url

    dummy_answer = 0
    extra_payload: Dict[str, Any] = {}
    return dummy_answer, submit_url, extra_payload
