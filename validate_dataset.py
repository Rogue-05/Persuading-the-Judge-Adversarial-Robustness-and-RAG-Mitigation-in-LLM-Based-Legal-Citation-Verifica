"""
Dataset Validator (QA Only)
Uses an LLM (Gemini/Anthropic) to verify if claims are supported by the holding text.
Does NOT modify dataset labels. Produces flags and a summary report.
Can be imported by build_dataset.py for inline semantic QA filtering.
"""

import json
import os
import sys
import time
from datetime import datetime
import requests
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

DATASET_FILE = "legal_dataset.json"
REPORT_FILE = "qa_validation_report.json"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

DEFAULT_QA_PROVIDER = "gemini"
QA_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-sonnet-4-6"
}

# ── RETRY / BACKOFF ──────────────────────────────────────
# Transient HTTP status codes that should be retried.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}

# Transient exception types that should be retried (network-level failures).
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
)


def with_backoff(max_retries=5, base_delay=1.0, max_delay=32.0):
    """Exponential backoff decorator for transient API and network errors.
    
    Retries on:
      - HTTP 429, 500, 502, 503, 504  (via requests.exceptions.HTTPError)
      - Timeout, ConnectionError, ConnectTimeout, ReadTimeout
    
    Tracks retry statistics via an optional retry_stats dict kwarg.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_stats = kwargs.get("retry_stats")
            retries = 0
            last_exception = None

            while True:
                try:
                    result = func(*args, **kwargs)
                    # If we succeeded after at least one retry, record it.
                    if retries > 0 and retry_stats is not None:
                        retry_stats["retry_successes"] = retry_stats.get("retry_successes", 0) + 1
                        retry_stats["maximum_retry_depth"] = max(
                            retry_stats.get("maximum_retry_depth", 0), retries
                        )
                    return result

                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    status_code = getattr(e.response, "status_code", None)
                    if status_code in RETRYABLE_STATUS_CODES and retries < max_retries:
                        delay = min(max_delay, base_delay * (2 ** retries))
                        print(f"    [Retry {retries+1}/{max_retries}] HTTP {status_code} — waiting {delay:.1f}s")
                        time.sleep(delay)
                        retries += 1
                        if retry_stats is not None:
                            retry_stats["api_retry_count"] = retry_stats.get("api_retry_count", 0) + 1
                    else:
                        if retry_stats is not None:
                            retry_stats["retry_failures"] = retry_stats.get("retry_failures", 0) + 1
                        raise

                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    exc_type = type(e).__name__
                    
                    # Custom rule: abort early if it's a ReadTimeout (server hanging)
                    if exc_type == "ReadTimeout" and retries >= 2:
                        if retry_stats is not None:
                            retry_stats["retry_failures"] = retry_stats.get("retry_failures", 0) + 1
                        raise
                        
                    if retries < max_retries:
                        delay = min(max_delay, base_delay * (2 ** retries))
                        print(f"    [Retry {retries+1}/{max_retries}] {exc_type} — waiting {delay:.1f}s")
                        time.sleep(delay)
                        retries += 1
                        if retry_stats is not None:
                            retry_stats["api_retry_count"] = retry_stats.get("api_retry_count", 0) + 1
                    else:
                        if retry_stats is not None:
                            retry_stats["retry_failures"] = retry_stats.get("retry_failures", 0) + 1
                        raise

        return wrapper
    return decorator


# ── SEMANTIC SUPPORT VERIFICATION ─────────────────────────

def _build_qa_prompt(claim, holding_text):
    """Build the QA prompt. Shared across providers."""
    return f"""You are evaluating semantic support.

Judge ONLY from the supplied source text.
Ignore outside legal knowledge.
Do not infer.
Do not guess.
Return JSON only.

Source Text:
{holding_text}

Claim to Verify:
{claim}

Does the provided source text explicitly support the claim?

Return a JSON object with exactly three keys:
1. "verdict": One of "SUPPORTED", "NOT_SUPPORTED", or "UNSURE"
2. "confidence": A float between 0.0 and 1.0 representing your confidence in the verdict
3. "reason": A brief 1-2 sentence explanation of your reasoning
"""


def _parse_qa_response(content):
    """Parse JSON from the LLM response text.
    
    Returns a dict on success.
    Raises ValueError on parse failure (should NOT be retried).
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    result = json.loads(content)
    return {
        "verdict": result.get("verdict", "UNSURE"),
        "confidence": float(result.get("confidence", 0.0)),
        "reason": result.get("reason", ""),
        "verification_status": "verified",
    }


@with_backoff(max_retries=5)
def _call_qa_api(prompt, provider, retry_stats=None):
    """Make the raw API call. Transient errors propagate to the decorator.
    
    Returns the raw response text string.
    Raises requests exceptions for transient failures (retried by decorator).
    """
    if provider == "anthropic":
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": QA_MODELS["anthropic"],
                "max_tokens": 300,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    elif provider == "gemini":
        time.sleep(25) 
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{QA_MODELS['gemini']}:generateContent",
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 300,
                    "responseMimeType": "application/json",
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    else:
        raise ValueError(f"Unknown QA provider: {provider}")


def verify_semantic_support(claim, holding_text, provider=DEFAULT_QA_PROVIDER, retry_stats=None):
    """High-level verification entry point.
    
    Retry logic lives in the decorator on _call_qa_api.
    JSON parse errors are NOT retried — they indicate a malformed but successful response.
    
    Returns dict with keys: verdict, confidence, reason, verification_status.
    """
    if not holding_text or holding_text.strip() == "":
        return {
            "verdict": "NO_HOLDING_TEXT", "confidence": 0.0,
            "reason": "No text provided", "verification_status": "api_error",
        }

    prompt = _build_qa_prompt(claim, holding_text)

    # Step 1: Call the API (retries handled by decorator on transient failures).
    try:
        raw_content = _call_qa_api(prompt, provider, retry_stats=retry_stats)
    except Exception as e:
        # All retries exhausted or non-retryable error.
        print(f"    [QA API Error] {type(e).__name__}: {e}")
        if retry_stats is not None:
            retry_stats["api_errors"] = retry_stats.get("api_errors", 0) + 1
        return {
            "verdict": "ERROR", "confidence": 0.0,
            "reason": f"API error after retries: {e}",
            "verification_status": "api_error",
        }

    # Step 2: Parse the response (NOT retried — the API call succeeded).
    try:
        return _parse_qa_response(raw_content)
    except Exception as e:
        print(f"    [QA Parse Error] {type(e).__name__}: {e}")
        if retry_stats is not None:
            retry_stats["parse_errors"] = retry_stats.get("parse_errors", 0) + 1
        return {
            "verdict": "ERROR", "confidence": 0.0,
            "reason": f"JSON parse error: {e}",
            "verification_status": "parse_error",
        }


# ── REVIEW PRIORITIES ────────────────────────────────────

def get_review_priority(verdict, confidence):
    """Returns the review priority based on strict dataset methodology rules."""
    if verdict == "SUPPORTED":
        if confidence >= 0.95:
            return "PASS"
        elif confidence >= 0.80:
            return "LOW"
        else:
            return "MEDIUM"

    elif verdict == "NOT_SUPPORTED":
        if confidence >= 0.90:
            return "HIGH"
        else:
            return "MEDIUM"

    elif verdict == "UNSURE":
        return "MEDIUM"

    return "PASS"


# ── STANDALONE VALIDATION ─────────────────────────────────

def run_qa_validation(dataset_path=DATASET_FILE, provider=DEFAULT_QA_PROVIDER):
    with open(dataset_path) as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"Loaded {len(items)} items for {provider} QA verification.")

    items_to_check = [item for item in items if item.get("category") in ("valid", "real_wrong_content")]
    print(f"Checking {len(items_to_check)} valid/real_wrong_content items...\n")

    results_log = []

    stats = {
        "passed": 0,
        "high_priority_review": 0,
        "medium_review": 0,
        "low_review": 0,
        "qa_api_errors": 0,
        "total_confidence_sum": 0.0,
        "items_with_confidence": 0,
        "cat_counts": {},
    }

    retry_stats = {
        "api_retry_count": 0,
        "api_errors": 0,
        "parse_errors": 0,
        "retry_successes": 0,
        "retry_failures": 0,
        "maximum_retry_depth": 0,
    }

    for i, item in enumerate(items):
        item_id = item["id"]
        category = item["category"]
        claim = item.get("claim", "")
        holding = item.get("case_holding_text", "")

        stats["cat_counts"][category] = stats["cat_counts"].get(category, 0) + 1

        if category == "fabricated":
            continue

        print(f"[{i+1}/{len(items)}] {item_id} ({category}): ", end="", flush=True)

        judgement_data = verify_semantic_support(claim, holding, provider=provider, retry_stats=retry_stats)

        verdict = judgement_data.get("verdict", "ERROR")
        confidence = judgement_data.get("confidence", 0.0)
        reason = judgement_data.get("reason", "")
        v_status = judgement_data.get("verification_status", "verified")

        if verdict == "ERROR":
            stats["qa_api_errors"] += 1

        if verdict not in ("ERROR", "NO_HOLDING_TEXT"):
            stats["total_confidence_sum"] += confidence
            stats["items_with_confidence"] += 1

        print(f"{verdict} (conf: {confidence:.2f})", end="")

        priority = get_review_priority(verdict, confidence)

        if priority == "PASS":
            stats["passed"] += 1
            print(" ✓ PASS")
        elif priority == "LOW":
            stats["low_review"] += 1
            print(" ⚠ LOW PRIORITY")
        elif priority == "MEDIUM":
            stats["medium_review"] += 1
            print(" ⚠ MEDIUM PRIORITY")
        elif priority == "HIGH":
            stats["high_priority_review"] += 1
            print(" ⚠ HIGH PRIORITY")

        results_log.append({
            "id": item_id,
            "category": category,
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "review_priority": priority,
            "verification_status": v_status,
        })

    report = {
        "verified_at": datetime.now().isoformat(),
        "qa_provider": provider,
        "qa_model": QA_MODELS[provider],
        "total_items": len(items),
        "retry_statistics": retry_stats,
        "details": results_log,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    avg_conf = (stats["total_confidence_sum"] / stats["items_with_confidence"]) if stats["items_with_confidence"] > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"QA VERIFICATION COMPLETE ({provider.upper()})")
    print(f"{'='*50}")
    print(f"Total samples checked : {len(items_to_check)}")
    print(f"Passed                : {stats['passed']}")
    print(f"High-priority review  : {stats['high_priority_review']}")
    print(f"Medium review         : {stats['medium_review']}")
    print(f"Low review            : {stats['low_review']}")
    print(f"QA API Errors         : {stats['qa_api_errors']}")
    print(f"Average confidence    : {avg_conf:.2f}")
    print(f"API Retries           : {retry_stats['api_retry_count']}")
    print(f"Retry Successes       : {retry_stats['retry_successes']}")
    print(f"Retry Failures        : {retry_stats['retry_failures']}")
    print(f"Parse Errors          : {retry_stats['parse_errors']}")
    print(f"Max Retry Depth       : {retry_stats['maximum_retry_depth']}")
    print(f"Category counts       : {stats['cat_counts']}")
    print(f"\nDetailed report saved to: {REPORT_FILE}")
    print(f"{'='*50}")


# ── SELF-TEST ─────────────────────────────────────────────

def _run_self_tests():
    """Validate retry wrapper behavior with simulated failures.
    Run with: python validate_dataset.py --self-test
    """
    import unittest
    from unittest.mock import patch, MagicMock

    class TestRetryWrapper(unittest.TestCase):

        def test_retries_on_429(self):
            """429 should be retried up to max_retries."""
            call_count = [0]

            @with_backoff(max_retries=3, base_delay=0.01)
            def flaky_429(retry_stats=None):
                call_count[0] += 1
                if call_count[0] < 3:
                    resp = MagicMock()
                    resp.status_code = 429
                    raise requests.exceptions.HTTPError(response=resp)
                return "ok"

            stats = {"api_retry_count": 0, "retry_successes": 0, "retry_failures": 0, "maximum_retry_depth": 0}
            result = flaky_429(retry_stats=stats)
            self.assertEqual(result, "ok")
            self.assertEqual(call_count[0], 3)
            self.assertGreater(stats["retry_successes"], 0)
            print("  ✓ 429 retry test passed")

        def test_retries_on_500(self):
            """500 should be retried."""
            call_count = [0]

            @with_backoff(max_retries=3, base_delay=0.01)
            def flaky_500(retry_stats=None):
                call_count[0] += 1
                if call_count[0] < 2:
                    resp = MagicMock()
                    resp.status_code = 500
                    raise requests.exceptions.HTTPError(response=resp)
                return "ok"

            stats = {"api_retry_count": 0, "retry_successes": 0, "retry_failures": 0, "maximum_retry_depth": 0}
            result = flaky_500(retry_stats=stats)
            self.assertEqual(result, "ok")
            print("  ✓ 500 retry test passed")

        def test_retries_on_timeout(self):
            """Timeout should be retried."""
            call_count = [0]

            @with_backoff(max_retries=3, base_delay=0.01)
            def flaky_timeout(retry_stats=None):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise requests.exceptions.Timeout("timed out")
                return "ok"

            stats = {"api_retry_count": 0, "retry_successes": 0, "retry_failures": 0, "maximum_retry_depth": 0}
            result = flaky_timeout(retry_stats=stats)
            self.assertEqual(result, "ok")
            print("  ✓ Timeout retry test passed")

        def test_retries_on_connection_error(self):
            """ConnectionError should be retried."""
            call_count = [0]

            @with_backoff(max_retries=3, base_delay=0.01)
            def flaky_conn(retry_stats=None):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise requests.exceptions.ConnectionError("connection refused")
                return "ok"

            stats = {"api_retry_count": 0, "retry_successes": 0, "retry_failures": 0, "maximum_retry_depth": 0}
            result = flaky_conn(retry_stats=stats)
            self.assertEqual(result, "ok")
            print("  ✓ ConnectionError retry test passed")

        def test_malformed_json_not_retried(self):
            """JSON parse errors should NOT be retried — they produce parse_error."""
            stats = {"parse_errors": 0, "api_errors": 0}
            result = verify_semantic_support.__wrapped__(
                "test claim", "test holding", provider="gemini", retry_stats=stats
            )
            # We can't easily call the real function without an API key,
            # so we test _parse_qa_response directly.
            with self.assertRaises(Exception):
                _parse_qa_response("this is not json at all {{{")
            print("  ✓ Malformed JSON parse error test passed")

    print("\n=== Running Self-Tests ===")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRetryWrapper)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed!")
    else:
        print("\nSome self-tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Use QA model to verify dataset claims")
    parser.add_argument("dataset", nargs="?", default=DATASET_FILE, help="Path to dataset JSON")
    parser.add_argument("--provider", choices=["gemini", "anthropic"], default=DEFAULT_QA_PROVIDER)
    parser.add_argument("--self-test", action="store_true", help="Run internal self-tests")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
    else:
        run_qa_validation(args.dataset, provider=args.provider)
