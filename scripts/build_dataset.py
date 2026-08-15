"""
Legal Citation Verification Dataset Builder (Adversarial Setup)
Builds items for: valid, fabricated, real_wrong_content categories.
Guarantees wrong-content labels by construction using semantic embeddings and inline QA.
Maximizes methodological robustness and reproducibility for ACL/NLLP submission.
Uses low-temperature generation.
"""

# ── MUST be set before ANY imports to prevent TF/PyTorch mutex deadlock ──
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # Suppress TF C++ logs
os.environ["USE_TORCH"] = "1"                      # Force HF to use PyTorch only
os.environ["USE_TF"] = "0"                         # Block TensorFlow from loading

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import numpy as np
import re
from difflib import SequenceMatcher
from datetime import datetime
from dotenv import load_dotenv

# Import our QA verifier and backoff logic
from validate_dataset import verify_semantic_support, with_backoff, QA_MODELS

load_dotenv()

# ── DETERMINISTIC SEEDS ───────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── CONFIG ────────────────────────────────────────────────
COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4"
OUTPUT_FILE = "data/legal_dataset.json" if os.path.exists("data") or os.path.exists("../data") else "legal_dataset.json"

LLM_PROVIDER = "anthropic"
QA_PROVIDER = "gemini"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

COURTLISTENER_API_KEY = os.environ.get("COURTLISTENER_API_KEY", "")

GENERATION_MODEL = "claude-sonnet-4-6"

QA_CONFIDENCE_THRESHOLD = 0.90
GENERATION_TEMPERATURE = 0.0
QA_TEMPERATURE = 0.0

DATASET_BUILDER_VERSION = "2.3.0"
QA_PROMPT_VERSION = "1.2.0"

LEGAL_TOPICS = [
    "employment discrimination",
    "contract breach damages",
    "First Amendment retaliation",
    "search and seizure Fourth Amendment",
    "negligence duty of care",
    "criminal procedure Miranda rights",
    "due process Fourteenth Amendment",
    "intellectual property patent infringement",
    "habeas corpus petition",
    "qualified immunity police",
    "immigration asylum claim",
    "securities fraud insider trading",
    "Eighth Amendment cruel unusual punishment",
    "antitrust monopoly Sherman Act",
    "environmental regulation Clean Air Act",
]

# ── STATS TRACKING ────────────────────────────────────────
build_stats = {
    "retrieved_cases": 0,
    "discarded_snippet_only": 0,
    "discarded_generation_failure": 0,
    "generated_valid_claims": 0,
    "accepted_valid_claims": 0,
    "rejected_valid_not_supported": 0,
    "rejected_valid_unsure": 0,
    "rejected_valid_low_confidence": 0,
    "rejected_valid_api_error": 0,
    "generated_wrong_claims": 0,
    "generated_fabricated_candidates": 0,
    "accepted_wrong_claims": 0,
    "rejected_supported": 0,
    "rejected_unsure": 0,
    "rejected_low_confidence": 0,
    "rejected_api_error": 0,
    "collision_rejection_count": 0,
    "accepted_syllabi": 0,
    "accepted_headnotes": 0,
    "accepted_opinion_extracts": 0,
    "accepted_html_llm_holdings": 0,
    "rejected_metadata_summaries": 0,
    "sub_opinions_checked_total": 0,
    "qa_confidences": [],
    "holding_source_distribution": {},
    "holding_trigger_distribution": {},
    "source_word_counts": [],
    "generation_failures": [],
    "adversarial_strategy_distribution": {},
    "difficulty_distribution": {},
}

# Shared retry statistics dict — passed to every retryable call.
retry_stats = {
    "api_retry_count": 0,
    "api_errors": 0,
    "parse_errors": 0,
    "retry_successes": 0,
    "retry_failures": 0,
    "maximum_retry_depth": 0,
}


# ── GLOBALS ───────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
if COURTLISTENER_API_KEY:
    SESSION.headers.update({"Authorization": f"Token {COURTLISTENER_API_KEY}"})

# ── COURTLISTENER SEARCH ──────────────────────────────────
@with_backoff(max_retries=8, base_delay=2.0, max_delay=60.0)
def search_cases(query, max_results=20, retry_stats=None):
    url = f"{COURTLISTENER_BASE}/search/"
    params = {
        "q": query,
        "type": "o",
        "order_by": "score desc",
        "stat_Published": "on",
    }
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "case_name": r.get("caseName"),
            "citation": r.get("citation", [None])[0] if r.get("citation") else None,
            "cluster_id": r.get("cluster_id"),
        })
    return results


# ── HOLDING EXTRACTION (HTML-based + LLM selection) ───────

HOLDING_PATTERNS = {
    # Strong procedural indicators
    "motion to dismiss": 8,
    "summary judgment": 8,
    "is granted": 8,
    "is denied": 8,
    "without leave to amend": 8,
    "with leave to amend": 8,
    # Strong holding language
    "we hold": 7,
    "we conclude": 7,
    "the court finds": 7,
    "the court concludes": 7,
    "accordingly": 6,
    "therefore": 6,
    # Disposition
    "affirm": 6,
    "reverse": 6,
    "vacate": 6,
    "remand": 6,
    # Merits reasoning
    "failed to establish": 5,
    "failed to state a claim": 5,
    "probable cause": 4,
    "qualified immunity": 3,
}


def _extract_paragraphs_from_opinion(opinion_data):
    """Extract clean paragraph blocks from the best available HTML field.

    Falls back through HTML fields in priority order, then to plain_text.
    Returns a list of paragraph strings (each >= 25 words).
    """
    # Try HTML fields in priority order
    html = None
    for field in ["html_with_citations", "html", "html_lawbox", "html_columbia"]:
        if opinion_data.get(field):
            html = opinion_data[field]
            break

    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        paragraphs = []
        for tag in soup.find_all(["p", "blockquote", "div"]):
            text = tag.get_text(" ", strip=True)
            if len(text.split()) >= 25:
                paragraphs.append(text)
        
        if paragraphs:
            return paragraphs

    # Fallback: split plain_text by double-newlines
    plain = opinion_data.get("plain_text", "")
    if plain:
        return [
            p.strip()
            for p in re.split(r"\n\s*\n", plain)
            if len(p.split()) >= 25
        ]

    return []


def _score_paragraph(text, position_ratio=0.0):
    """Score a paragraph for holding likelihood.

    Args:
        text: The paragraph text.
        position_ratio: 0.0 = start of opinion, 1.0 = end of opinion.
            Paragraphs near the end receive a position boost because
            dispositive rulings typically appear there.

    Returns:
        Integer score (higher = more likely to be a holding).
    """
    t = text.lower()
    score = 0
    for phrase, weight in HOLDING_PATTERNS.items():
        if phrase in t:
            score += weight

    # Position boost: last 30% of opinion gets +3
    if position_ratio >= 0.70:
        score += 3

    return score


def _select_holding_with_llm(candidate_paragraphs):
    """Use the generation LLM to select the best holding paragraphs.

    Sends the top candidate paragraphs and asks the model to return
    only the ones containing the actual legal holding.
    """
    numbered = "\n\n".join(
        f"[Paragraph {i+1}]\n{p}" for i, p in enumerate(candidate_paragraphs)
    )

    prompt = f"""You are given candidate paragraphs extracted from a judicial opinion.

Select the paragraphs that contain the court's actual legal holding or dispositive reasoning.

Ignore:
* factual background
* procedural history
* citations to or quotations of precedent
* general legal standards recited without application

Prefer:
* operative rulings ("granted", "denied", "affirmed", "reversed")
* dispositive reasoning that resolves the dispute
* legal conclusions actually reached by this court
* legal rules applied to the specific facts to reach a result

Return ONLY the original text of the selected paragraphs exactly as written.
Do not summarize. Do not rewrite. Do not explain your choices.
Separate selected paragraphs with a blank line.

Candidate Paragraphs:
{numbered}"""

    result = llm_call(prompt, temperature=0.0, max_tokens=3000, retry_stats=retry_stats)
    return result.strip() if result else None


@with_backoff(max_retries=6, base_delay=5.0, max_delay=120.0)
def get_opinion_text(cluster_id, retry_stats=None):
    """Retrieve and extract the legal holding from a CourtListener opinion.

    Pipeline:
      1. Fetch the cluster and its sub-opinions.
      2. Extract paragraphs from the best available HTML field.
      3. Score paragraphs by legal-holding likelihood.
      4. Send the top 10 candidates to Claude for holding selection.
      5. Return the LLM-selected holding text.

    Returns:
        (holding_text, source_type, trigger) or (None, None, None).
    """
    url = f"{COURTLISTENER_BASE}/clusters/{cluster_id}/"

    resp = SESSION.get(url, timeout=20)
    if resp.status_code in (401, 403):
        print(f"    [WARNING] CourtListener returned {resp.status_code} for cluster {cluster_id} — key may be rate-limited")
        resp.raise_for_status()  # Let call_with_fallback handle key rotation
    resp.raise_for_status()
    data = resp.json()

    if not data.get("sub_opinions"):
        build_stats["discarded_snippet_only"] += 1
        return None, None, None

    # Fetch up to 5 sub-opinions, prefer majority/lead
    fetched_ops = []
    for sub_url in data["sub_opinions"][:5]:
        build_stats["sub_opinions_checked_total"] += 1
        sub_resp = SESSION.get(sub_url, timeout=20)
        if sub_resp.ok:
            sub_data = sub_resp.json()
            op_type = sub_data.get("type", "zzz_unknown")
            fetched_ops.append((op_type, sub_data))

    fetched_ops.sort(key=lambda x: x[0])

    for op_type, opinion_data in fetched_ops:
        paragraphs = _extract_paragraphs_from_opinion(opinion_data)
        if not paragraphs:
            continue

        # Score every paragraph
        total = len(paragraphs)
        scored = []
        for idx, p in enumerate(paragraphs):
            pos_ratio = idx / total if total > 1 else 0.0
            score = _score_paragraph(p, position_ratio=pos_ratio)
            scored.append((score, idx, p))

        # Sort descending by score, take top 10
        scored.sort(key=lambda x: -x[0])
        top_candidates = [p for _score, _idx, p in scored[:10]]

        if not top_candidates:
            continue

        # LLM holding selection
        holding_text = _select_holding_with_llm(top_candidates)

        if holding_text and len(holding_text.split()) >= 20:
            build_stats["accepted_html_llm_holdings"] += 1
            return holding_text, "html_llm_holding", "llm_selected"

    build_stats["discarded_snippet_only"] += 1
    return None, None, None


# ── LLM HELPER ────────────────────────────────────────────
@with_backoff(max_retries=5)
def llm_call(prompt, temperature=0.7, max_tokens=500, retry_stats=None):
    print("    [API] Calling Claude...")
    if LLM_PROVIDER == "anthropic":
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": GENERATION_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["content"]
        return "".join(block["text"] for block in content if block["type"] == "text")
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ── DRAFTING ──────────────────────────────────────────────
def is_bad_generation(text):
    if not text:
        return True
        
    text_lower = text.strip().lower()
    if text_lower == "invalid_source_text":
        return True
        
    words = text.split()
    if len(words) < 15:
        bad_phrases = [
            "i need", "please provide", "what you've provided", 
            "the actual holding", "invalid_source_text"
        ]
        if any(p in text_lower for p in bad_phrases):
            return True
            
    return False


def check_needs_manual_review(claim):
    claim_lower = claim.lower()
    
    # Very short claim
    if len(claim_lower.split()) < 10:
        return True
        
    legal_terms = [
        "must", "may", "required", "sufficient", "insufficient", "liable",
        "jurisdiction", "summary judgment", "prima facie", "dismiss",
        "affirm", "reverse", "hold", "constitute", "establish",
        "employment relationship"
    ]
    
    # Lacks legal terminology
    if not any(term in claim_lower for term in legal_terms):
        return True
        
    return False


def draft_unsupported_claim(holding_text, valid_claim, case_name, topic):
    clear_contradiction = ["Reverse legal rule", "Reverse legal outcome", "Reverse causation", "Reverse required element"]
    qualifier_or_scope = ["Strengthen legal requirement", "Weaken legal requirement", "Remove necessary qualifier", "Introduce unsupported qualifier", "Narrow legal scope", "Broaden legal scope"]
    subtle_reasoning = ["Reverse exception", "Reverse burden of proof"]
    
    cat = random.choices(["clear", "qualifier", "subtle"], weights=[0.3, 0.4, 0.3], k=1)[0]
    if cat == "clear":
        strategy = random.choice(clear_contradiction)
        difficulty = "clear_contradiction"
    elif cat == "qualifier":
        strategy = random.choice(qualifier_or_scope)
        difficulty = "qualifier_or_scope"
    else:
        strategy = random.choice(subtle_reasoning)
        difficulty = "subtle_reasoning"
    
    prompt = f"""You are creating a difficult negative example for a legal citation verification benchmark.

You are given:
1. The original legal holding.
2. The valid supported claim.

Generate ONE legal proposition that is:
* highly similar to the valid claim
* uses the same legal doctrine
* uses nearly identical terminology
* sounds legally plausible
* could realistically fool a reader

BUT
must NOT be supported by the supplied holding.

Rules:
* Do NOT invent a different case.
* Do NOT fabricate new statutes.
* Do NOT change jurisdiction.
* Do NOT change parties.
* Stay within the same legal issue.

The unsupported claim should differ by changing: a legal condition, a causal relationship, a required element, a legal standard, an exception, or the court's conclusion.
The goal is a subtle contradiction.

Modification Strategy to Use: {strategy}
(Apply exactly this one modification to the valid claim to make it unsupported).

Return ONLY the unsupported claim.

Holding Text:
{holding_text}

Valid Claim:
{valid_claim}"""
    
    for _ in range(3):
        claim = llm_call(prompt, temperature=0.0, retry_stats=retry_stats).strip()
        if not is_bad_generation(claim):
            return claim, strategy, difficulty
            
    build_stats["generation_failures"].append({
        "failure_reason": "wrong_claim_generation_failed",
        "source_text": holding_text,
        "case_name": case_name,
        "topic": topic
    })
    return None, None, None


def draft_claim_for_valid_case(holding_text, case_name, topic):
    prompt = f"""Read ONLY the provided legal holding.

Generate ONE sentence describing the central legal proposition decided by the court.

Prioritize:
* legal rules
* legal standards
* legal conclusions
* burdens of proof
* jurisdictional rulings
* required legal elements
* holdings regarding liability or entitlement

Avoid selecting:
* factual background
* procedural history
* party identities
* dates
* locations
* narrative facts

If both factual context and a legal conclusion are present, always choose the legal conclusion.

Requirements:
* one sentence
* no inference
* no extrapolation
* no outside legal knowledge
* no speculation
* no facts not explicitly supported
* If the supplied text does not contain an actual legal holding and instead appears to be keywords, metadata, or an incomplete summary, return EXACTLY: INVALID_SOURCE_TEXT
Do not attempt to infer or guess at a holding that isn't there.

Return only the claim.

Holding Text:
{holding_text}"""

    for _ in range(3):
        claim = llm_call(prompt, temperature=GENERATION_TEMPERATURE, retry_stats=retry_stats).strip()
        if claim.strip().upper() == "INVALID_SOURCE_TEXT":
            build_stats["generation_failures"].append({
                "failure_reason": "invalid_source_text",
                "source_text": holding_text,
                "case_name": case_name,
                "topic": topic
            })
            return claim
        if not is_bad_generation(claim):
            return claim
            
    build_stats["discarded_generation_failure"] += 1
    build_stats["generation_failures"].append({
        "failure_reason": "valid_claim_generation_failed",
        "source_text": holding_text,
        "case_name": case_name,
        "topic": topic
    })
    return None


def draft_fabricated_citation(topic, seen_fab_names):
    circuits = ["1st Cir.", "2nd Cir.", "3rd Cir.", "4th Cir.", "5th Cir.", "6th Cir.", "7th Cir.", "8th Cir.", "9th Cir.", "10th Cir.", "11th Cir.", "D.C. Cir."]
    year = random.randint(1980, 2023)
    circuit = random.choice(circuits)
    
    import string
    rand_letter = random.choice(string.ascii_uppercase)
    
    def_types = ["School District", "Hospital System", "Police Department", "Tech Corporation", "Insurance Company", "State University", "Retail Chain", "Construction Firm", "City Council", "Pharmaceuticals", "Manufacturing Inc.", "County Sheriff", "Transit Authority", "Bank & Trust", "Real Estate LLC"]
    def_type = random.choice(def_types)

    prompt = f"""Invent a plausible-sounding but FAKE US court case citation about "{topic}".
Format: Case Name, Volume Reporter Page ({circuit} {year}) — e.g. "Harmon v. State Board of Licensing, 412 F.3d 88 ({circuit} {year})"
Also write a claim (1-2 sentences) this fake case supposedly supports.

Generate genuinely new case names. Also vary: court, year, plaintiff surname, defendant organization, legal issue, sentence style. Do not produce template-like fabricated cases.
CRITICAL RULE 1: The plaintiff's surname MUST start with the letter '{rand_letter}' and be highly unique.
CRITICAL RULE 2: The defendant MUST be a {def_type}.

Return as JSON only, no other text: {{"citation": "...", "case_name": "...", "claim": "..."}}"""
    for _ in range(3):
        raw = llm_call(prompt, temperature=0.9, retry_stats=retry_stats)
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            res = json.loads(raw[start:end])
            if not is_bad_generation(res.get("claim", "")):
                return res
        except Exception:
            pass
            
    build_stats["discarded_generation_failure"] += 1
    build_stats["generation_failures"].append({
        "failure_reason": "fabricated_generation_failed",
        "source_text": None,
        "case_name": None,
        "topic": topic
    })
    return None


# ── INTEGRITY VALIDATION ──────────────────────────────────
def validate_dataset_integrity(dataset):
    """
    Validates duplicates correctly.
    valid and real_wrong_content can share case_name/citation (intentional swap design).
    Returns a cleaned dataset with invalid items removed.
    """
    ids = set()
    valid_citations = set()
    valid_claims = set()
    fab_citations = set()
    fab_names = set()
    fab_claims = set()

    cleaned_dataset = []
    error_items = []

    for item in dataset:
        item_errors = []
        if item["id"] in ids:
            item_errors.append(f"Duplicate ID {item['id']}")
        ids.add(item["id"])

        cat = item["category"]
        claim = item.get("claim")
        cite = item.get("citation")
        name = item.get("case_name")

        if cat == "valid":
            if cite in valid_citations and cite is not None:
                item_errors.append(f"Duplicate valid citation {cite}")
            if claim in valid_claims:
                item_errors.append(f"Duplicate valid claim text in {item['id']}")
            valid_citations.add(cite)
            valid_claims.add(claim)

        elif cat == "fabricated":
            if cite in fab_citations and cite is not None:
                item_errors.append(f"Duplicate fabricated citation {cite}")
            if name in fab_names and name is not None:
                item_errors.append(f"Duplicate fabricated case name {name}")
            if claim in fab_claims:
                item_errors.append(f"Duplicate fabricated claim text in {item['id']}")
            fab_citations.add(cite)
            fab_names.add(name)
            fab_claims.add(claim)

        req_keys = ["id", "category", "claim", "citation", "case_name", "correct_verdict", "generation_source"]
        for k in req_keys:
            if k not in item:
                item_errors.append(f"Missing field '{k}' in {item['id']}")

        if cat == "valid" and not item.get("source_text_used"):
            item_errors.append(f"Missing source_text_used in valid item {item['id']}")

        if cat == "real_wrong_content":
            if not item.get("verification_verdict"):
                item_errors.append(f"Missing verification_verdict in {item['id']}")
            if not item.get("verification_status"):
                item_errors.append(f"Missing verification_status in {item['id']}")

        if item_errors:
            print(f"  [ERROR] Validation failed for {item['id']}: {', '.join(item_errors)}")
            item["validation_errors"] = item_errors
            error_items.append(item)
        else:
            cleaned_dataset.append(item)

    if error_items:
        print(f"\n=== INTEGRITY VALIDATION CAUGHT {len(error_items)} ERRORS ===")
        print(f"Removed {len(error_items)} bad items and saved them to errors_legal_dataset.json")
        try:
            if os.path.exists("errors_legal_dataset.json"):
                with open("errors_legal_dataset.json", "r") as f:
                    existing_errors = json.load(f)
            else:
                existing_errors = []
            
            existing_ids = {e["id"] for e in existing_errors}
            for e in error_items:
                if e["id"] not in existing_ids:
                    existing_errors.append(e)
                    
            with open("errors_legal_dataset.json", "w") as f:
                json.dump(existing_errors, f, indent=2)
        except Exception as ex:
            print(f"Failed to write errors log: {ex}")

    print(f"Integrity checks passed! {len(cleaned_dataset)} items are valid.")
    return cleaned_dataset


def call_with_fallback(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.ReadTimeout):
            print(f"\n[FATAL] CourtListener servers are hanging (ReadTimeout). Aborting early to save progress.")
            raise RuntimeError("FATAL_READ_TIMEOUT")
            
        if "Authorization" in SESSION.headers:
            print(f"\n[WARNING] CourtListener API key exhausted ({type(e).__name__}). Falling back to anonymous requests.")
            del SESSION.headers["Authorization"]
            return func(*args, **kwargs)
        else:
            print(f"\n[FATAL] CourtListener anonymous API failed after all retries ({type(e).__name__}).")
            raise RuntimeError("API_EXHAUSTED")


# ── DATASET BUILD PIPELINE ────────────────────────────────
def build_dataset(items_per_topic=5):
    dataset = []
    item_id = 0
    completed_topics = set()
    seen_fab_names = set()
    seen_valid_citations = set()

    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing {OUTPUT_FILE}, resuming...")
        try:
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
                dataset = data.get("items", [])
                
                # Determine max item_id to resume from
                for item in dataset:
                    num = int(item["id"].split("_")[1])
                    if num > item_id:
                        item_id = num
                        
                    if item.get("category") == "fabricated" and item.get("case_name"):
                        seen_fab_names.add(item["case_name"].lower())
                    if item.get("category") == "valid" and item.get("citation"):
                        seen_valid_citations.add(item["citation"])
                        
                # Determine which topics are fully complete (have fabricated items)
                topic_counts = {}
                for item in dataset:
                    t = item.get("topic")
                    if t:
                        topic_counts[t] = topic_counts.get(t, 0) + 1
                
                # If a topic has items, we'll assume it finished its loop previously
                # since we save at the very end of a topic.
                completed_topics = set(topic_counts.keys())
                
                print(f"Resuming with {len(dataset)} items across {len(completed_topics)} topics. Next ID: {item_id + 1}")
        except Exception as e:
            print(f"Failed to load existing dataset: {e}")

    for topic_index, topic in enumerate(LEGAL_TOPICS):
        if topic in completed_topics:
            print(f"\n=== Topic: {topic} (ALREADY COMPLETED, SKIPPING) ===")
            continue

        print(f"\n=== Topic: {topic} ===")
            
        time.sleep(10) # Prevent rapid-fire search limits between topics
        
        try:
            cases = call_with_fallback(search_cases, topic, max_results=20, retry_stats=retry_stats)
            build_stats["retrieved_cases"] += len(cases)

            # 1. Generate VALID cases
            print("  Generating valid cases...")
            valid_count = 0
            for case in cases:
                if valid_count >= items_per_topic:
                    break
                if not case.get("cluster_id"):
                    continue
    
                cite = case.get("citation") or case.get("case_name")
                if cite in seen_valid_citations:
                    print(f"    [SKIP] Duplicate citation already generated: {cite}")
                    continue
    
                time.sleep(5)  # CourtListener detail endpoint rate limit
                text, source_type, trigger = call_with_fallback(get_opinion_text, case["cluster_id"], retry_stats=retry_stats)
                if not text:
                    print(f"    [SKIP] No holding extracted for: {case['case_name']} (cluster {case['cluster_id']})")
                    continue
    
                build_stats.setdefault("source_word_counts", []).append(len(text.split()))
    
                claim = draft_claim_for_valid_case(text, case["case_name"], topic)
                if not claim:
                    continue
                if claim.strip().upper() == "INVALID_SOURCE_TEXT":
                    build_stats["discarded_generation_failure"] += 1
                    continue
    
                # NEW: QA verify the valid claim
                print(f"    Evaluating generated valid claim with {QA_PROVIDER.upper()}...")
                v_qa = verify_semantic_support(
                    claim, text,
                    provider=QA_PROVIDER, retry_stats=retry_stats,
                )
                v_verdict = v_qa.get("verdict", "ERROR")
                v_conf = v_qa.get("confidence", 0.0)
                v_status = v_qa.get("verification_status", "api_error")
                
                build_stats["generated_valid_claims"] += 1
    
                if v_status != "verified" or v_verdict != "SUPPORTED" or v_conf < QA_CONFIDENCE_THRESHOLD:
                    if v_verdict == "NOT_SUPPORTED":
                        build_stats["rejected_valid_not_supported"] += 1
                    elif v_verdict == "UNSURE":
                        build_stats["rejected_valid_unsure"] += 1
                    elif v_verdict == "ERROR":
                        build_stats["rejected_valid_api_error"] += 1
                    elif v_verdict == "SUPPORTED":
                        build_stats["rejected_valid_low_confidence"] += 1
                    continue
    
                # Update distribution stats
                build_stats["holding_source_distribution"][source_type] = (
                    build_stats["holding_source_distribution"].get(source_type, 0) + 1
                )
                if trigger:
                    build_stats["holding_trigger_distribution"][trigger] = (
                        build_stats["holding_trigger_distribution"].get(trigger, 0) + 1
                    )
    
                item_id += 1
                valid_item = {
                    "id": f"law_{item_id:03d}",
                    "category": "valid",
                    "claim": claim,
                    "citation": cite,
                    "case_name": case["case_name"],
                    "case_holding_text": text,
                    "text_source": source_type,
                    "holding_trigger_phrase": trigger,
                    "source_text_used": text,
                    "generation_source": "generated_from_holding",
                    "needs_manual_review": check_needs_manual_review(claim),
                    "correct_verdict": "valid",
                    "verification_verdict": v_verdict,
                    "verification_confidence": v_conf,
                    "verification_status": v_status,
                    "source_url": f"https://www.courtlistener.com/opinion/{case['cluster_id']}/",
                    "topic": topic,
                }
                dataset.append(valid_item)
                seen_valid_citations.add(cite)
                build_stats["accepted_valid_claims"] += 1
                valid_count += 1
                
                # 2. Generate WRONG-CONTENT case immediately via LLM adversarial swap
                build_stats["generated_wrong_claims"] += 1
                unsupported_claim, strategy, difficulty = draft_unsupported_claim(text, claim, case["case_name"], topic)
                
                if unsupported_claim:
                    print(f"    Evaluating generated wrong claim with {QA_PROVIDER.upper()}...")
                    qa_res = verify_semantic_support(
                        unsupported_claim, text,
                        provider=QA_PROVIDER, retry_stats=retry_stats,
                    )
                    verdict = qa_res.get("verdict", "ERROR")
                    conf = qa_res.get("confidence", 0.0)
                    reason = qa_res.get("reason", "")
                    v_status = qa_res.get("verification_status", "api_error")
                    
                    if v_status == "verified" and verdict == "NOT_SUPPORTED" and conf >= QA_CONFIDENCE_THRESHOLD:
                        item_id += 1
                        dataset.append({
                            "id": f"law_{item_id:03d}",
                            "category": "real_wrong_content",
                            "claim": unsupported_claim,
                            "citation": case["citation"] or case["case_name"],
                            "case_name": case["case_name"],
                            "case_holding_text": text,
                            "text_source": source_type,
                            "holding_trigger_phrase": trigger,
                            "source_text_used": text,
                            "generation_source": "llm_adversarial_generation",
                            "parent_valid_claim": claim,
                            "adversarial_strategy": strategy,
                            "difficulty_bucket": difficulty,
                            "correct_verdict": "invalid",
                            "verification_verdict": verdict,
                            "verification_confidence": conf,
                            "verification_reason": reason,
                            "verification_status": v_status,
                            "source_url": f"https://www.courtlistener.com/opinion/{case['cluster_id']}/",
                            "topic": topic,
                        })
                        build_stats["accepted_wrong_claims"] += 1
                        build_stats["qa_confidences"].append(conf)
                        build_stats["adversarial_strategy_distribution"][strategy] = build_stats["adversarial_strategy_distribution"].get(strategy, 0) + 1
                        build_stats["difficulty_distribution"][difficulty] = build_stats["difficulty_distribution"].get(difficulty, 0) + 1
                    else:
                        if verdict == "SUPPORTED":
                            build_stats["rejected_supported"] += 1
                        elif verdict == "UNSURE":
                            build_stats["rejected_unsure"] += 1
                        elif verdict == "ERROR":
                            build_stats["rejected_api_error"] += 1
                        elif verdict == "NOT_SUPPORTED":
                            build_stats["rejected_low_confidence"] += 1
                else:
                    build_stats["discarded_generation_failure"] += 1
                    
                time.sleep(10)  # Rate limiting for CourtListener between cases
    
            # Check if we should proceed to fabricated
            if valid_count == 0:
                print(f"  [WARNING] No valid cases found for '{topic}'. Dropping topic and skipping fabricated generation.")
                try:
                    if os.path.exists("errors_legal_dataset.json"):
                        with open("errors_legal_dataset.json", "r") as f:
                            existing_errors = json.load(f)
                    else:
                        existing_errors = []
                    existing_errors.append({"topic": topic, "error": "No valid cases found hence dropped"})
                    with open("errors_legal_dataset.json", "w") as f:
                        json.dump(existing_errors, f, indent=2)
                except Exception as ex:
                    print(f"  [ERROR] Failed to write to errors_legal_dataset.json: {ex}")
                continue

            # 3. Generate FABRICATED cases
            print("  Generating fabricated cases...")
            fab_count = 0
            while fab_count < items_per_topic:
                fab = draft_fabricated_citation(topic, seen_fab_names)
                if not fab:
                    continue
    
                if fab["case_name"].lower() in seen_fab_names:
                    print(f"  WARNING: duplicate fabricated case name generated by LLM, skipping: {fab['case_name']}")
                    continue
    
                # Programmatic deduplication (string similarity)
                too_similar = False
                for seen_name in seen_fab_names:
                    if SequenceMatcher(None, fab["case_name"].lower(), seen_name).ratio() > 0.75:
                        too_similar = True
                        break
                
                if too_similar:
                    print(f"  WARNING: fabricated case name too similar to existing case, skipping: {fab['case_name']}")
                    continue
    
                build_stats["generated_fabricated_candidates"] += 1
    
                # Strong collision check: explicitly compare returned strings
                fab_name_lower = fab["case_name"].split(" v. ")[0].lower()
                fab_cite_lower = fab["citation"].lower() if fab["citation"] else ""
    
                collided = False
                for query in [fab["case_name"], fab["citation"], f"{fab['case_name']} {fab['citation']}"]:
                    if collided:
                        break
                    if not query:
                        continue
    
                    time.sleep(5) # Prevent rapid-fire search limits
                    check = call_with_fallback(search_cases, query, max_results=2, retry_stats=retry_stats)
    
                    for res in check:
                        res_name = (res["case_name"] or "").lower()
                        res_cite = (res["citation"] or "").lower()
    
                        if fab_name_lower in res_name or (fab_cite_lower and fab_cite_lower in res_cite):
                            collided = True
                            break
    
                if collided:
                    print(f"  WARNING: strict collision with real case, skipping: {fab['case_name']} / {fab['citation']}")
                    build_stats["collision_rejection_count"] += 1
                    continue
    
                item_id += 1
                dataset.append({
                    "id": f"law_{item_id:03d}",
                    "category": "fabricated",
                    "claim": fab["claim"],
                    "citation": fab["citation"],
                    "case_name": fab["case_name"],
                    "case_holding_text": None,
                    "text_source": None,
                    "source_text_used": None,
                    "generation_source": "fabricated",
                    "correct_verdict": "invalid",
                    "source_url": None,
                    "topic": topic,
                })
                seen_fab_names.add(fab["case_name"].lower())
                fab_count += 1
                time.sleep(5)
            
        except RuntimeError as e:
            if str(e) == "FATAL_READ_TIMEOUT":
                print(f"\n  [WARNING] CourtListener servers are unresponsive. Saving whatever was generated for '{topic}' and exiting gracefully.")
                save_dataset(dataset, path=OUTPUT_FILE)
                import sys
                sys.exit(0)
            elif str(e) == "API_EXHAUSTED":
                print(f"\n  [WARNING] CourtListener servers exhausted while processing '{topic}'. Saving whatever was generated and moving to the next topic...")
            else:
                raise e
            
        # Save incrementally after every topic
        print(f"\n  [Checkpoint] Saving progress after topic '{topic}'...")
        save_dataset(dataset, path=OUTPUT_FILE)
        print("  [Checkpoint] Saved successfully.")

    return dataset


def save_dataset(dataset, path=OUTPUT_FILE):
    print("\n=== Validating and Saving Dataset ===")
    dataset = validate_dataset_integrity(dataset)

    # Calculate distributions
    confs = build_stats["qa_confidences"]
    word_counts = build_stats.get("source_word_counts", [])
    avg_word_count = int(np.mean(word_counts)) if word_counts else 0
    avg_conf = float(np.mean(confs)) if confs else 0.0

    gen_valid = build_stats.get("generated_valid_claims", 0)
    acc_valid = build_stats.get("accepted_valid_claims", 0)
    valid_acceptance_rate = round(acc_valid / gen_valid, 3) if gen_valid > 0 else 0.0
    
    gen_wrong = build_stats.get("generated_wrong_claims", 0)
    acc_wrong = build_stats.get("accepted_wrong_claims", 0)
    wrong_acceptance_rate = round(acc_wrong / gen_wrong, 3) if gen_wrong > 0 else 0.0
    
    gen_fab = build_stats.get("generated_fabricated_candidates", 0)
    acc_fab = len([i for i in dataset if i["category"] == "fabricated"])
    fabricated_generation_rate = round(acc_fab / gen_fab, 3) if gen_fab > 0 else 0.0
    
    failure_dist = {}
    for f in build_stats.get("generation_failures", []):
        reason = f["failure_reason"]
        failure_dist[reason] = failure_dist.get(reason, 0) + 1

    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "provider": LLM_PROVIDER,
        "model": GENERATION_MODEL,
        "generation_temperature": GENERATION_TEMPERATURE,
        "qa_provider": QA_PROVIDER,
        "qa_model": QA_MODELS.get(QA_PROVIDER, "unknown"),
        "qa_temperature": QA_TEMPERATURE,
        "dataset_builder_version": DATASET_BUILDER_VERSION,
        "prompt_version": QA_PROMPT_VERSION,
        "total_items": len(dataset),
        "build_statistics": {
            "retrieved_cases": build_stats["retrieved_cases"],
            "discarded_snippet_only": build_stats["discarded_snippet_only"],
            "discarded_generation_failure": build_stats["discarded_generation_failure"],
            "generated_valid_claims": gen_valid,
            "accepted_valid_claims": acc_valid,
            "valid_acceptance_rate": valid_acceptance_rate,
            "rejected_valid_not_supported": build_stats.get("rejected_valid_not_supported", 0),
            "rejected_valid_unsure": build_stats.get("rejected_valid_unsure", 0),
            "rejected_valid_low_confidence": build_stats.get("rejected_valid_low_confidence", 0),
            "rejected_valid_api_error": build_stats.get("rejected_valid_api_error", 0),
            "generated_wrong_claims": gen_wrong,
            "accepted_wrong_claims": acc_wrong,
            "wrong_acceptance_rate": wrong_acceptance_rate,
            "generated_fabricated_candidates": gen_fab,
            "fabricated_generation_rate": fabricated_generation_rate,
            "rejected_supported": build_stats["rejected_supported"],
            "rejected_unsure": build_stats["rejected_unsure"],
            "rejected_low_confidence": build_stats["rejected_low_confidence"],
            "rejected_api_error": build_stats["rejected_api_error"],
            "collision_rejection_count": build_stats["collision_rejection_count"],
            "accepted_syllabi": build_stats.get("accepted_syllabi", 0),
            "accepted_headnotes": build_stats.get("accepted_headnotes", 0),
            "accepted_opinion_extracts": build_stats.get("accepted_opinion_extracts", 0),
            "rejected_metadata_summaries": build_stats.get("rejected_metadata_summaries", 0),
            "sub_opinions_checked_total": build_stats.get("sub_opinions_checked_total", 0),
            "average_source_word_count": avg_word_count,
            "average_verification_confidence": round(avg_conf, 3),
            "holding_source_distribution": build_stats["holding_source_distribution"],
            "holding_trigger_distribution": build_stats["holding_trigger_distribution"],
            "adversarial_strategy_distribution": build_stats.get("adversarial_strategy_distribution", {}),
            "difficulty_distribution": build_stats.get("difficulty_distribution", {}),
            "generation_failure_distribution": failure_dist,
        },
        "generation_failures": build_stats.get("generation_failures", []),
        "retry_statistics": retry_stats
    }

    final_data = {**metadata, "items": dataset}
    with open(path, "w") as f:
        json.dump(final_data, f, indent=2)
    print(f"\nSaved {len(dataset)} items to {path}")
    print("--------------Moving to next Topic-------------")


if __name__ == "__main__":
    ds = build_dataset(items_per_topic=5)
    save_dataset(ds)