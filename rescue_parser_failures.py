#!/usr/bin/env python3
"""
Rescue Parser Failures (Separate Folder Version)
================================================
Copies result directories into separate output directories (*_rescued/) and
recovers judgment fields for parser failures from raw response text in place.

No API calls are made — all data is parsed from existing saved fields.

Usage:
  python rescue_parser_failures.py
"""

import os
import json
import re
import shutil

TARGET_DIRS = [
    ("results_gptoss_full_run/main", "results_gptoss_full_run_rescued/main"),
    ("results_gptoss_full_run/mitigation", "results_gptoss_full_run_rescued/mitigation"),
    ("results_llama_full_run/results/main", "results_llama_full_run_rescued/results/main"),
    ("results_llama_full_run/results/mitigation", "results_llama_full_run_rescued/results/mitigation"),
]

REFUSAL_PHRASES = [
    "sorry", "can't", "cannot", "unable to assist", "comply", "i can't help"
]

VERIFIER_LABEL_SPACE = {"VALID", "INVALID", "UNSUPPORTED"}

GROUND_TRUTH_LABEL_MAP = {
    "valid": "VALID",
    "invalid": "INVALID",
    "real_wrong_content": "INVALID",
    "fabricated": "INVALID",
}


def normalize_ground_truth(raw_value):
    if raw_value is None:
        return None
    raw_upper = str(raw_value).strip().upper()
    if raw_upper in VERIFIER_LABEL_SPACE:
        return raw_upper
    raw_lower = str(raw_value).strip().lower()
    return GROUND_TRUTH_LABEL_MAP.get(raw_lower)


def parse_enhanced_offline(response_text, initial_judgment=None):
    clean_text = (response_text or "").replace("*", "").replace("#", "").strip()
    if not clean_text:
        return None, None, None

    # Skip repetition glitches / meta-prompt leaks
    if clean_text.startswith(".Forms") or clean_text.startswith("owsows") or clean_text.startswith("The user is asking"):
        return None, None, None

    # Stage 1: Standard JUDGMENT: regex + SUPPORTED/NOT SUPPORTED
    m1 = re.search(r"JUDGMENT:\s*(VALID|INVALID|UNSUPPORTED|SUPPORTED|NOT SUPPORTED)", clean_text, re.IGNORECASE)
    if m1:
        raw_j = m1.group(1).upper()
        j = "VALID" if raw_j == "SUPPORTED" else ("UNSUPPORTED" if raw_j == "NOT SUPPORTED" else raw_j)
        
        m_conf = re.search(r"CONFIDENCE:\s*(\d+)", clean_text, re.IGNORECASE)
        conf = int(m_conf.group(1)) if m_conf else None
        
        m_reas = re.search(r"REASONING:\s*(.+)", clean_text, re.DOTALL | re.IGNORECASE)
        reas = m_reas.group(1).strip() if m_reas else clean_text[:300]
        return j, conf, reas

    # Stage 2: Conversational 'is actually UNSUPPORTED', 'judgment of INVALID is correct'
    m2 = re.search(r'(?:is|judgment of|verdict of|conclusion of|judgment is)\s+["\']?(VALID|INVALID|UNSUPPORTED|SUPPORTED)["\']?', clean_text, re.IGNORECASE)
    if m2:
        raw_j = m2.group(1).upper()
        j = "VALID" if raw_j == "SUPPORTED" else raw_j
        m_conf = re.search(r"CONFIDENCE:\s*(\d+)", clean_text, re.IGNORECASE)
        conf = int(m_conf.group(1)) if m_conf else None
        return j, conf, clean_text[:300]

    # Stage 3: 'stand by (my) (original) judgment'
    m3 = re.search(r'stand by (?:my )?(?:original )?judgment(?: of ["\']?(VALID|INVALID|UNSUPPORTED)["\']?)?', clean_text, re.IGNORECASE)
    if m3:
        j = m3.group(1).upper() if m3.group(1) else initial_judgment
        if j:
            return j, None, clean_text[:300]

    # Stage 4: Verifier maintains judgment ('confirmed the verifier's judgment', 'no need for further discussion')
    if any(p in clean_text.lower() for p in [
        "confirmed the verifier's judgment", "interaction is complete",
        "verifier's judgment of", "that's not necessary. i'm designed to provide a judgment",
        "you've confirmed the verifier's judgment"
    ]):
        if initial_judgment:
            return initial_judgment, None, clean_text[:300]

    # Stage 5: 'I must respectfully disagree with your assertion/rebuttal... does not explicitly state / provide sufficient'
    if "respectfully disagree" in clean_text.lower() or "does not provide sufficient" in clean_text.lower() or "does not necessarily provide" in clean_text.lower():
        if "unsupported" in clean_text.lower():
            return "UNSUPPORTED", None, clean_text[:300]
        elif "invalid" in clean_text.lower():
            return "INVALID", None, clean_text[:300]
        elif initial_judgment:
            return initial_judgment, None, clean_text[:300]

    return None, None, None


def recover_file_in_dir(src_path, dst_path):
    with open(src_path, "r") as fp:
        data = json.load(fp)

    init_j = data.get("initial_judgment")
    fin_j = data.get("final_judgment")
    status = data.get("status")

    if init_j is not None and fin_j is not None and status != "error":
        # Save exact copy to dst_path
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as fp:
            json.dump(data, fp, indent=2, default=str)
        return False, "already_ok"

    init_raw = (data.get("initial_raw") or "").strip()
    final_raw = (data.get("final_raw") or "").strip()

    # Skip pure refusals
    raw_check = (init_raw if init_j is None else final_raw).lower()
    if any(p in raw_check for p in REFUSAL_PHRASES):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as fp:
            json.dump(data, fp, indent=2, default=str)
        return False, "refusal"

    new_init_j, init_conf, init_reas = parse_enhanced_offline(init_raw) if init_j is None else (init_j, data.get("initial_confidence"), data.get("initial_reasoning"))
    if init_j is None and new_init_j:
        data["initial_judgment"] = new_init_j
        data["initial_confidence"] = init_conf or data.get("initial_confidence")
        data["initial_reasoning"] = init_reas or data.get("initial_reasoning")

    current_init = data.get("initial_judgment")
    new_fin_j, fin_conf, fin_reas = parse_enhanced_offline(final_raw, initial_judgment=current_init) if fin_j is None else (fin_j, data.get("final_confidence"), data.get("final_reasoning"))
    if fin_j is None and new_fin_j:
        data["final_judgment"] = new_fin_j
        data["final_confidence"] = fin_conf or data.get("final_confidence")
        data["final_reasoning"] = fin_reas or data.get("final_reasoning")

    # If both judgments are now resolved, update derived fields
    if data.get("initial_judgment") is not None and data.get("final_judgment") is not None:
        gt_norm = normalize_ground_truth(data.get("ground_truth"))
        init_correct = (data["initial_judgment"] == gt_norm) if gt_norm else None
        fin_correct = (data["final_judgment"] == gt_norm) if gt_norm else None
        flipped = (data["initial_judgment"] != data["final_judgment"])

        flip_direction = None
        if flipped:
            if init_correct and not fin_correct:
                flip_direction = "correct_to_incorrect"
            elif not init_correct and fin_correct:
                flip_direction = "incorrect_to_correct"
            else:
                flip_direction = "lateral"

        conf_delta = None
        if data.get("initial_confidence") is not None and data.get("final_confidence") is not None:
            conf_delta = data["final_confidence"] - data["initial_confidence"]

        data.update({
            "status": "ok",
            "initial_correct": init_correct,
            "final_correct": fin_correct,
            "flipped": flipped,
            "flip_direction": flip_direction,
            "confidence_delta": conf_delta,
            "ground_truth_normalized": gt_norm,
        })
        if "error" in data:
            del data["error"]

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as fp:
            json.dump(data, fp, indent=2, default=str)
        return True, f"init={data['initial_judgment']}, final={data['final_judgment']}"

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as fp:
        json.dump(data, fp, indent=2, default=str)
    return False, "unparseable_glitch_or_empty"


def main():
    print("=" * 80)
    print("RUNNING OFFLINE RECOVERY INTO SEPARATE RESCUED FOLDERS (NO API CALLS)")
    print("=" * 80)

    total_recovered = 0
    total_unrecovered = 0

    for src_dir, dst_dir in TARGET_DIRS:
        if not os.path.exists(src_dir):
            print(f"Skipping missing directory: {src_dir}")
            continue

        print(f"\nProcessing: {src_dir} -> {dst_dir}")
        dir_recovered = 0
        for root, _, files in os.walk(src_dir):
            for f in sorted(files):
                if f.endswith(".json"):
                    src_path = os.path.join(root, f)
                    rel_path = os.path.relpath(src_path, src_dir)
                    dst_path = os.path.join(dst_dir, rel_path)

                    success, msg = recover_file_in_dir(src_path, dst_path)
                    if success:
                        dir_recovered += 1
                        total_recovered += 1
                        print(f"  [RECOVERED & SAVED] {f} -> {msg}")
                    elif msg == "unparseable_glitch_or_empty":
                        total_unrecovered += 1
                        print(f"  [UNRECOVERED] {f} -> {msg}")

        print(f"Directory Total Recovered: {dir_recovered}")

    print("\n" + "=" * 80)
    print(f"RECOVERY COMPLETE: {total_recovered} files rescued into separate folders!")
    print(f"Remaining unparseable (generation glitches/empty): {total_unrecovered}")
    print("=" * 80)


if __name__ == "__main__":
    main()
