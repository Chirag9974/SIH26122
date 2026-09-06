"""Deterministic consistency checks (PDF section 10).

Runs after the LLM structured output and before final Pydantic validation.
Repairs what can be repaired deterministically, forces needs_review when
something is unsafe, and never invents content: unresolvable fields become
null. Every emitted warning stays inside the closed vocab.WARNINGS set.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import re

from common.schema import normalize_nested
from common.vocab import (ACTION_ALIASES, LOCATION_ALIASES, UNCERTAIN_CUES,
                   NEGATION_CUES, STATUS_CUES)
from extraction.extractor import detect_relevance, IRRELEVANT_HINTS


def _parse_iso(s):
    """True when s is a parseable ISO timestamp."""
    if not s or not isinstance(s, str):
        return False
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


# alias -> canonical maps for deterministic canonicalization
_ALIAS_TO_ACTION: dict[str, str] = {}
for _canon, _surfaces in ACTION_ALIASES.items():
    _ALIAS_TO_ACTION[_canon.lower()] = _canon
    for _s in _surfaces:
        _ALIAS_TO_ACTION[_s.lower()] = _canon

_LOC_TO_CANON: dict[str, str] = {}
for _canon, _surfaces in LOCATION_ALIASES.items():
    _LOC_TO_CANON[_canon.lower()] = _canon
    for _s in _surfaces:
        _LOC_TO_CANON[_s.lower()] = _canon

# Site-logistics cues the baseline knows plus a few more seen in LLM
# hallucinations. Used ONLY to guard the LLM path; the deterministic
# baseline behavior is unchanged.
_LLM_IRRELEVANT_HINTS = tuple(IRRELEVANT_HINTS) + (
    "generator set", "security patrol", "diesel bowser",
)

# words that describe execution status, never a work type
_STATUS_WORDS = {
    "done", "completed", "complete", "compl", "cmpltd", "completd", "finished",
    "started", "under way", "in progress", "ongoing", "over", "achieved",
    "closed out", "all done", "wrapped up", "carried out", "executed",
    "performed", "held", "conducted", "delayed", "halted", "suspended",
}

# Pseudo-values the model occasionally writes into context.line_number;
# never a real line number.
_LINE_SENTINELS = {
    "synced", "n/a", "na", "none", "same", "same as above",
    "as above", "tbd", "-",
}


def _has_contrast(text: str) -> bool:
    """True when the sentence contrasts two claims (complete-but-pending)."""
    low = " ".join(str(text or "").lower().split())
    return any(m in low for m in (" but ", "however", " although", " though",
                                  "lekin"))


def _canon_action(value):
    """Map a model action string to the canonical work-type vocabulary."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in _STATUS_WORDS:
        return None
    if v in _ALIAS_TO_ACTION:
        return _ALIAS_TO_ACTION[v]
    best = None
    for alias, canon in _ALIAS_TO_ACTION.items():
        if len(alias) >= 4 and alias in v:
            if best is None or len(alias) > len(best[0]):
                best = (alias, canon)
    return best[1] if best else None


def _canon_location(value):
    """Map a location shorthand to the canonical location name."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in _LOC_TO_CANON:
        return _LOC_TO_CANON[v]
    best = None
    for alias, canon in _LOC_TO_CANON.items():
        if len(alias) >= 4 and alias in v:
            if best is None or len(alias) > len(best[0]):
                best = (alias, canon)
    return best[1] if best else None


def _text_actions(raw: str) -> set:
    """All canonical actions whose aliases occur in the raw text."""
    low = raw.lower()
    found = set()
    for alias, canon in _ALIAS_TO_ACTION.items():
        if len(alias) < 4:
            continue
        idx = low.find(alias)
        while idx >= 0:
            before = low[idx - 1] if idx else " "
            end = idx + len(alias)
            after = low[end] if end < len(low) else " "
            if not before.isalnum() and not after.isalnum():
                found.add(canon)
                break
            idx = low.find(alias, idx + 1)
    return found


def _sole_action_alias(raw: str):
    """The one canonical action whose alias occurs in the text, else None.

    Multi-activity reports mention several aliases; only unambiguous texts
    qualify for action recovery.
    """
    found = _text_actions(raw)
    return found.pop() if len(found) == 1 else None


def _repair_ts(s, rdate: str | None):
    """Resolve a stated clock time to a full ISO timestamp.

    Bare "08:00" or "08:00:00" -> report_date + T08:00:00 (the report describes
    that day; resolution, not invention). "YYYY-MM-DD HH:MM(:SS)" gets its
    separator normalized. Returns (iso_or_original, repaired_bool).
    """
    if not s or not isinstance(s, str):
        return s, False
    s2 = s.strip()
    if " " in s2 and len(s2) >= 11 and s2[:4].isdigit():
        s2 = s2.replace(" ", "T", 1)
    if "T" in s2:
        date_part, _, time_part = s2.partition("T")
        bits = time_part.split(":")
        # Accept glued junk prefixes ("...-log24T11:30:00" from the model's
        # bookkeeping): anything up to 16 chars before the T, as long as the
        # clock part is clean. A valid ISO date is exactly 10 chars.
        if (10 <= len(date_part) <= 16
                and len(bits) in (2, 3)
                and all(b.isdigit() for b in bits)):
            date_iso = date_part[-10:]
            hh = min(int(bits[0]), 23)
            mm = int(bits[1])
            ss = int(bits[2]) if len(bits) == 3 else 0
            return f"{date_iso}T{hh:02d}:{mm:02d}:{ss:02d}", True
        return s, False
    bits = s2.split(":")
    if rdate and len(bits) in (2, 3) and all(b.isdigit() for b in bits):
        hh = min(int(bits[0]), 23)
        mm = int(bits[1])
        ss = int(bits[2]) if len(bits) == 3 else 0
        return f"{rdate}T{hh:02d}:{mm:02d}:{ss:02d}", True
    return s, False


def _recover_location(raw: str):
    """Sole canonical location present in the raw text, else None.

    Evidence-backed recovery when the model omitted location: if exactly one
    known location (or alias) occurs in the report, it is clearly implied.
    """
    low = raw.lower()
    found = set()
    for alias, canon in _LOC_TO_CANON.items():
        idx = low.find(alias)
        while idx >= 0:
            before = low[idx - 1] if idx else " "
            end = idx + len(alias)
            after = low[end] if end < len(low) else " "
            if not before.isalnum() and not after.isalnum():
                found.add(canon)
                break
            idx = low.find(alias, idx + 1)
    return found.pop() if len(found) == 1 else None


def _locate(raw: str, seg: str):
    """Start index of seg in raw (case-insensitive, word-bounded preferred)."""
    low, segl = raw.lower(), seg.lower().strip()
    if not segl or len(segl) > len(low):
        return None
    cands = [i for i in range(len(low) - len(segl) + 1)
             if low.startswith(segl, i)]
    if not cands:
        return None
    for i in cands:
        before = low[i - 1] if i else " "
        after = low[i + len(segl)] if i + len(segl) < len(low) else " "
        if not before.isalnum() and not after.isalnum():
            return i
    return cands[0]


def _evidence_span(raw: str, ev: dict):
    """Resolve the event's evidence to an exact span of the raw text.

    Tries evidence.source_text, then the activity description. Returns
    (span, None) or (None, "unsupported_evidence").
    """
    evd = ev.get("evidence") or {}
    act = ev.get("activity") or {}
    for seg in (evd.get("source_text"), act.get("description")):
        if not seg or not isinstance(seg, str):
            continue
        i = _locate(raw, seg)
        if i is not None:
            return raw[i:i + len(seg.strip())].strip(), None
    return None, "unsupported_evidence"


def _sentence_actions(sent: str) -> set:
    """Canonical work types named in one sentence (via alias tables)."""
    return _text_actions(sent or "")


def _segments(sent: str) -> list:
    """Clause segments of a sentence (split on , ; and ' and ')."""
    return [p.strip() for p in re.split(r"[;,]|\band\b", sent or "")
            if p.strip()]


def _segment_of(sent: str, seg: str) -> str:
    """The clause segment of sent that contains seg (the evidence span)."""
    low = (sent or "").lower()
    segl = (seg or "").lower().strip()
    pos = low.find(segl[:40]) if segl else -1
    if pos < 0:
        return sent or ""
    start, end = 0, len(low)
    for m in re.finditer(r"[;,]|\band\b", low):
        if m.start() < pos:
            start = m.end()
        elif m.start() >= pos + len(segl[:40]):
            end = m.start()
            break
    return sent[start:end].strip()


def _incomplete_without_action(seg: str) -> bool:
    """A clause reporting unfinished work WITHOUT naming an activity.

    Such a clause contradicts a completed sibling event ("two joints were
    found unfinished"); a clause that names its own activity ("grouting has
    not started") is just another event, not a conflict.
    """
    low = (seg or "").lower()
    cues = ("not done", "not completed", "unfinished", "incomplete",
            "nahi", "partly", "partial")
    return any(c in low for c in cues) and not _sentence_actions(low)


def _sentence_containing(raw: str, seg: str):
    """The sentence of raw that contains seg (evidence-anchored hedge probe).

    Hedges often sit just outside the evidence span ("... from 4 PM appears
    complete."), so uncertainty must be judged on the full sentence.
    """
    i = _locate(raw, seg)
    if i is None:
        return None
    start = max(raw.rfind(".", 0, i), raw.rfind(";", 0, i),
                raw.rfind("\n", 0, i)) + 1
    end = len(raw)
    for ch in (".", ";", "\n"):
        j = raw.find(ch, i + len(seg.strip()))
        if j >= 0:
            end = min(end, j)
    return raw[start:end]


def validate_extraction(pred: dict, report: dict) -> tuple[dict, list[str]]:
    """Apply PDF section 10 checks. Returns (normalized prediction, issues).

    issues are failure tags for error_analysis.py; they describe what the
    validator had to repair or flag, not what it invented.
    """
    raw = report["raw_text"]
    issues: list[str] = []
    n = normalize_nested(pred)
    rel, events = n["relevance"], n["events"]

    # Rule: relevance=false means zero events. If the model emitted events
    # anyway, the events are the stronger signal: flip relevance to true.
    if not rel["is_relevant"] and events:
        rel["is_relevant"] = True
        rel["reason"] = rel.get("reason") or "events present in text"
        issues.append("relevance_false_with_events")

    # Work types the text itself names (computed once; raw never changes).
    _text_acts_seen = _text_actions(raw)

    for ev in events:
        flags = {w for w in ev.get("warnings", []) if isinstance(w, str)}
        # The model's self-flag is unreliable: it hedges plain facts and
        # stamps needs_review on them (observed on qwen2.5:7b), while missing
        # real gaps. The deterministic rules below are the sole authority on
        # review flags (PDF sections 3 and 10).
        nr = False
        exx, qty, tim = ev["execution"], ev["quantity"], ev["time"]

        # Evidence traceability: the claim must exist in the raw text.
        span, miss = _evidence_span(raw, ev)
        if span is None:
            flags.add("unsupported_evidence")
            nr = True
            issues.append("unsupported_evidence")
        else:
            ev["evidence"]["source_text"] = span

        # Sentence probe for hedge/negation judgement: hedges often sit just
        # outside the evidence span ("... from 4 PM appears complete."), so
        # judge on the WHOLE sentence containing the evidence.
        evd_span = (ev.get("evidence") or {}).get("source_text") or ""
        sent = _sentence_containing(raw, evd_span) or evd_span
        text_probe = " ".join(str(x) for x in (
            sent,
            (ev.get("activity") or {}).get("description")) if x)

        # Progress in 0-100 only (normalizer clamps; guard the residual).
        pct = exx["progress_percent"]
        if pct is not None and not (0 <= pct <= 100):
            exx["progress_percent"] = None
            nr = True
            issues.append("progress_out_of_range")
        elif isinstance(pct, float):
            exx["progress_percent"] = int(round(pct))
            if not pct.is_integer():
                issues.append("progress_rounded")

        # Quantity: non-negative; completed <= total when comparable.
        c, t = qty.get("completed"), qty.get("total")
        if c is not None and t is not None and c > t:
            ev["quantity"] = {"completed": None, "total": None, "unit": None}
            exx["progress_percent"] = None
            nr = True
            flags.add("quantity_partial")
            issues.append("quantity_completed_gt_total")

        # Resolve stated clock times against the report date first.
        for tk in ("start", "end"):
            fixed, did = _repair_ts(tim[tk], report.get("report_date"))
            if did:
                tim[tk] = fixed
                flags.add("relative_date_resolved")
                issues.append("time_resolved_from_report_date")

        # Times must be parseable when present.
        bad_ts = any(x is not None and not _parse_iso(x)
                     for x in (tim["start"], tim["end"]))
        if bad_ts:
            if tim["start"] is not None and not _parse_iso(tim["start"]):
                tim["start"] = None
            if tim["end"] is not None and not _parse_iso(tim["end"]):
                tim["end"] = None
            tim["certainty"] = "missing"
            nr = True
            issues.append("invalid_timestamp")

        # Overnight rollover: end may roll to the next day when evidence
        # supports it (end < start). More than one day of gap is unsafe.
        if tim["start"] and tim["end"]:
            s = datetime.fromisoformat(tim["start"])
            e = datetime.fromisoformat(tim["end"])
            if e < s:
                e2 = e + timedelta(days=1)
                if (e2 - s) <= timedelta(days=1):
                    tim["end"] = e2.isoformat()
                    flags.add("relative_date_resolved")
                    issues.append("overnight_rollover")
                else:
                    tim["end"] = None
                    nr = True
                    issues.append("overnight_gap_too_large")

        # Midnight-exact start with no textual support is a model placeholder
        # ("Work started ..." -> 00:00). Drop unsupported midnight starts.
        if (tim["start"] and tim["start"][11:16] == "00:00"
                and not any(tok in text_probe.lower()
                            for tok in ("midnight", "00:00", "0:00",
                                        "24:00"))):
            tim["start"] = None
            tim["certainty"] = "missing"
            flags.add("missing_time")
            issues.append("unsupported_midnight_start_dropped")

        # Unsupported uncertainty: the model hedges although the text has no
        # hedge word. A clean claim stays affirmed (PDF: no invention).
        if exx["assertion"] == "uncertain":
            low_probe = text_probe.lower()
            # Hedge words that qualify a DIFFERENT field ("around 16:30",
            # "roughly 40 percent") do not make the STATUS uncertain.
            field_hedged = any(c in low_probe and
                               any(f in low_probe[max(0, low_probe.find(c) - 12):
                                                  low_probe.find(c) + len(c) + 14]
                                   for f in (":", "percent", "%", " m "))
                               for c in UNCERTAIN_CUES if len(c) >= 3)
            unc = (any(c in low_probe for c in UNCERTAIN_CUES if len(c) >= 3)
                   and not field_hedged)
            # Negation must sit right next to this event's own activity term
            # ("termination nahi hua", "grouting has not started"). Cues next
            # to other words refer to missing fields ("location not
            # confirmed", "no start time was recorded", "confirm nahi kiya"),
            # never to the activity itself.
            act_terms = [w for w in
                         ((ev["activity"].get("action") or "") + " " +
                          (ev["activity"].get("description") or "")).lower().split()
                         if len(w) >= 4 and w not in _STATUS_WORDS]
            report_words = ("confirm", "record", "reported", "verified",
                            "update", "information")
            own_acts = _sentence_actions(
                ((ev["activity"].get("action") or "") + " " +
                 (ev["activity"].get("description") or "")).lower())
            neg = False
            for cue in NEGATION_CUES:
                if len(cue) < 3:
                    continue
                j = low_probe.find(cue)
                while j >= 0 and not neg:
                    after = low_probe[j + len(cue):j + len(cue) + 16].lstrip()
                    if not any(after.startswith(w) for w in report_words):
                        before_toks = low_probe[:j].rstrip().split()
                        window = before_toks[-3:]
                        if any(t in act_terms for t in window):
                            clause = low_probe[max(0, j - 80):j + len(cue) + 30]
                            others = _sentence_actions(clause) - own_acts
                            if not others:
                                neg = True
                    j = low_probe.find(cue, j + 1)
                if neg:
                    break
            if neg and not unc:
                # "nahi hua" / "was not ...": the model hedged a plainly
                # negated claim; the text cue is evidence enough.
                exx["assertion"] = "negated"
                flags.add("negated_statement")
                flags.discard("uncertain_statement")
                issues.append("negation_recovered_from_text")
            elif not unc and not neg and not _has_contrast(text_probe):
                exx["assertion"] = "affirmed"
                flags.discard("uncertain_statement")
                issues.append("unsupported_uncertainty_downgraded")

        # Negated + except-exclusion: "Everything is progressing normally
        # except the loop checks" describes the EXCLUDED item, not a negated
        # activity. Drop the meaningless assertion, keep the status, no
        # review (the exclusion itself is stated plainly).
        if (exx["assertion"] == "negated" and exx["status"] == "not_started"
                and (" except " in sent.lower()
                     or " other than " in sent.lower())):
            exx["assertion"] = "affirmed"
            issues.append("exception_exclusion_not_negation")

        # Negation can never stand next to a completion-like status.
        if exx["assertion"] == "negated" and exx["status"] in (
                "completed", "started", "in_progress"):
            exx["status"] = None
            exx["progress_percent"] = None
            nr = True
            flags.add("negated_statement")
            issues.append("negation_as_completion")

        # Warning hygiene: every negated claim carries negated_statement.
        if exx["assertion"] == "negated":
            flags.add("negated_statement")

        # Uncertainty stays uncertain and always needs review.
        if "uncertain_statement" in flags:
            exx["assertion"] = "uncertain"
        if exx["assertion"] == "uncertain":
            nr = True

        # Fractional progress implies in_progress (a partial quantity cannot
        # be a completed event); resolve status BEFORE conflict coherence so
        # the check sees the final status.
        if (exx["status"] is None
                and ev["quantity"]["completed"] is not None
                and ev["quantity"]["total"] is not None
                and 0 < ev["quantity"]["completed"] < ev["quantity"]["total"]):
            exx["status"] = "in_progress"
            issues.append("status_inferred_from_partial_quantity")

        # Status recovery: an explicit completion cue in the event's own
        # CLAUSE with no contradicting cue ("have been completed") is
        # evidence the model dropped. Whole-sentence probing would import
        # sibling clauses' cues ("rack b pe 24 spool done, crew left at 4").
        # Never applied to negated assertions (the cue there is the negated
        # object, e.g. "not completed").
        if (exx["status"] is None and exx["assertion"] != "negated"
                and not _incomplete_without_action(sent)):
            low_seg = _segment_of(sent, evd_span).lower()
            blocked = any(c in low_seg for c in STATUS_CUES["not_started"]
                          + STATUS_CUES["cancelled"] + STATUS_CUES["suspended"])
            if not blocked:
                for st in ("completed", "started", "in_progress"):
                    if any(c in low_seg for c in STATUS_CUES[st]):
                        exx["status"] = st
                        issues.append("status_recovered_from_text")
                        break

        # Conflicting-report coherence: judge AFTER status resolution. A
        # coherent assertion whose own evidence clause names at most this one
        # activity - and whose sentence contains no activity-less unfinished
        # clause ("two joints found unfinished") - cannot make the report
        # "conflicting"; the model stamps the warning on its own hedging
        # (observed on qwen2.5:7b).
        if "conflicting_report" in flags:
            ok_status = ((exx["assertion"] == "affirmed"
                          and exx["status"] in ("completed", "started",
                                                "in_progress"))
                         or (exx["assertion"] == "negated"
                             and exx["status"] == "not_started"))
            seg_probe = _segment_of(sent, evd_span)
            incomplete_bare = any(_incomplete_without_action(s)
                                  for s in _segments(sent)
                                  if s not in (evd_span or ""))
            coherent = (ok_status
                        and (len(events) <= 1
                             or len(_sentence_actions(seg_probe)) <= 1)
                        and not incomplete_bare)
            if coherent:
                flags.discard("conflicting_report")
                issues.append("conflicting_report_rejected")
            else:
                nr = True

        # A relevant event with no resolvable status is unsafe to auto-accept.
        if exx["status"] is None:
            nr = True
            issues.append("missing_status")

        # Missing-time warning consistency.
        if tim["start"] is None and tim["end"] is None:
            flags.add("missing_time")

        # Canonicalize activity.action: the model may put a status word or
        # paraphrase there. Alias-map it; recover from the description when it
        # collapses to a status word; null it when unrecognized.
        act_raw = ev["activity"].get("action")
        act_canon = _canon_action(act_raw)
        if act_canon is None and act_raw:
            act_canon = _canon_action(ev["activity"].get("description"))
            if act_canon is not None:
                issues.append("action_recovered_from_description")
        # Swap guard (single-event docs only): a canonical action the text
        # never mentions, while the text names exactly one other work type,
        # means the model swapped them ("Painting work" -> concreting).
        if (act_canon is not None and len(events) == 1
                and _text_acts_seen and act_canon not in _text_acts_seen
                and len(_text_acts_seen - {act_canon}) == 1):
            act_canon = next(iter(_text_acts_seen - {act_canon}))
            issues.append("action_swap_corrected_from_text")
        if act_canon is not None and act_canon != act_raw:
            ev["activity"]["action"] = act_canon
            issues.append("action_canonicalized")
        elif act_canon is None and act_raw:
            ev["activity"]["action"] = None
            issues.append("action_unrecognized")
            nr = True
        if ev["activity"]["action"] is None:
            sole = next(iter(_text_acts_seen)) if len(_text_acts_seen) == 1 else None
            if sole is None:
                # Alias tables verb-match; natural phrasing ("cable was
                # pulled", "painting work") does not. Try token-to-alias
                # containment as a last, still text-anchored resort.
                toks = {t.strip(".,;:()!?") for t in raw.lower().split()}
                toks.discard("")
                cands = set()
                for tok in toks:
                    if len(tok) < 5 or tok in _STATUS_WORDS:
                        continue
                    for al, canon in _ALIAS_TO_ACTION.items():
                        if len(al) >= 5 and (tok in al or al in tok):
                            cands.add(canon)
                if len(cands) == 1:
                    sole = cands.pop()
            if sole is not None:
                ev["activity"]["action"] = sole
                issues.append("action_recovered_from_text")

        # Pseudo-values are not line numbers ("synced" leaks from the
        # model's cross-event bookkeeping) and neither is echoed prose
        # ("in Line 18"). A real line id carries digits and is not a
        # preposition phrase.
        lnv = ev["context"].get("line_number")
        if isinstance(lnv, str):
            low_ln = lnv.strip().lower()
            toks = low_ln.replace('"', ' ').split()
            m_ln = re.fullmatch(r"[a-z]{2,6}(\d{1,3})", low_ln)
            if m_ln:
                # Glued junk with a real id ("log24" -> "24").
                ev["context"]["line_number"] = m_ln.group(1)
                issues.append("line_number_glued_recovered")
            elif (low_ln in _LINE_SENTINELS
                    or not any(ch.isdigit() for ch in low_ln)
                    or (len(toks) > 1 and any(t in ("in", "on", "at", "the",
                                                    "near", "log", "line")
                                              for t in toks))):
                ev["context"]["line_number"] = None
                issues.append("line_number_sentinel_nulled")

        # Canonicalize location shorthand ("u-300" -> "Unit 300").
        loc_raw = ev["context"].get("location")
        loc_canon = _canon_location(loc_raw)
        if loc_canon is not None and loc_canon != loc_raw:
            ev["context"]["location"] = loc_canon
            issues.append("location_canonicalized")

        # Location recovery: sole known location in the text fills an empty
        # context.location (clearly implied, never invented).
        if ev["context"].get("location") is None:
            recovered = _recover_location(raw)
            if recovered is not None:
                ev["context"]["location"] = recovered
                issues.append("location_recovered_from_text")

        # Line-number recovery: exactly one "Line NN" mention in the text is
        # evidence for an empty context.line_number (same convention as the
        # deterministic baseline: "Line 18" -> "18").
        if ev["context"].get("line_number") is None:
            lns = set(re.findall(r"\bline\s*(?:no\.?\s*)?(\d{1,3})(?:-\d+)?\b",
                                 raw, re.I))
            if len(lns) == 1:
                ev["context"]["line_number"] = lns.pop()
                issues.append("line_number_recovered_from_text")

        # Progress derivation per gold convention:
        #   completed + affirmed + no quantity -> 100
        #   quantity present -> round(completed * 100 / total)
        if (exx["status"] == "completed" and exx["assertion"] == "affirmed"
                and "conflicting_report" not in flags
                and exx["progress_percent"] is None
                and ev["quantity"]["completed"] is None):
            exx["progress_percent"] = 100
        c, t = ev["quantity"]["completed"], ev["quantity"]["total"]
        if c is not None and t is not None and t > 0:
            exx["progress_percent"] = round(c * 100 / t)

        ev["warnings"] = sorted(flags)
        ev["needs_review"] = nr
        ev.pop("_ev_text", None)

    # High-precision site-logistics guard: the deterministic relevance
    # detector (baseline knowledge) overrides LLM hallucinations on
    # canteen/logistics/visitor texts. LLM path only.
    _rel = detect_relevance(raw)
    _text_acts = _text_actions(raw)
    _anchored = any(ev["activity"]["action"] in _text_acts for ev in events)
    _hint = next((h for h in IRRELEVANT_HINTS if h in raw.lower()), None)
    if (not _rel["is_relevant"] and events and not _anchored
            and _hint is not None):
        # Only genuine site-logistics texts (an actual hint phrase present)
        # may zero out events. Generic low-confidence relevance never does:
        # work described without vocabulary aliases is still work.
        events = []
        rel["is_relevant"] = False
        rel["confidence"] = min(rel.get("confidence", 0.9), 0.95)
        rel["reason"] = "site-logistics content, no project execution information"
        issues.append("irrelevant_hint_forced_empty")

    # Drop empty events: no action and no status carries no information.
    kept = []
    for ev in events:
        if ev["activity"]["action"] or ev["execution"]["status"] is not None:
            kept.append(ev)
        else:
            issues.append("empty_event_dropped")
    events = kept

    return {"relevance": rel, "events": events}, issues
