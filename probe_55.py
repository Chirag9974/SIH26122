"""Probe: run the 55 user-provided test sentences through the LLM extractor.

Writes one JSON line per sentence to data/evaluation/probe_55_results.jsonl
and progress to stdout. Resumable: cached calls return instantly.
"""
import sys, json, time

sys.path.insert(0, "src")
from extraction.extractor_llm import extract_report

REPORT_DATE = "2026-09-06"  # fixed so "today"/"yesterday" resolve deterministically

CASES = [
    ("N01", "normal",    "Piping crew wrapped up the 18-inch line work at Rack C by 5 PM."),
    ("N02", "normal",    "Work on Pump P-204 was completed during the morning shift."),
    ("N03", "normal",    "Cable tray installation in Substation 1 started after lunch."),
    ("N04", "normal",    "The foundation at Area 20 is now complete."),
    ("N05", "normal",    "Fit-up for the 12in spool was finished around 16:30."),
    ("M01", "missing",   "Line 24 erection is complete."),
    ("M02", "missing",   "Pump P-204 installation is still going on."),
    ("M03", "missing",   "Painting work was completed today."),
    ("M04", "missing",   "Cable termination started, location not confirmed yet."),
    ("M05", "missing",   "About 60 percent of the work is done on the north rack."),
    ("S01", "messy",     "P4 ka kaam 60% ho gaya, baaki kal."),
    ("S02", "messy",     "rack b pe 24 spool done, crew left at 4."),
    ("S03", "messy",     "wldg complete but 3 joints baki."),
    ("S04", "messy",     "pump fit karke alignment chalu hai."),
    ("S05", "messy",     "u-200 me loop check abhi chal rha."),
    ("X01", "negation",  "No welding was carried out on Line 18 today."),
    ("X02", "negation",  "Pump P-204 installation did not start."),
    ("X03", "negation",  "Cable termination nahi hua aaj."),
    ("X04", "negation",  "The foundation was not completed despite the concrete pour."),
    ("X05", "negation",  "No progress was made on the north rack."),
    ("U01", "uncertain", "Pump P-204 installation appears to be complete."),
    ("U02", "uncertain", "Line 24 may have been hydrotested yesterday."),
    ("U03", "uncertain", "The supervisor says the cable work is probably finished."),
    ("U04", "uncertain", "Around 70% seems to be done, but this needs confirmation."),
    ("U05", "uncertain", "Steel erection is reportedly complete; foreman confirmation is pending."),
    ("T01", "time",      "Erection started at 9 AM and finished at 4 PM."),
    ("T02", "time",      "Work ran from 22:00 until 02:00."),
    ("T03", "time",      "Line 24 work started yesterday and finished this morning."),
    ("T04", "time",      "The crew began around 10 and stopped by 3."),
    ("T05", "time",      "Painting was completed by the end of the night shift."),
    ("Q01", "quantity",  "8 out of 12 spools were erected today."),
    ("Q02", "quantity",  "145 m of cable was pulled against a total of 220 m."),
    ("Q03", "quantity",  "Three foundations out of five have been completed."),
    ("Q04", "quantity",  "12 joints welded today, balance 18."),
    ("Q05", "quantity",  "Roughly 40 percent complete; exact quantity not available."),
    ("E01", "multi",     "Civil finished Foundation F12 while electrical started cable pulling in Area 20."),
    ("E02", "multi",     "Pump alignment was completed in the morning and grouting started afterward."),
    ("E03", "multi",     "Line 24 fit-up is complete, welding has started, and hydrotest is still pending."),
    ("E04", "multi",     "Rack B erection stopped at noon; the team moved to Rack C and started another spool."),
    ("E05", "multi",     "P-204 installation completed, alignment is ongoing, and grouting has not started."),
    ("H01", "hard",      "Line 24 is done, although two joints were found unfinished during inspection."),
    ("H02", "hard",      "P4 is 30% ahead of P5."),
    ("H03", "hard",      "The work is delayed because steel hasn't reached site, but the completed portion is approximately 30%."),
    ("H04", "hard",      "Same spool as yesterday was reported complete again today."),
    ("H05", "hard",      "Rack B work was stopped due to rain and resumed after the weather cleared."),
    ("H06", "hard",      "The remaining spool was finished today; no start time was recorded."),
    ("H07", "hard",      "Installation is complete as per contractor, but the supervisor has marked it pending."),
    ("H08", "hard",      "Work started on the north rack, then the crew shifted to another line without closing out the first one."),
    ("H09", "hard",      "Everything in Unit 200 is progressing normally except the instrument loop checks."),
    ("H10", "hard",      "P4 ka kaam ho gaya hai bol rahe hain, lekin supervisor ne abhi confirm nahi kiya."),
    ("Z01", "special",   "aaj lunch bhi nahi hua, site pe bahut kaam tha"),
    ("Z02", "special",   "Line 24 completed but actually only 60% was done."),
    ("Z03", "special",   "about 5 of 12 spools finished"),
    ("Z04", "special",   "P4 is 20% ahead of P5"),
    ("Z05", "special",   "P4 probably complete hai, foreman se confirm karna hai"),
]

OUT = "data/evaluation/probe_55_results.jsonl"

def summarize(out):
    return {
        "engine": out.get("engine"),
        "relevance": out.get("relevance"),
        "top_needs_review": bool(out.get("needs_review")),
        "top_warnings": out.get("warnings", []),
        "events": [
            {
                "action": (ev.get("activity") or {}).get("action"),
                "desc": (ev.get("activity") or {}).get("description"),
                "status": (ev.get("execution") or {}).get("status"),
                "assertion": (ev.get("execution") or {}).get("assertion"),
                "progress": (ev.get("execution") or {}).get("progress_percent"),
                "start": (ev.get("time") or {}).get("start"),
                "end": (ev.get("time") or {}).get("end"),
                "time_certainty": (ev.get("time") or {}).get("certainty"),
                "location": (ev.get("context") or {}).get("location"),
                "line": (ev.get("context") or {}).get("line_number"),
                "equipment": (ev.get("context") or {}).get("equipment"),
                "qty_completed": (ev.get("quantity") or {}).get("completed"),
                "qty_total": (ev.get("quantity") or {}).get("total"),
                "unit": (ev.get("quantity") or {}).get("unit"),
                "issue_type": (ev.get("issue") or {}).get("type"),
                "needs_review": bool(ev.get("needs_review")),
                "warnings": ev.get("warnings", []),
            }
            for ev in (out.get("events") or [])
        ],
    }

def main():
    done = set()
    try:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass

    with open(OUT, "a", encoding="utf-8") as f:
        for cid, cat, text in CASES:
            if cid in done:
                print(f"[skip] {cid} cached", flush=True)
                continue
            t0 = time.time()
            out = extract_report(
                text,
                metadata={"report_id": f"PROBE-{cid}", "report_date": REPORT_DATE},
            )
            dt = time.time() - t0
            rec = {"id": cid, "cat": cat, "text": text,
                   "latency_s": round(dt, 1), **summarize(out)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            acts = [e["action"] for e in rec["events"]]
            print(f"[ok] {cid} ({cat}) {dt:5.1f}s events={len(acts)} "
                  f"actions={acts} review={rec['top_needs_review']} "
                  f"rel={rec['relevance'].get('is_relevant')}", flush=True)
    print("PROBE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
