"""Shared terminology for SIH26122: disciplines, work types, aliases, phrasing banks.

This is the *terminology layer* from spec section 2. Both the synthetic data
generator and the extractor read it. That is intentional -- an alias table is a
legitimate data asset. The generator's report *templates* are NOT shared with
the extractor, so the benchmark stays honest.

Phrasing banks (STATUS_PHRASES, TIME_PHRASES, REPORT_TEMPLATES) are used only
by gen_reports.py to produce realistic human-written-looking text, not by the
extractor (which relies on ACTION_ALIASES, DISCIPLINE_ALIASES, etc.).
"""

PROJECT_ID = "OIL-GGS-2026"

# discipline -> {code, works, locations, ref_prefix}
# work: action (canonical), verb (schedule wording), obj (name template),
#       unit, qty range, field_noun (how site teams say the object)
DISCIPLINES = {
    "Piping": {
        "code": "PIP",
        "locations": ["Rack A", "Rack B", "Rack C", "Unit 200", "Pipe Rack North"],
        "works": [
            {"action": "erection", "verb": "Erect", "obj": "{size} Spool Line {ref}",
             "unit": "spool", "qty": (6, 18), "field_noun": "{size} spool"},
            {"action": "welding", "verb": "Weld", "obj": "{size} Joints Line {ref}",
             "unit": "joint", "qty": (10, 40), "field_noun": "{size} joints"},
            {"action": "fit-up", "verb": "Fit-up", "obj": "{size} Spool Line {ref}",
             "unit": "spool", "qty": (4, 12), "field_noun": "{size} spool fit-up"},
            {"action": "hydrotest", "verb": "Hydrotest", "obj": "Line {ref}",
             "unit": "line", "qty": (1, 1), "field_noun": "line hydrotest"},
            {"action": "painting", "verb": "Paint", "obj": "{size} Line {ref}",
             "unit": "m", "qty": (20, 120), "field_noun": "{size} line painting"},
        ],
    },
    "Civil": {
        "code": "CIV",
        "locations": ["Area 10", "Area 20", "Tank Farm", "Substation Pad", "Unit 100"],
        "works": [
            {"action": "excavation", "verb": "Excavate", "obj": "Foundation {ref}",
             "unit": "m3", "qty": (30, 200), "field_noun": "foundation excavation"},
            {"action": "shuttering", "verb": "Shutter", "obj": "Foundation {ref}",
             "unit": "m2", "qty": (20, 90), "field_noun": "shuttering"},
            {"action": "concreting", "verb": "Cast Concrete", "obj": "Foundation {ref}",
             "unit": "m3", "qty": (10, 80), "field_noun": "concrete pour"},
            {"action": "backfilling", "verb": "Backfill", "obj": "Foundation {ref}",
             "unit": "m3", "qty": (20, 120), "field_noun": "backfilling"},
        ],
    },
    "Electrical": {
        "code": "ELE",
        "locations": ["Substation 1", "MCC Room", "Area 20", "Unit 300", "Cable Trench T-4"],
        "works": [
            {"action": "cable pulling", "verb": "Pull", "obj": "Cable {ref}",
             "unit": "m", "qty": (50, 400), "field_noun": "cable pulling"},
            {"action": "termination", "verb": "Terminate", "obj": "Cable {ref}",
             "unit": "termination", "qty": (2, 24), "field_noun": "cable termination"},
            {"action": "installation", "verb": "Install", "obj": "Lighting Panel {ref}",
             "unit": "panel", "qty": (1, 4), "field_noun": "panel installation"},
            {"action": "earthing", "verb": "Install", "obj": "Earthing Grid {ref}",
             "unit": "m", "qty": (30, 150), "field_noun": "earthing work"},
        ],
    },
    "Instrumentation": {
        "code": "INS",
        "locations": ["Unit 200", "Unit 300", "Control Room", "Rack B", "Wellhead Pad 2"],
        "works": [
            {"action": "installation", "verb": "Install", "obj": "Transmitter {ref}",
             "unit": "instrument", "qty": (1, 8), "field_noun": "transmitter installation"},
            {"action": "loop check", "verb": "Loop Check", "obj": "Loop {ref}",
             "unit": "loop", "qty": (1, 6), "field_noun": "loop checking"},
            {"action": "calibration", "verb": "Calibrate", "obj": "Instrument {ref}",
             "unit": "instrument", "qty": (1, 10), "field_noun": "calibration"},
            {"action": "tubing", "verb": "Run Tubing", "obj": "Impulse Line {ref}",
             "unit": "m", "qty": (10, 60), "field_noun": "impulse tubing"},
        ],
    },
    "Equipment": {
        "code": "EQP",
        "locations": ["Unit 100", "Unit 200", "Compressor House", "Tank Farm", "Area 10"],
        "works": [
            {"action": "installation", "verb": "Install", "obj": "Pump {ref}",
             "unit": "unit", "qty": (1, 2), "field_noun": "pump installation"},
            {"action": "alignment", "verb": "Align", "obj": "Pump {ref}",
             "unit": "unit", "qty": (1, 2), "field_noun": "alignment"},
            {"action": "grouting", "verb": "Grout", "obj": "Baseplate {ref}",
             "unit": "m3", "qty": (1, 6), "field_noun": "grouting"},
            {"action": "erection", "verb": "Erect", "obj": "Vessel {ref}",
             "unit": "unit", "qty": (1, 1), "field_noun": "vessel erection"},
        ],
    },
    "Structural": {
        "code": "STR",
        "locations": ["Rack A", "Rack B", "Unit 200", "Area 20", "Compressor House"],
        "works": [
            {"action": "erection", "verb": "Erect", "obj": "Steel Frame {ref}",
             "unit": "t", "qty": (2, 30), "field_noun": "steel erection"},
            {"action": "bolting", "verb": "Bolt-up", "obj": "Steel Frame {ref}",
             "unit": "joint", "qty": (10, 80), "field_noun": "bolt-up"},
            {"action": "grating", "verb": "Install Grating", "obj": "Platform {ref}",
             "unit": "m2", "qty": (10, 90), "field_noun": "grating installation"},
        ],
    },
}

SIZES = ["6in", "8in", "12in", "18in", "24in"]

# canonical action -> field-speak surface forms the extractor must recognise
ACTION_ALIASES = {
    "erection": ["erection", "erect", "erected", "erecting", "erectn", "erectio",
                 "fitted", "fitting up", "put up"],
    "welding": ["welding", "weld", "welded", "wldg", "welding done"],
    "fit-up": ["fit-up", "fit up", "fitup", "fitted up"],
    "hydrotest": ["hydrotest", "hydro test", "hydro-test", "hydrotesting", "pressure test"],
    "painting": ["painting", "paint", "painted", "pntg", "coating"],
    "excavation": ["excavation", "excavate", "excavated", "excvn", "digging"],
    "shuttering": ["shuttering", "shutter", "shuttered", "formwork", "form work"],
    "concreting": ["concreting", "concrete", "concreted", "pour", "poured", "casting", "cast"],
    "backfilling": ["backfilling", "backfill", "backfilled", "back filling"],
    "cable pulling": ["cable pulling", "cable pull", "pulled cable", "cable laying", "cbl pulling"],
    "termination": ["termination", "terminate", "terminated", "termn", "glanding"],
    "installation": ["installation", "install", "installed", "instl", "erected in place",
                     "fixing", "fixed", "mounted"],
    "earthing": ["earthing", "earth pit", "grounding", "earthed"],
    "loop check": ["loop check", "loop checking", "loop checked", "loop test"],
    "calibration": ["calibration", "calibrate", "calibrated", "calib"],
    "tubing": ["tubing", "impulse tubing", "tube run", "tubing laid"],
    "alignment": ["alignment", "align", "aligned", "algn"],
    "grouting": ["grouting", "grout", "grouted"],
    "bolting": ["bolting", "bolt-up", "bolt up", "bolted", "torquing"],
    "grating": ["grating", "grating installation", "grating laid"],
}

DISCIPLINE_ALIASES = {
    "Piping": ["piping", "pip", "pipe", "mech-piping"],
    "Civil": ["civil", "civ", "cvl"],
    "Electrical": ["electrical", "ele", "elec", "elect"],
    "Instrumentation": ["instrumentation", "ins", "inst", "instru", "e&i"],
    "Equipment": ["equipment", "eqp", "equip", "mechanical", "mech"],
    "Structural": ["structural", "str", "struct", "steel"],
}

# location shorthand seen in field reports -> canonical location
LOCATION_ALIASES = {
    "Rack A": ["rack a", "rack-a", "rb-a", "r/a", "rack_a"],
    "Rack B": ["rack b", "rack-b", "rb-b", "r/b", "rack_b"],
    "Rack C": ["rack c", "rack-c", "rb-c", "r/c"],
    "Unit 100": ["unit 100", "u-100", "u100", "unit-100"],
    "Unit 200": ["unit 200", "u-200", "u200", "unit-200"],
    "Unit 300": ["unit 300", "u-300", "u300", "unit-300"],
    "Area 10": ["area 10", "a-10", "ar 10", "area-10"],
    "Area 20": ["area 20", "a-20", "ar 20", "area-20"],
    "Tank Farm": ["tank farm", "tf", "tankfarm"],
    "Substation Pad": ["substation pad", "ss pad", "sub-station pad"],
    "Substation 1": ["substation 1", "ss-1", "ss 1", "substn 1"],
    "MCC Room": ["mcc room", "mcc", "mcc-room"],
    "Control Room": ["control room", "ctrl room", "cr"],
    "Compressor House": ["compressor house", "comp house", "k-house"],
    "Pipe Rack North": ["pipe rack north", "prn", "pipe rack n"],
    "Cable Trench T-4": ["cable trench t-4", "trench t-4", "t-4 trench"],
    "Wellhead Pad 2": ["wellhead pad 2", "wh pad 2", "whp-2"],
    # unscheduled locations: real project areas with NO scheduled activities.
    # Used only to generate genuine no_match reports.
    "Unit 400": ["unit 400", "u-400", "u400", "unit-400"],
    "Pipe Rack South": ["pipe rack south", "prs", "pipe rack s"],
    "Substation 2": ["substation 2", "ss-2", "ss 2", "substn 2"],
    "Area 30": ["area 30", "a-30", "ar 30", "area-30"],
    "Flare Area": ["flare area", "flare"],
}

#: Locations guaranteed absent from the generated schedule (no_match cases).
UNSCHEDULED_LOCATIONS = ["Unit 400", "Pipe Rack South", "Substation 2",
                         "Area 30", "Flare Area"]

STATUS_CUES = {
    "completed": ["completed", "complete", "done", "finished", "closed out", "achieved",
                  "compl", "cmpltd", "over",
                  # common field-report misspellings
                  "completd", "cmpleted", "finshed"],
    "started": ["started", "commenced", "begun", "began", "started work", "taken up",
                "mobilised", "mobilized", "balance tomorrow"],
    "in_progress": ["in progress", "ongoing", "continuing", "continued", "under progress",
                    "wip", "in-progress", "progressing", "balance work", "partly done",
                    "still open", "still pending", "found incomplete", "under way",
                    "partway through"],
    "suspended": ["suspended", "stopped midway", "stopped at", "halted", "carried over",
                  "shifted to tomorrow", "on hold", "paused for the day"],
    "cancelled": ["cancelled", "canceled", "called off", "dropped"],
    "not_started": ["not started", "not yet started", "yet to start", "could not start",
                    "did not start", "no work", "nil progress"],
}

NEGATION_CUES = ["not ", "no ", "did not", "didn't", "could not", "couldn't", "never",
                 "without", "cancelled", "canceled", "nil ", "failed to"]

UNCERTAIN_CUES = ["may have", "might", "probably", "possibly", "seems", "appears",
                  "reportedly", "unclear", "to be confirmed", "tbc", "approx", "around",
                  "about", "likely", "as per verbal"]

# warnings vocabulary (closed set)
WARNINGS = [
    "missing_line_number",
    "missing_location",
    "missing_time",
    "ambiguous_activity",
    "possible_duplicate",
    "conflicting_report",
    "relative_date_resolved",
    "negated_statement",
    "uncertain_statement",
    "quantity_partial",
    "no_schedule_candidate",
]

# ---------------------------------------------------------------------------
# Phrasing banks for gen_reports.py — realistic human-written-like surface text.
# NOT shared with the extractor; only used to synthesize report text.
# ---------------------------------------------------------------------------

# Natural completion-status expressions (used in normal / partial / multi)
STATUS_PHRASES = {
    "completed": ["completed", "got it done", "wrapped up", "all done",
                  "finished off", "knocked out", "closed out", "done for the day"],
    "in_progress": ["in progress", "ongoing", "partway through", "under way",
                    "3 of 7 done so far", "still going"],
    "started": ["started", "began", "kicked off", "mobilised today", "took up"],
    "suspended": ["stopped midway", "halted at", "carried over to tomorrow",
                  "paused", "shifted to tomorrow", "wrapped up for today, balance carries over"],
    "cancelled": ["cancelled", "called off", "scrapped", "dropped"],
    "not_started": ["did not start", "could not be started", "not taken up",
                    "deferred", "nil progress"],
}

# Realistic time-window shorthands (the extractor must parse these)
TIME_PHRASES = [
    "08:00 to 12:00", "0900 to 1630", "8 AM to 4 PM", "7 AM to 3 PM",
    "07:30 to 11:30", "10 to 4", "9 AM to 5 PM", "0800-1700",
    "morning shift", "afternoon shift", "09:00 to 13:00", "14:00 to 18:00",
]

# Crew / foreman shorthand names
CREW_NAMES = {
    "Piping": ["PIP-3 crew", "Maurice's team", "Line-fitters", "Pipefitters"],
    "Civil": ["Civil gang", "Excavation crew", "Formwork team", "Foundations crew"],
    "Electrical": ["ELE-2 gang", "Cable crew", "Terminations team", "Power team"],
    "Instrumentation": ["INS gang", "Instrument techs", "Loop-techs", "Cal team"],
    "Equipment": ["MECH crew", "Riggers", "Installation gang", "Alignment team"],
    "Structural": ["Steel gang", "Bolters", "Grating crew", "Deck team"],
}

# Structural templates per case kind — each generates a different sentence skeleton.
# Placeholders: {crew}, {work}, {id}, {loc}, {time}, {status}, {reason}, {qty}
# These are hand-written to read like real supervisor shorthand.
REPORT_TEMPLATES = {
    "normal_formal": [
        "{crew} completed {work} {id} at {loc} {time}.",
        "At {loc}, {work} {id} was completed {time}.",
        "{work} {id}: all done at {loc} {time}. {crew} on site.",
        "Completed {work} {id} at {loc} {time}.",
    ],
    "normal_shorthand": [
        "{code} - {work}{id} @ {loc}, {time}, done.",
        "{code}: {work} {id} {loc} {time} done.",
        "{code} - {work}{id} @ {loc}, {time}, compl.",
        "{id} {work} {loc} {time} done. {code}.",
    ],
    "noisy": [
        "{work}{id} @ {loc} {time} completd.",
        "{code}-{work} {id} {loc} {time} cmpltd.",
        "{id} {work} @ {loc} {time} done.",
        "{work} at {loc} {time} - {code}",
    ],
    "multi": [
        "{lines}",
    ],
    "partial": [
        "Started {work} {id} at {loc} {time}. {qty} completed today.",
        "{work} {id} at {loc}: {qty} done {time}.",
        "{crew} started {work} {id} at {loc}. {qty} of total done by {time}.",
        "Partial: {work} {id} {loc} {time}. {qty} so far.",
    ],
    "conflict": [
        "{work} {id} at {loc} completed {time}. Later found incomplete, {qty} still open.",
        "{work} {id} {loc} done {time}. Update: still {qty} outstanding.",
        "Completed {work} {id} at {loc}. Then QA flagged {qty} incomplete.",
    ],
    "uncertain": [
        "{work} {id} at {loc} probably completed, to be confirmed.",
        "Looks like {work} {id} at {loc} got done {time}, unverified.",
        "{work} {id} {loc} seems done but needs double-check.",
    ],
    "delay": [
        "{work} {id} at {loc} delayed due to {reason}.",
        "Could not finish {work} {id} at {loc} {reason}.",
        "{work} {id} {loc} on hold — {reason}.",
    ],
    "negative": [
        "{work} {id} at {loc} could not be started today, {reason}.",
        "{work} {id} at {loc} has been cancelled, {reason}.",
        "No {work} at {loc} today — {reason}.",
    ],
    "suspended": [
        "{work} {id} at {loc} stopped midway {time}. Carried over to night shift.",
        "{crew} started {work} {id} at {loc}, halted {time}. Balance tomorrow.",
        "{work} {id} {loc}: suspended {time}. Night shift picks up.",
    ],
}
