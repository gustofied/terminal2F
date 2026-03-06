"""Generates the v2 dataset for the email_to_cc_bcc RL environment.

Changes from v1:
- Sender picked first and excluded from all recipient fields
- Deterministic To/CC/BCC routing based on roles (lowest priority number = To)
- BCC only from reserve observers (compliance/legal/HR/finance), never active participants
- Three modes: normal (no BCC), compliance (confidential cue), broadcast (external privacy)
- Compliance mode guarantees at least one compliance role in reserve
- Broadcast mode guarantees enough externals active for meaningful BCC
- LLM only sees mentionable names (To + CC), not BCC people
- Email format includes Date: and From: headers with monotonic timestamps
- Stronger validator: sender not in fields, only allowed names in text, first-name BCC check
"""

import argparse
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import data_designer.config as dd
from data_designer.interface.data_designer import DataDesigner

# --- constants ---

DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Legal", "Finance",
    "HR", "Product", "Design", "Operations", "Customer Support",
    "Research", "Executive", "IT", "Procurement",
]

SCENARIO_TYPES = {
    "project_update": ["milestone reached", "delay notification", "resource request", "scope change"],
    "meeting_request": ["team standup", "client review", "board meeting", "one_on_one"],
    "escalation": ["missed deadline", "quality issue", "customer complaint", "blocked dependency"],
    "feedback": ["positive review", "constructive criticism", "peer recognition", "360 feedback"],
    "announcement": ["policy change", "new hire", "org restructure", "product launch"],
    "approval_request": ["budget sign-off", "travel approval", "access request", "vendor selection"],
    "incident_report": ["outage", "data breach", "workplace safety", "compliance violation"],
    "contract_review": ["renewal", "amendment", "termination", "new agreement"],
    "hiring_decision": ["offer approval", "rejection", "interview feedback", "salary negotiation"],
    "budget_request": ["new project", "headcount increase", "tool purchase", "training budget"],
    "deadline_change": ["extension request", "acceleration", "dependency shift", "reprioritization"],
    "complaint": ["service quality", "internal process", "harassment", "workload"],
    "onboarding": ["first week setup", "team introduction", "access provisioning", "mentor assignment"],
    "offboarding": ["resignation", "termination", "knowledge transfer", "exit interview"],
    "vendor_negotiation": ["pricing discussion", "SLA review", "contract dispute", "new vendor pitch"],
    "security_issue": ["phishing attempt", "unauthorized access", "vulnerability disclosure", "audit finding"],
    "performance_review": ["annual review", "PIP", "promotion discussion", "goal setting"],
    "knowledge_sharing": ["tech talk", "documentation update", "lessons learned", "best practices"],
}

INTERNAL_ROLES = [
    "Project Lead", "Team Lead", "Senior Engineer", "Junior Engineer",
    "Engineering Manager", "VP Engineering", "CTO", "Product Manager",
    "Designer", "QA Lead", "DevOps Engineer", "Data Analyst",
    "Marketing Manager", "Sales Director", "Account Executive",
    "Legal Counsel", "Finance Director", "HR Manager",
]

EXTERNAL_ROLES = [
    "Client PM", "Client Engineer", "External Consultant",
    "Vendor Contact", "Vendor Engineer", "Contractor",
    "Partner Lead", "Agency Contact",
]

# Roles that can be BCC'd as compliance observers (from reserve only)
COMPLIANCE_ROLES = {"Legal Counsel", "HR Manager", "Finance Director", "VP Engineering", "CTO"}

# Action priority for To assignment: lower number = more likely action owner (To)
ACTION_PRIORITY = {
    "Project Lead": 1, "Team Lead": 1, "Product Manager": 1,
    "Senior Engineer": 2, "QA Lead": 2, "Designer": 2, "Data Analyst": 2,
    "DevOps Engineer": 2, "Account Executive": 2,
    "Engineering Manager": 3, "Sales Director": 3, "Marketing Manager": 3,
    "Client PM": 3, "Partner Lead": 3,
    "Client Engineer": 4, "External Consultant": 4, "Vendor Contact": 4,
    "Agency Contact": 4,
    "Junior Engineer": 5, "Vendor Engineer": 5, "Contractor": 5,
    "Legal Counsel": 6, "Finance Director": 6, "HR Manager": 6,
    "VP Engineering": 7, "CTO": 8,
}

# Seniority rank: higher number = more senior. Used for sender selection and upward promotion.
SENIORITY = {
    "Junior Engineer": 1, "Contractor": 1, "Vendor Engineer": 1,
    "Senior Engineer": 2, "QA Lead": 2, "Designer": 2, "Data Analyst": 2,
    "DevOps Engineer": 2, "Account Executive": 2,
    "Client Engineer": 2, "External Consultant": 2, "Vendor Contact": 2, "Agency Contact": 2,
    "Project Lead": 3, "Team Lead": 3, "Product Manager": 3,
    "Client PM": 3, "Partner Lead": 3,
    "Engineering Manager": 4, "Sales Director": 4, "Marketing Manager": 4,
    "Legal Counsel": 4, "Finance Director": 4, "HR Manager": 4,
    "VP Engineering": 5, "CTO": 6,
}

EXTERNAL_AUDIENCES = {"with_client", "with_vendor", "with_partner", "with_contractor"}

COMPANY_DOMAINS = [
    "acme.com", "globex.io", "initech.com", "umbrella.co",
    "waystar.com", "hooli.net", "piedpiper.io", "dunder.com",
    "sterling.co", "prestige.com",
]

PERSONAL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "protonmail.com",
    "hotmail.com", "icloud.com",
]

EXTERNAL_DOMAINS = [
    "clientcorp.com", "partnerhq.io", "vendorx.com", "agencypro.co",
    "consultfirm.com", "lawgroup.com", "bigclient.com", "extdev.io",
]

AUDIENCES = [
    "internal_only", "with_client", "with_vendor",
    "with_partner", "with_contractor", "cross_department",
]

SENSITIVITIES = ["public", "internal", "confidential", "restricted"]
HIERARCHIES = ["upward", "downward", "lateral", "mixed"]

# v2: modes instead of raw sensitivity for BCC logic
MODES = ["normal", "normal", "normal", "compliance", "compliance", "broadcast"]

CHANGE_TYPES = [
    "person_added", "person_removed", "role_shift",
    "escalation", "de_escalation", "made_confidential",
    "made_public", "delegation", "external_party_joins",
    "external_party_leaves", "urgency_increase", "scope_change",
]


# --- people + email generation ---

def _make_email(first, last, domain, taken):
    first_l = first.lower().replace(" ", "").replace("'", "")
    last_l = last.lower().replace(" ", "").replace("'", "")
    styles = [
        f"{first_l}.{last_l}",
        f"{first_l[0]}.{last_l}",
        f"{first_l}_{last_l}",
        f"{first_l}{last_l}",
    ]
    random.shuffle(styles)
    for local in styles:
        addr = f"{local}@{domain}"
        if addr not in taken:
            taken.add(addr)
            return addr
    addr = f"{first_l}.{last_l}{random.randint(10, 99)}@{domain}"
    taken.add(addr)
    return addr


def _build_email_list(persons, audience, company_domain, external_domain):
    has_external = audience in ("with_client", "with_vendor", "with_partner", "with_contractor")
    taken = set()
    entries = []

    if has_external:
        n_int, n_ext = 4, 3
    else:
        n_int, n_ext = 7, 0
    int_roles = random.sample(INTERNAL_ROLES, n_int)
    ext_roles = random.sample(EXTERNAL_ROLES, n_ext) if n_ext else []
    ii, ei = 0, 0

    for i, person in enumerate(persons):
        if isinstance(person, dict):
            first = person.get("first_name", "Unknown")
            last = person.get("last_name", "Person")
        else:
            parts = str(person).split()
            first = parts[0] if parts else "Unknown"
            last = parts[-1] if len(parts) > 1 else "Person"

        is_ext = (i >= 5 and has_external) or (i == 1 and has_external)

        if is_ext:
            role = ext_roles[ei % len(ext_roles)]
            ei += 1
            domain = external_domain
            is_external = True
        else:
            role = int_roles[ii % len(int_roles)]
            ii += 1
            domain = random.choice(PERSONAL_DOMAINS) if random.random() < 0.1 else company_domain
            is_external = False

        entries.append({
            "name": f"{first} {last}",
            "email": _make_email(first, last, domain, taken),
            "role": role,
            "is_external": is_external,
        })

    return entries


# --- deterministic routing (v2) ---

def _pick_sender(active_entries):
    """Pick sender deterministically: most senior internal person."""
    internals = [e for e in active_entries if not e["is_external"]]
    pool = internals if internals else active_entries
    return max(pool, key=lambda e: (SENIORITY.get(e["role"], 3), e["email"]))


def distribute_emails(active_entries, reserve_entries, mode, hierarchy):
    """Deterministic To/CC/BCC based on roles. No shuffle.

    - Sender: picked first and excluded from all fields
    - To: action owner (lowest ROLE_PRIORITY number = most hands-on)
    - CC: remaining active participants
    - BCC: compliance observers from reserve (compliance mode only), or
           all external contacts (broadcast mode only)
    - Active participants never go to BCC
    """
    if not active_entries:
        return {"to": [], "cc": [], "bcc": []}, None

    # Pick sender and exclude from routing
    sender = _pick_sender(active_entries)
    routable = [e for e in active_entries if e["email"] != sender["email"]]

    if not routable:
        return {"to": [], "cc": [], "bcc": []}, sender

    # Sort by action priority: lowest number = action owner = To
    sorted_active = sorted(
        routable,
        key=lambda e: (ACTION_PRIORITY.get(e["role"], 5), e["email"]),
    )

    to_entry = sorted_active[0]
    cc_entries = sorted_active[1:]

    # Hierarchy adjustments (deterministic, uses SENIORITY)
    if hierarchy == "upward" and cc_entries:
        # Find the most senior CC person and promote to To
        managers = sorted(
            [e for e in cc_entries if SENIORITY.get(e["role"], 3) >= 4],
            key=lambda e: (-SENIORITY.get(e["role"], 3), e["email"]),
        )
        if managers:
            promoted = managers[0]
            cc_entries.remove(promoted)
            to_entries = [to_entry, promoted]
        else:
            to_entries = [to_entry]
    elif hierarchy == "downward" and len(cc_entries) >= 1:
        to_entries = [to_entry]
    else:
        to_entries = [to_entry]

    # BCC from reserve only
    bcc_entries = []
    if mode == "compliance":
        for e in reserve_entries:
            if e["role"] in COMPLIANCE_ROLES:
                bcc_entries.append(e)
        if not bcc_entries and reserve_entries:
            bcc_entries.append(reserve_entries[0])
    elif mode == "broadcast":
        # All externals go to BCC. To/CC internal only.
        all_routable_ext = [e for e in routable if e["is_external"]]
        all_routable_int = [e for e in routable if not e["is_external"]]
        bcc_entries = all_routable_ext
        if all_routable_int:
            sorted_int = sorted(
                all_routable_int,
                key=lambda e: (ACTION_PRIORITY.get(e["role"], 5), e["email"]),
            )
            to_entries = [sorted_int[0]]
            cc_entries = sorted_int[1:]
        else:
            # No internals available, fall back to original routing
            pass

    return {
        "to": [e["email"] for e in to_entries],
        "cc": [e["email"] for e in cc_entries],
        "bcc": [e["email"] for e in bcc_entries],
    }, sender


def apply_change(current, change_type, active_entries, reserve_entries, mode, sender_email=None):
    """Deterministic change effects. No random pops. Sender never added to any field."""
    to, cc, bcc = current["to"].copy(), current["cc"].copy(), current["bcc"].copy()

    all_entries = active_entries + reserve_entries
    by_email = {e["email"]: e for e in all_entries}
    used = set(to + cc + bcc + ([sender_email] if sender_email else []))
    reserve_emails = [e["email"] for e in reserve_entries if e["email"] not in used]

    if change_type == "person_added" and reserve_emails:
        cc.append(reserve_emails[0])
    elif change_type == "external_party_joins" and reserve_emails:
        ext = [e for e in reserve_emails if by_email.get(e, {}).get("is_external")]
        if ext:
            cc.append(ext[0])
        elif reserve_emails:
            cc.append(reserve_emails[0])
    elif change_type == "person_removed" and len(cc) > 0:
        cc.pop()
    elif change_type == "external_party_leaves" and cc:
        ext_cc = [e for e in cc if by_email.get(e, {}).get("is_external")]
        if ext_cc:
            cc.remove(ext_cc[-1])
        elif cc:
            cc.pop()
    elif change_type == "role_shift" and cc:
        cc_first = cc.pop(0)
        to.append(cc_first)
    elif change_type == "escalation" and cc:
        # Most senior CC person (highest seniority) moves to To
        cc_sorted = sorted(cc, key=lambda e: -SENIORITY.get(by_email.get(e, {}).get("role", ""), 3))
        promoted = cc_sorted[0]
        cc.remove(promoted)
        to.insert(0, promoted)
    elif change_type == "de_escalation" and len(to) > 1:
        demoted = to.pop()
        cc.append(demoted)
    elif change_type == "made_confidential":
        if mode != "compliance":
            for e in reserve_entries:
                if e["role"] in COMPLIANCE_ROLES and e["email"] not in bcc:
                    bcc.append(e["email"])
                    break
            mode = "compliance"
    elif change_type == "made_public" and bcc:
        cc.extend(bcc)
        bcc.clear()
        mode = "normal"
    elif change_type == "delegation" and to and cc:
        to[0], cc[0] = cc[0], to[0]
    elif change_type in ("urgency_increase", "scope_change"):
        if reserve_emails:
            cc.append(reserve_emails[0])
        elif cc:
            cc_first = cc.pop(0)
            to.append(cc_first)
    else:
        if reserve_emails:
            cc.append(reserve_emails[0])

    return {"to": to, "cc": cc, "bcc": bcc}, mode


def _apply_change_with_retry(current, change_type, active_entries, reserve_entries, mode, sender_email=None):
    result, new_mode = apply_change(current, change_type, active_entries, reserve_entries, mode, sender_email)
    if result != current:
        return result, change_type, new_mode
    fallbacks = sorted(c for c in CHANGE_TYPES if c != change_type)
    for alt in fallbacks:
        result, new_mode = apply_change(current, alt, active_entries, reserve_entries, mode, sender_email)
        if result != current:
            return result, alt, new_mode
    return result, change_type, mode


def _visible_recipients(answer):
    parts = []
    if answer["to"]:
        parts.append(f"To: {', '.join(answer['to'])}")
    if answer["cc"]:
        parts.append(f"CC: {', '.join(answer['cc'])}")
    return "; ".join(parts) if parts else "To: (sender only)"


def _mentionable_entries(all_entries, answer):
    """Return only entries whose email is in To or CC (not BCC)."""
    visible = set(answer["to"] + answer["cc"])
    return [e for e in all_entries if e["email"] in visible]


def _remove_sender_from_answer(answer, sender_email):
    """Remove sender from answer fields. If To becomes empty, promote first CC."""
    for field in ("to", "cc", "bcc"):
        if sender_email in answer[field]:
            answer[field].remove(sender_email)
    if not answer["to"] and answer["cc"]:
        answer["to"].append(answer["cc"].pop(0))
    return answer


def _format_email_list(entries):
    return "\n".join(f"- {e['name']} <{e['email']}> - {e['role']}" for e in entries)


def _random_date(after=None):
    """Generate a random recent business date+time, monotonically after `after` if given."""
    if after:
        base = datetime.strptime(after, "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
        delta_minutes = random.randint(15, 72 * 60)  # 15 min to 3 days
        day = base + timedelta(minutes=delta_minutes)
    else:
        base = datetime(2025, 1, 6)
        day = base + timedelta(days=random.randint(0, 250))
    while day.weekday() >= 5:
        day += timedelta(days=1)
    if not after:
        hour = random.randint(7, 18)
        minute = random.choice([0, 15, 30, 45])
        day = day.replace(hour=hour, minute=minute)
    return day.strftime("%a, %d %b %Y %H:%M:%S -0700")


# --- mode cue text (drives BCC through email content) ---

COMPLIANCE_CUES = [
    "This information is confidential and should not be forwarded.",
    "Please treat the contents of this email as confidential.",
    "This message contains sensitive information. Do not distribute.",
    "For internal review only. Do not share outside the immediate team.",
    "This is a confidential communication regarding an ongoing matter.",
]

BROADCAST_CUES = [
    "Sending this update to all external partners individually for privacy.",
    "This notice is being sent to each contact separately.",
    "Please note this is a broadcast communication to multiple external parties.",
]


# --- mode constraint helpers ---

def _ensure_compliance_in_reserve(active_entries, reserve_entries, mode):
    """If mode is compliance, ensure at least one compliance role is in reserve."""
    if mode != "compliance":
        return active_entries, reserve_entries
    has_compliance = any(e["role"] in COMPLIANCE_ROLES for e in reserve_entries)
    if has_compliance:
        return active_entries, reserve_entries
    # Swap: find a compliance role in active and a non-compliance in reserve
    for i, ae in enumerate(active_entries):
        if ae["role"] in COMPLIANCE_ROLES:
            for j, re_ in enumerate(reserve_entries):
                if re_["role"] not in COMPLIANCE_ROLES:
                    active_entries[i], reserve_entries[j] = reserve_entries[j], active_entries[i]
                    return active_entries, reserve_entries
    return active_entries, reserve_entries


def _ensure_externals_for_broadcast(active_entries, reserve_entries, mode):
    """If mode is broadcast, ensure at least 2 externals in active by swapping (preserves group size)."""
    if mode != "broadcast":
        return active_entries, reserve_entries
    ext_active = sum(1 for e in active_entries if e["is_external"])
    if ext_active >= 2:
        return active_entries, reserve_entries
    # Swap: find internals in active to swap with externals in reserve
    int_indices = [i for i, e in enumerate(active_entries) if not e["is_external"]]
    ext_reserve = [i for i, e in enumerate(reserve_entries) if e["is_external"]]
    for ri in ext_reserve:
        if ext_active >= 2 or not int_indices:
            break
        ai = int_indices.pop()
        active_entries[ai], reserve_entries[ri] = reserve_entries[ri], active_entries[ai]
        ext_active += 1
    return active_entries, reserve_entries


# --- custom column: builds email_list + ground truth answers ---

@dd.custom_column_generator(
    required_columns=[
        "start_people", "person_1", "person_2", "person_3", "person_4",
        "person_5", "person_6", "person_7",
        "mode", "hierarchy", "audience",
        "change_turn_2", "change_turn_3",
    ],
    side_effect_columns=[
        "answer_2", "answer_3", "email_list",
        "mentionable_list_1", "mentionable_list_2", "mentionable_list_3",
        "sender_1", "sender_2", "sender_3",
        "to_name_1", "to_name_2", "to_name_3",
        "date_1", "date_2", "date_3",
        "mode_cue_1", "mode_cue_2", "mode_cue_3",
        "actual_change_2", "actual_change_3",
    ],
)
def build_ground_truth(row):
    n = int(float(row["start_people"]))
    company_domain = random.choice(COMPANY_DOMAINS)
    external_domain = random.choice(EXTERNAL_DOMAINS)

    persons = [row[f"person_{i}"] for i in range(1, 8)]
    all_entries = _build_email_list(persons, row["audience"], company_domain, external_domain)

    active_entries = all_entries[:n]
    reserve_entries = all_entries[n:]
    mode = row["mode"]

    # Fix mode/audience mismatch: broadcast requires externals
    if mode == "broadcast" and row["audience"] not in EXTERNAL_AUDIENCES:
        mode = "normal"

    # Guarantee mode constraints
    active_entries, reserve_entries = _ensure_compliance_in_reserve(active_entries, reserve_entries, mode)
    active_entries, reserve_entries = _ensure_externals_for_broadcast(active_entries, reserve_entries, mode)

    by_email = {e["email"]: e for e in all_entries}
    row["email_list"] = _format_email_list(all_entries)

    # Turn 1
    a1, sender_1 = distribute_emails(active_entries, reserve_entries, mode, row["hierarchy"])
    row["answer_1"] = json.dumps(a1)
    row["to_name_1"] = by_email[a1["to"][0]]["name"] if a1["to"] else ""
    mentionable_1 = _mentionable_entries(all_entries, a1)
    row["mentionable_list_1"] = _format_email_list(mentionable_1)
    row["sender_1"] = f"{sender_1['name']} <{sender_1['email']}>"
    row["date_1"] = _random_date()
    if mode == "compliance":
        row["mode_cue_1"] = random.choice(COMPLIANCE_CUES)
    elif mode == "broadcast":
        row["mode_cue_1"] = random.choice(BROADCAST_CUES)
    else:
        row["mode_cue_1"] = ""

    sender_email = sender_1["email"]

    # Turn 2
    a2, ch2, mode = _apply_change_with_retry(a1, row["change_turn_2"], active_entries, reserve_entries, mode, sender_email)
    row["actual_change_2"] = ch2
    # Pick reply sender from To+CC, different from turn 1 sender
    reply_pool = [e for e in all_entries if e["email"] in a2["to"] + a2["cc"] and e["email"] != sender_email]
    sender_2 = reply_pool[0] if reply_pool else sender_1
    s2_email = sender_2["email"]
    _remove_sender_from_answer(a2, s2_email)
    row["answer_2"] = json.dumps(a2)
    row["to_name_2"] = by_email[a2["to"][0]]["name"] if a2["to"] else ""
    mentionable_2 = _mentionable_entries(all_entries, a2)
    row["mentionable_list_2"] = _format_email_list(mentionable_2)
    row["sender_2"] = f"{sender_2['name']} <{sender_2['email']}>"
    row["date_2"] = _random_date(after=row["date_1"])
    if mode == "compliance":
        row["mode_cue_2"] = random.choice(COMPLIANCE_CUES)
    else:
        row["mode_cue_2"] = ""

    # Turn 3
    a3, ch3, mode = _apply_change_with_retry(a2, row["change_turn_3"], active_entries, reserve_entries, mode, s2_email)
    row["actual_change_3"] = ch3
    # Pick reply sender, different from turn 2 sender
    reply_pool_3 = [e for e in all_entries if e["email"] in a3["to"] + a3["cc"] and e["email"] != s2_email]
    sender_3 = reply_pool_3[0] if reply_pool_3 else sender_2
    s3_email = sender_3["email"]
    _remove_sender_from_answer(a3, s3_email)
    row["answer_3"] = json.dumps(a3)
    row["to_name_3"] = by_email[a3["to"][0]]["name"] if a3["to"] else ""
    mentionable_3 = _mentionable_entries(all_entries, a3)
    row["mentionable_list_3"] = _format_email_list(mentionable_3)
    row["sender_3"] = f"{sender_3['name']} <{sender_3['email']}>"
    row["date_3"] = _random_date(after=row["date_2"])
    if mode == "compliance":
        row["mode_cue_3"] = random.choice(COMPLIANCE_CUES)
    else:
        row["mode_cue_3"] = ""

    return row


# --- validation ---

def validate_row(row):
    """Returns True if row passes all checks."""
    # Build name lookup from email_list
    email_to_name = {}
    all_names = []
    for line in row["email_list"].split("\n"):
        if "<" in line:
            name = line.split("<")[0].strip("- ").strip()
            email = line.split("<")[1].split(">")[0]
            email_to_name[email] = name
            all_names.append(name)

    for turn in [1, 2, 3]:
        answer = json.loads(row[f"answer_{turn}"])
        question = row[f"question_{turn}"]

        # Check To is not empty
        if not answer["to"]:
            return False

        # Check no field overlap
        to_set = set(answer["to"])
        cc_set = set(answer["cc"])
        bcc_set = set(answer["bcc"])
        if to_set & cc_set or to_set & bcc_set or cc_set & bcc_set:
            return False

        # Check BCC names (full and first) don't appear in email text
        for bcc_email in answer["bcc"]:
            name = email_to_name.get(bcc_email, "")
            if not name:
                continue
            if name.lower() in question.lower():
                return False
            first_name = name.split()[0]
            if len(first_name) > 2 and first_name.lower() in question.lower():
                return False

        # Check sender email not in any answer field
        # (sender is in the From: header of the question)
        from_match = re.search(r"From:.*<([^>]+)>", question)
        if from_match:
            sender_email = from_match.group(1)
            if sender_email in to_set | cc_set | bcc_set:
                return False

        # Check only mentionable names appear (To+CC, not BCC or unused roster)
        allowed_emails = to_set | cc_set
        # Also allow sender name
        allowed_names = {email_to_name.get(e, "").lower() for e in allowed_emails}
        if from_match:
            allowed_names.add(email_to_name.get(from_match.group(1), "").lower())
        allowed_names.discard("")

        for name in all_names:
            if name.lower() in allowed_names:
                continue
            if name.lower() in question.lower():
                return False

        # If email contains a broadcast/compliance cue, BCC must be non-empty
        q_lower = question.lower()
        has_broadcast_cue = (
            any(cue.lower() in q_lower for cue in BROADCAST_CUES)
            or "contact separately" in q_lower
            or "broadcast communication" in q_lower
        )
        has_compliance_cue = (
            any(cue.lower() in q_lower for cue in COMPLIANCE_CUES)
            or "do not forward" in q_lower
            or "do not distribute" in q_lower
        )
        if (has_broadcast_cue or has_compliance_cue) and not bcc_set:
            return False

    return True


# --- pipeline ---

LLM_RULES = """RULES (follow ALL of these exactly):
- Write ONLY the email (Date, From, Subject, then body). No To/CC/BCC headers.
- ONLY mention people from the "People to mention" list. Do NOT mention, greet, reference, or name anyone else. This is critical.
- NEVER mention BCC, blind copy, or who is secretly copied. Do not hint at hidden recipients.
- NEVER use brackets, braces, or any placeholder syntax - no [Name], [Date], [Project], {System}, [Your Name], [Sender], etc. Every detail must be concrete and specific.
- Spell every person's name EXACTLY as shown in the people list. Do not alter, shorten, or misspell names."""


def build_pipeline():
    designer = DataDesigner()
    config = dd.DataDesignerConfigBuilder()

    # samplers
    config.add_column(dd.SamplerColumnConfig(
        name="start_people", sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=4, high=6, decimal_places=0), drop=True,
    ))
    for i in range(1, 8):
        config.add_column(dd.SamplerColumnConfig(
            name=f"person_{i}", sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
            params=dd.PersonFromFakerSamplerParams(), drop=True,
        ))
    for col, vals in [
        ("department", DEPARTMENTS),
        ("scenario_type", list(SCENARIO_TYPES.keys())),
        ("audience", AUDIENCES),
        ("mode", MODES),
        ("hierarchy", HIERARCHIES),
    ]:
        config.add_column(dd.SamplerColumnConfig(
            name=col, sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=vals), drop=True,
        ))
    config.add_column(dd.SamplerColumnConfig(
        name="scenario_subtype", sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(category="scenario_type", values=SCENARIO_TYPES),
        drop=True,
    ))
    for turn in ["turn_2", "turn_3"]:
        config.add_column(dd.SamplerColumnConfig(
            name=f"change_{turn}", sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=CHANGE_TYPES), drop=True,
        ))

    # custom column - ground truth
    config.add_column(dd.CustomColumnConfig(name="answer_1", generator_function=build_ground_truth))

    # llm columns - email content (v2: mentionable names only, Date/From headers)
    config.add_column(dd.LLMTextColumnConfig(
        name="question_1", model_alias="openrouter-text",
        prompt="""Write a realistic work email for this context:
Department: {{ department }}
Scenario: {{ scenario_type }} ({{ scenario_subtype }})
Audience: {{ audience }}, Hierarchy: {{ hierarchy }}

People to mention (ONLY these people may appear in the email):
{{ mentionable_list_1 }}

Sender: {{ sender_1 }}
Primary action owner (must be directly tasked or addressed in the email): {{ to_name_1 }}
{% if mode_cue_1 %}
The email should include this note: "{{ mode_cue_1 }}"
{% endif %}

""" + LLM_RULES + """

Format (use exactly this structure):

Date: {{ date_1 }}
From: {{ sender_1 }}
Subject: <one concrete subject line>

<email body, 2-4 paragraphs, referencing people by name. The email must explicitly task or address {{ to_name_1 }}.>""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_2", model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

The situation changed: {{ actual_change_2 }}

People to mention (ONLY these people may appear in the reply):
{{ mentionable_list_2 }}

Sender of the reply: {{ sender_2 }}
Primary action owner (must be directly tasked or addressed in the reply): {{ to_name_2 }}
{% if mode_cue_2 %}
The reply should include this note: "{{ mode_cue_2 }}"
{% endif %}

""" + LLM_RULES + """

Format (use exactly this structure):

Date: {{ date_2 }}
From: {{ sender_2 }}
Subject: Re: <original subject>

<reply body, 1-2 paragraphs, referencing people by name. The reply must explicitly task or address {{ to_name_2 }}.>""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_3", model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

{{ question_2 }}

The situation changed again: {{ actual_change_3 }}

People to mention (ONLY these people may appear in the reply):
{{ mentionable_list_3 }}

Sender of the reply: {{ sender_3 }}
Primary action owner (must be directly tasked or addressed in the reply): {{ to_name_3 }}
{% if mode_cue_3 %}
The reply should include this note: "{{ mode_cue_3 }}"
{% endif %}

""" + LLM_RULES + """

Format (use exactly this structure):

Date: {{ date_3 }}
From: {{ sender_3 }}
Subject: Re: <original subject>

<reply body, 1-2 paragraphs, referencing people by name. The reply must explicitly task or address {{ to_name_3 }}.>""",
    ))

    return designer, config


KEEP_COLUMNS = [
    "email_list", "question_1", "question_2", "question_3",
    "answer_1", "answer_2", "answer_3",
]


def main():
    parser = argparse.ArgumentParser(description="Generate email-to-cc-bcc v2 dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preview", type=int, nargs="?", const=5, metavar="N",
                       help="preview N rows (default 5)")
    group.add_argument("--generate", type=int, metavar="N",
                       help="generate N rows and save to parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "v2.parquet",
                        help="output parquet path (default: nuggets/email-to-cc-bcc/v2.parquet)")
    args = parser.parse_args()

    designer, config = build_pipeline()

    if args.generate:
        print(f"generating {args.generate} rows...")
        results = designer.create(config_builder=config, num_records=args.generate)
        df = results.load_dataset()[KEEP_COLUMNS]

        # Validate
        total = len(df)
        valid_mask = df.apply(validate_row, axis=1)
        rejected = total - valid_mask.sum()
        df = df[valid_mask].reset_index(drop=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"done - {len(df)} rows saved to {args.output} ({rejected} rejected)")
    else:
        n = args.preview or 5
        print(f"previewing {n} rows...")
        preview = designer.preview(config_builder=config, num_records=n)
        drop_cols = [c for c in preview.dataset.columns if c not in KEEP_COLUMNS]
        preview.dataset.drop(columns=drop_cols, inplace=True)
        for i in range(len(preview.dataset)):
            preview.display_sample_record(i)


if __name__ == "__main__":
    main()
