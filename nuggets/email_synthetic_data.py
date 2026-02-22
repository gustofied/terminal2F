"""
Email-to-CC-BCC synthetic dataset generator.

The task: given an email and a email_list of 7 people (name, email, role),
assign the right people to To, CC, and BCC. Some people may not be in any field.

Single-phase DataDesigner pipeline:
  1. Samplers — people, context dimensions, change types
  2. Custom column — build email_list (7 people with emails + roles) + ground truth assignments
  3. LLM columns — write actual email content (subject + body)

Dataset columns (7):
  email_list     — all 7 people, always present (name <email> — role)
  question_1 — initial email (subject + body)
  question_2 — reply email
  question_3 — second reply
  answer_1   — {"to": ["email"], "cc": [...], "bcc": [...]}
  answer_2   — updated assignments
  answer_3   — updated assignments

The environment assembles the full prompt at runtime:
  "Here are the potential recipients: [email_list]. Here is the email: [question]. Assign To, CC, BCC."

Setup (separate venv — project pins pyarrow>=22 which conflicts with data-designer's <20):
  cd nuggets
  uv venv
  uv pip install 'data-designer>=0.5.1' faker
  .venv/bin/python email_synthetic_data.py
"""

import json
import random

import data_designer.config as dd
from data_designer.interface.data_designer import DataDesigner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_RECORDS = 10_000

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

# Roles grouped by whether they're internal or external-facing
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

CHANGE_TYPES = [
    "person_added", "person_removed", "role_shift",
    "escalation", "de_escalation", "made_confidential",
    "made_public", "delegation", "external_party_joins",
    "external_party_leaves", "urgency_increase", "scope_change",
]


# ---------------------------------------------------------------------------
# People + email generation
# ---------------------------------------------------------------------------


def _make_email(first: str, last: str, domain: str, taken: set[str]) -> str:
    """Generate unique email from name + domain."""
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
        email = f"{local}@{domain}"
        if email not in taken:
            taken.add(email)
            return email
    # Fallback: add random digits
    local = f"{first_l}.{last_l}{random.randint(10, 99)}"
    email = f"{local}@{domain}"
    taken.add(email)
    return email


def _build_email_list(
    persons: list,
    audience: str,
    company_domain: str,
    external_domain: str,
) -> list[dict]:
    """Build 7-person email_list with names, emails, and roles.

    First 5 are internal (or mixed based on audience), last 2 are external/reserve.
    """
    has_external = audience in ("with_client", "with_vendor", "with_partner", "with_contractor")
    taken_emails: set[str] = set()
    entries = []

    for i, person in enumerate(persons):
        if isinstance(person, dict):
            first = person.get("first_name", "Unknown")
            last = person.get("last_name", "Person")
        else:
            parts = str(person).split()
            first = parts[0] if parts else "Unknown"
            last = parts[-1] if len(parts) > 1 else "Person"

        name = f"{first} {last}"

        # Decide internal vs external
        # If has_external: persons 6-7 (index 5-6) are always external,
        # and one of the first 5 is external too (the client/vendor contact)
        if i >= 5:
            # Reserve people — external if audience calls for it
            is_ext = has_external
        elif i == 1 and has_external:
            # Make person 2 the external contact in the initial group
            is_ext = True
        else:
            is_ext = False

        if is_ext:
            role = random.choice(EXTERNAL_ROLES)
            domain = external_domain
        else:
            role = random.choice(INTERNAL_ROLES)
            # Mostly company domain, occasionally personal (contractors, freelancers)
            if random.random() < 0.1:
                domain = random.choice(PERSONAL_DOMAINS)
            else:
                domain = company_domain

        email = _make_email(first, last, domain, taken_emails)
        entries.append({"name": name, "email": email, "role": role})

    return entries


# ---------------------------------------------------------------------------
# Ground truth logic
# ---------------------------------------------------------------------------


def distribute_emails(active_emails, sensitivity, hierarchy):
    """Distribute active email addresses into to/cc/bcc."""
    if not active_emails:
        return {"to": [], "cc": [], "bcc": []}

    shuffled = active_emails.copy()
    random.shuffle(shuffled)

    to = [shuffled[0]]
    rest = shuffled[1:]

    if not rest:
        return {"to": to, "cc": [], "bcc": []}

    if sensitivity in ("confidential", "restricted"):
        bcc_count = max(1, len(rest) // 2)
        bcc = rest[:bcc_count]
        cc = rest[bcc_count:]
    elif sensitivity == "internal":
        if len(rest) >= 3:
            bcc = rest[-1:]
            cc = rest[:-1]
        else:
            bcc = []
            cc = rest
    else:
        bcc = []
        cc = rest

    if hierarchy == "upward" and cc:
        to.append(cc.pop(0))
    elif hierarchy == "downward" and len(to) > 1:
        cc.insert(0, to.pop())

    return {"to": to, "cc": cc, "bcc": bcc}


def apply_change(current, change_type, reserve_emails):
    """Apply a change event. Mutates reserve_emails."""
    to = current["to"].copy()
    cc = current["cc"].copy()
    bcc = current["bcc"].copy()
    all_active = to + cc + bcc

    if change_type == "person_added" and reserve_emails:
        cc.append(reserve_emails.pop(0))

    elif change_type == "external_party_joins" and reserve_emails:
        to.append(reserve_emails.pop(0))

    elif change_type == "person_removed" and len(all_active) > 2:
        if cc:
            cc.pop(random.randrange(len(cc)))
        elif bcc:
            bcc.pop(random.randrange(len(bcc)))
        elif len(to) > 1:
            to.pop()

    elif change_type == "external_party_leaves":
        if cc:
            cc.pop(random.randrange(len(cc)))
        elif len(to) > 1:
            to.pop()

    elif change_type == "role_shift":
        if cc and to:
            to.append(cc.pop(0))
        elif len(to) > 1:
            cc.append(to.pop())

    elif change_type == "escalation":
        if cc:
            to.insert(0, cc.pop(0))

    elif change_type == "de_escalation":
        if len(to) > 1:
            cc.append(to.pop())

    elif change_type == "made_confidential":
        if cc:
            bcc.append(cc.pop(random.randrange(len(cc))))

    elif change_type == "made_public":
        if bcc:
            cc.append(bcc.pop(0))

    elif change_type == "delegation":
        if to and cc:
            to[0], cc[0] = cc[0], to[0]

    elif change_type in ("urgency_increase", "scope_change"):
        if reserve_emails:
            to.append(reserve_emails.pop(0))
        elif cc and to:
            to.append(cc.pop(0))

    else:
        if cc and to:
            to.append(cc.pop(0))
        elif len(to) > 1:
            cc.append(to.pop())

    return {"to": to, "cc": cc, "bcc": bcc}


# ---------------------------------------------------------------------------
# Custom column: build email_list + ground truth
# ---------------------------------------------------------------------------


def _format_email_list(entries: list[dict]) -> str:
    """Format people email_list as text block."""
    lines = []
    for e in entries:
        lines.append(f"- {e['name']} <{e['email']}> — {e['role']}")
    return "\n".join(lines)


@dd.custom_column_generator(
    required_columns=[
        "start_people", "person_1", "person_2", "person_3", "person_4",
        "person_5", "person_6", "person_7",
        "sensitivity", "hierarchy", "audience",
        "change_turn_2", "change_turn_3",
    ],
    side_effect_columns=["answer_2", "answer_3", "email_list"],
)
def build_ground_truth(row):
    n = int(float(row["start_people"]))
    audience = row["audience"]

    # Pick one company domain and one external domain for this row
    company_domain = random.choice(COMPANY_DOMAINS)
    external_domain = random.choice(EXTERNAL_DOMAINS)

    # Build full 7-person email_list
    persons = [row[f"person_{i}"] for i in range(1, 8)]
    all_entries = _build_email_list(persons, audience, company_domain, external_domain)

    active_entries = all_entries[:n]
    reserve_entries = all_entries[n:]

    active_emails = [e["email"] for e in active_entries]
    reserve_emails = [e["email"] for e in reserve_entries]

    # Roster: always all 7 people (survives in final dataset)
    row["email_list"] = _format_email_list(all_entries)

    # Turn 1
    answer_1 = distribute_emails(active_emails, row["sensitivity"], row["hierarchy"])
    row["answer_1"] = json.dumps(answer_1)

    # Turn 2
    answer_2 = apply_change(answer_1, row["change_turn_2"], reserve_emails)
    row["answer_2"] = json.dumps(answer_2)

    # Turn 3
    answer_3 = apply_change(answer_2, row["change_turn_3"], reserve_emails)
    row["answer_3"] = json.dumps(answer_3)

    return row


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    data_designer = DataDesigner()
    config = dd.DataDesignerConfigBuilder()

    # --- Step 1: Samplers ---

    config.add_column(dd.SamplerColumnConfig(
        name="start_people",
        sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=2, high=5, decimal_places=0),
        drop=True,
    ))

    for i in range(1, 8):
        config.add_column(dd.SamplerColumnConfig(
            name=f"person_{i}",
            sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
            params=dd.PersonFromFakerSamplerParams(),
            drop=True,
        ))

    config.add_column(dd.SamplerColumnConfig(
        name="department",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=DEPARTMENTS),
        drop=True,
    ))

    config.add_column(dd.SamplerColumnConfig(
        name="scenario_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=list(SCENARIO_TYPES.keys())),
        drop=True,
    ))

    config.add_column(dd.SamplerColumnConfig(
        name="scenario_subtype",
        sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(
            category="scenario_type",
            values=SCENARIO_TYPES,
        ),
        drop=True,
    ))

    config.add_column(dd.SamplerColumnConfig(
        name="audience",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=AUDIENCES),
        drop=True,
    ))

    config.add_column(dd.SamplerColumnConfig(
        name="sensitivity",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=SENSITIVITIES),
        drop=True,
    ))

    config.add_column(dd.SamplerColumnConfig(
        name="hierarchy",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=HIERARCHIES),
        drop=True,
    ))

    for turn in ["turn_2", "turn_3"]:
        config.add_column(dd.SamplerColumnConfig(
            name=f"change_{turn}",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=CHANGE_TYPES),
            drop=True,
        ))

    # --- Step 2: Custom column — email_list + ground truth ---

    config.add_column(dd.CustomColumnConfig(
        name="answer_1",
        generator_function=build_ground_truth,
    ))

    # --- Step 3: LLM columns — actual email content ---
    # The LLM generates email text only. The environment assembles the full
    # prompt (email_list + email + instruction) at runtime.

    config.add_column(dd.LLMTextColumnConfig(
        name="question_1",
        model_alias="openrouter-text",
        prompt="""Write a realistic work email for this context:
Department: {{ department }}
Scenario: {{ scenario_type }} ({{ scenario_subtype }})
Audience: {{ audience }}, Sensitivity: {{ sensitivity }}, Hierarchy: {{ hierarchy }}

People involved:
{{ email_list }}

Recipient assignments (use to guide which people to reference, do NOT include in output): {{ answer_1 }}

Write ONLY the email in this exact format:

Subject: <one line>

<email body, 2-4 paragraphs>

Reference the relevant people by name naturally in the email body. Do not include any headers like From/To/CC. Just Subject and body.""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_2",
        model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

The situation changed: {{ change_turn_2 }}
People involved:
{{ email_list }}
Updated assignments (guide only, do NOT include): {{ answer_2 }}

Write ONLY a reply in this exact format:

Subject: Re: <original subject>

<reply body, 1-2 paragraphs>

Reference relevant people by name. Do not include From/To/CC headers.""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_3",
        model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

{{ question_2 }}

The situation changed again: {{ change_turn_3 }}
People involved:
{{ email_list }}
Updated assignments (guide only, do NOT include): {{ answer_3 }}

Write ONLY another reply in this exact format:

Subject: Re: <original subject>

<reply body, 1-2 paragraphs>

Reference relevant people by name. Do not include From/To/CC headers.""",
    ))

    return data_designer, config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_designer, config = build_pipeline()

    # Preview
    print("Generating preview (5 records)...")
    preview = data_designer.preview(config_builder=config, num_records=5)
    for i in range(len(preview.dataset)):
        preview.display_sample_record(i)

    # Full generation (uncomment when ready)
    # print(f"Generating {NUM_RECORDS} records...")
    # results = data_designer.create(config_builder=config, num_records=NUM_RECORDS)
    # dataset = results.load_dataset()
    # print(f"Done. Dataset shape: {dataset.shape}")
