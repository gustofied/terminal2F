"""Generates the dataset for the email_to_cc_bcc RL environment."""

import argparse
import json
import random
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
        else:
            role = int_roles[ii % len(int_roles)]
            ii += 1
            domain = random.choice(PERSONAL_DOMAINS) if random.random() < 0.1 else company_domain

        entries.append({
            "name": f"{first} {last}",
            "email": _make_email(first, last, domain, taken),
            "role": role,
        })

    return entries


# --- ground truth ---

def distribute_emails(active_emails, sensitivity, hierarchy):
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
        bcc, cc = rest[:bcc_count], rest[bcc_count:]
    elif sensitivity == "internal" and len(rest) >= 3:
        bcc, cc = rest[-1:], rest[:-1]
    else:
        bcc, cc = [], rest

    if hierarchy == "upward" and cc:
        to.append(cc.pop(0))
    elif hierarchy == "downward" and len(to) > 1:
        cc.insert(0, to.pop())

    return {"to": to, "cc": cc, "bcc": bcc}


def apply_change(current, change_type, reserve_emails):
    to, cc, bcc = current["to"].copy(), current["cc"].copy(), current["bcc"].copy()
    all_active = to + cc + bcc

    if change_type == "person_added" and reserve_emails:
        cc.append(reserve_emails.pop(0))
    elif change_type == "external_party_joins" and reserve_emails:
        to.append(reserve_emails.pop(0))
    elif change_type == "person_removed" and len(all_active) > 2:
        for lst in [cc, bcc]:
            if lst:
                lst.pop(random.randrange(len(lst)))
                break
        else:
            if len(to) > 1:
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
    elif change_type == "escalation" and cc:
        to.insert(0, cc.pop(0))
    elif change_type == "de_escalation" and len(to) > 1:
        cc.append(to.pop())
    elif change_type == "made_confidential" and cc:
        bcc.append(cc.pop(random.randrange(len(cc))))
    elif change_type == "made_public" and bcc:
        cc.append(bcc.pop(0))
    elif change_type == "delegation" and to and cc:
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


def _apply_change_with_retry(current, change_type, reserve_emails):
    result = apply_change(current, change_type, reserve_emails)
    if result != current:
        return result, change_type
    fallbacks = [c for c in CHANGE_TYPES if c != change_type]
    random.shuffle(fallbacks)
    for alt in fallbacks:
        reserve_copy = reserve_emails.copy()
        result = apply_change(current, alt, reserve_copy)
        if result != current:
            reserve_emails.clear()
            reserve_emails.extend(reserve_copy)
            return result, alt
    return result, change_type


def _visible_recipients(answer):
    parts = []
    if answer["to"]:
        parts.append(f"To: {', '.join(answer['to'])}")
    if answer["cc"]:
        parts.append(f"CC: {', '.join(answer['cc'])}")
    return "; ".join(parts) if parts else "To: (sender only)"


def _format_email_list(entries):
    return "\n".join(f"- {e['name']} <{e['email']}> — {e['role']}" for e in entries)


# --- custom column: builds email_list + ground truth answers ---

@dd.custom_column_generator(
    required_columns=[
        "start_people", "person_1", "person_2", "person_3", "person_4",
        "person_5", "person_6", "person_7",
        "sensitivity", "hierarchy", "audience",
        "change_turn_2", "change_turn_3",
    ],
    side_effect_columns=[
        "answer_2", "answer_3", "email_list",
        "llm_hint_1", "llm_hint_2", "llm_hint_3",
        "actual_change_2", "actual_change_3",
    ],
)
def build_ground_truth(row):
    n = int(float(row["start_people"]))
    company_domain = random.choice(COMPANY_DOMAINS)
    external_domain = random.choice(EXTERNAL_DOMAINS)

    persons = [row[f"person_{i}"] for i in range(1, 8)]
    all_entries = _build_email_list(persons, row["audience"], company_domain, external_domain)

    active_emails = [e["email"] for e in all_entries[:n]]
    reserve_emails = [e["email"] for e in all_entries[n:]]

    row["email_list"] = _format_email_list(all_entries)

    a1 = distribute_emails(active_emails, row["sensitivity"], row["hierarchy"])
    row["answer_1"] = json.dumps(a1)
    row["llm_hint_1"] = _visible_recipients(a1)

    a2, ch2 = _apply_change_with_retry(a1, row["change_turn_2"], reserve_emails)
    row["answer_2"] = json.dumps(a2)
    row["llm_hint_2"] = _visible_recipients(a2)
    row["actual_change_2"] = ch2

    a3, ch3 = _apply_change_with_retry(a2, row["change_turn_3"], reserve_emails)
    row["answer_3"] = json.dumps(a3)
    row["llm_hint_3"] = _visible_recipients(a3)
    row["actual_change_3"] = ch3

    return row


# --- pipeline ---

LLM_RULES = """RULES (follow ALL of these exactly):
- Write ONLY Subject line + body. No From/To/CC/BCC headers.
- NEVER mention BCC, blind copy, or who is secretly copied. Do not hint at hidden recipients.
- NEVER use brackets, braces, or any placeholder syntax — no [Name], [Date], [Project], {System}, [Your Name], [Sender], etc. Every detail must be concrete and specific. Invent realistic names, dates, project names, and system names as needed.
- For sign-offs: use a real name from the people list as the sender, or just "Best," with no name. NEVER write [Your Name] or [Sender Name].
- Spell every person's name EXACTLY as shown in the people list. Do not alter, shorten, or misspell names."""


def build_pipeline():
    designer = DataDesigner()
    config = dd.DataDesignerConfigBuilder()

    # samplers
    config.add_column(dd.SamplerColumnConfig(
        name="start_people", sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=2, high=6, decimal_places=0), drop=True,
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
        ("sensitivity", SENSITIVITIES),
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

    # custom column — ground truth
    config.add_column(dd.CustomColumnConfig(name="answer_1", generator_function=build_ground_truth))

    # llm columns — email content
    config.add_column(dd.LLMTextColumnConfig(
        name="question_1", model_alias="openrouter-text",
        prompt="""Write a realistic work email for this context:
Department: {{ department }}
Scenario: {{ scenario_type }} ({{ scenario_subtype }})
Audience: {{ audience }}, Sensitivity: {{ sensitivity }}, Hierarchy: {{ hierarchy }}

People involved:
{{ email_list }}

Visible recipients (use to guide which people to mention — do NOT include this in the email): {{ llm_hint_1 }}

""" + LLM_RULES + """

Format:

Subject: <one concrete subject line>

<email body, 2-4 paragraphs, referencing relevant people by name>""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_2", model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

The situation changed: {{ actual_change_2 }}
People involved:
{{ email_list }}
Visible recipients now (guide only, do NOT include): {{ llm_hint_2 }}

""" + LLM_RULES + """

Format:

Subject: Re: <original subject>

<reply body, 1-2 paragraphs, referencing relevant people by name>""",
    ))

    config.add_column(dd.LLMTextColumnConfig(
        name="question_3", model_alias="openrouter-text",
        prompt="""Given this email thread:

{{ question_1 }}

{{ question_2 }}

The situation changed again: {{ actual_change_3 }}
People involved:
{{ email_list }}
Visible recipients now (guide only, do NOT include): {{ llm_hint_3 }}

""" + LLM_RULES + """

Format:

Subject: Re: <original subject>

<reply body, 1-2 paragraphs, referencing relevant people by name>""",
    ))

    return designer, config


KEEP_COLUMNS = [
    "email_list", "question_1", "question_2", "question_3",
    "answer_1", "answer_2", "answer_3",
]


def main():
    parser = argparse.ArgumentParser(description="Generate email-to-cc-bcc dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preview", type=int, nargs="?", const=5, metavar="N",
                       help="preview N rows (default 5)")
    group.add_argument("--generate", type=int, metavar="N",
                       help="generate N rows and save to parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "email_to_cc_bcc.parquet",
                        help="output parquet path (default: nuggets/email_to_cc_bcc.parquet)")
    args = parser.parse_args()

    designer, config = build_pipeline()

    if args.generate:
        print(f"generating {args.generate} rows...")
        results = designer.create(config_builder=config, num_records=args.generate)
        df = results.load_dataset()[KEEP_COLUMNS]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"done — {len(df)} rows saved to {args.output}")
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
