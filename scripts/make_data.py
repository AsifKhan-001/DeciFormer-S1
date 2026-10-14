import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from system_one.data import DEFAULT_QUESTION  # noqa: E402

APPS = ["WhatsApp", "Instagram", "the mobile app", "the website", "the dashboard", "Gmail", "Zoom",
        "the checkout page", "the admin panel", "Slack"]
PREFIX = ["", "Hi, ", "Hello, ", "Hey team, ", "Please help: ", "Urgent: "]
SUFFIX = ["", ".", " please", " as soon as possible", " since yesterday", " and nobody replied"]
LABELS = {0: "technical_team", 1: "finance_team", 2: "account_team", 3: "advertisement_team", 4: "general_support"}
# a leading "~" marks a hard / ambiguous template
TEMPLATES = {
    0: ["{app} keeps crashing when I open it", "The login page is not opening in {app}",
        "I get error 500 while loading {app}", "{app} is very slow and freezes on startup",
        "~The payment page is not loading in {app}", "Notifications from {app} stopped working after the update",
        "I cannot upload files in {app}, the button does nothing", "{app} shows a blank white screen",
        "The search feature in {app} returns an error", "Video calls in {app} keep disconnecting",
        "~The verification screen in {app} never finishes loading"],
    1: ["I was charged twice for my subscription in {app}", "Please send me the invoice for last month's payment on {app}",
        "My refund for the order from {app} has not arrived yet",
        "~The payment page shows I was charged but no receipt was issued in {app}",
        "Why was I billed more than the quoted price in {app}", "I want to change the billing cycle of my {app} plan",
        "~My card was declined but the money left my bank for {app}", "Can you explain the tax charged on my {app} bill",
        "I need a GST invoice for my {app} purchase", "Cancel my auto-renewal and refund the unused amount on {app}"],
    2: ["I forgot my password for {app}", "My {app} account was locked after too many attempts",
        "I want to change the email address linked to my {app} account", "Please delete my {app} account permanently",
        "~I cannot log in because my account says suspended on {app}",
        "How do I enable two factor authentication on {app}",
        "Someone changed my username on {app} without my permission", "Merge my two {app} profiles into one",
        "I want to update the phone number on my {app} profile",
        "~The verification code for my {app} account never arrives"],
    3: ["I want to run a promotion campaign for my brand on {app}", "How much does a banner ad on {app} cost",
        "My sponsored post on {app} was rejected", "I need the reach and click report for my ad on {app}",
        "Can we advertise our new product to users of {app}",
        "~My ad campaign on {app} stopped delivering impressions", "I want to become a sponsor partner for {app}",
        "Please change the target audience of my ad on {app}", "What are the ad formats supported by {app}",
        "~I was charged for ads on {app} that never ran"],
    4: ["What are your customer support working hours", "I want to give feedback about {app}",
        "Where can I find the user guide for {app}", "Is {app} available in my country",
        "How do I contact your office", "Can you tell me about the company and its services",
        "I have a general question about how {app} works", "Do you have a careers page",
        "What languages does {app} support", "Where is your head office located"],
}


def build(seed: int = 42):
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, templates in TEMPLATES.items():
        bases = [(t.lstrip("~"), t.startswith("~"), a) for t in templates for a in APPS]
        rng.shuffle(bases)
        n = len(bases)
        cuts = {"train": bases[: int(n * 0.7)], "val": bases[int(n * 0.7): int(n * 0.85)], "test": bases[int(n * 0.85):]}
        for name, items in cuts.items():
            for tmpl, hard, app in items:
                text = tmpl.format(app=app)
                if "{app}" not in tmpl:
                    text = tmpl
                variants = {(p, s) for p in rng.sample(PREFIX, 3) for s in rng.sample(SUFFIX, 2)}
                for p, s in sorted(variants):
                    q = f"{p}{text[0].lower() + text[1:] if p else text}{s}"
                    splits[name].append({"query": q, "label": label, "difficulty": "hard" if hard else "easy",
                                         "domain": "support"})
    seen = set()
    for name in ("train", "val", "test"):
        uniq = []
        for r in splits[name]:
            if r["query"] not in seen:
                seen.add(r["query"])
                uniq.append(r)
        rng.shuffle(uniq)
        splits[name] = uniq
    return splits, rng


def main():
    out = Path(__file__).resolve().parents[1] / "data"
    (out / "processed").mkdir(parents=True, exist_ok=True)
    splits, rng = build()
    (out / "labels.json").write_text(json.dumps({str(k): v for k, v in LABELS.items()}, indent=2))
    names = [LABELS[i].replace("_", " ") for i in range(len(LABELS))]
    for name, rows in splits.items():
        with open(out / "processed" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(out / "processed" / f"{name}_dyn.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                order = list(range(len(names)))
                rng.shuffle(order)
                f.write(json.dumps({"context": r["query"], "question": DEFAULT_QUESTION,
                                    "options": [names[i] for i in order], "target": order.index(r["label"]),
                                    "difficulty": r["difficulty"]}) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
