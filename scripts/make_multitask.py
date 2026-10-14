import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_data import LABELS, build  # noqa: E402

AMOUNTS = ["100", "500", "1000", "5000", "$250", "$1,000", "$10,000"]
ITEMS = ["iPhone", "laptop", "Amazon gift card", "cash prize", "vacation package", "Netflix subscription"]
APPS = ["WhatsApp", "Gmail", "the dashboard", "the mobile app", "Zoom", "the website"]
NAMES = [v.replace("_", " ") for v in LABELS.values()]

CLASSES = {
    "spam": [
        "Click now and win a {amt} {item}", "Congratulations, you have won a {item}, claim your prize here",
        "Limited time offer: get a free {item}, send your bank details", "Your account is suspended, verify your password at this link",
        "Earn {amt} per day working from home, no experience needed", "You are selected for a {amt} {item}, reply with your card number",
        "Act fast, only today you can get a {item} for free", "Final warning: pay {amt} immediately or your account will be closed",
        "Hot singles near you, click the link to join free", "Double your money in 24 hours, invest {amt} now",
        "Claim your {item} before midnight, no purchase necessary", "Dear winner, transfer a small fee to receive your {amt} reward"],
    "not spam": [
        "Can we move tomorrow's meeting to 3 pm?", "Please find the attached report for last quarter",
        "Thanks for your help with the project", "Lunch at 1 pm on Friday works for me",
        "The invoice for order {amt} is attached", "Reminder: team standup starts in ten minutes",
        "I reviewed your pull request and left a few comments", "Could you send me the notes from yesterday's call?",
        "Happy birthday! Hope you have a great day", "The delivery of your {item} is scheduled for Monday",
        "Here is the agenda for next week's planning session", "Let me know if {app} access works for you now"],
    "positive": [
        "I love this {item}, it works great", "Excellent service and very friendly staff", "Really happy with the quality of {app}",
        "Fantastic experience, I would recommend it to everyone", "The support team solved my problem quickly, thank you",
        "Best purchase I made this year", "Super smooth and easy to use", "Wonderful product, exceeded my expectations"],
    "negative": [
        "This is terrible, the {item} broke after one day", "Very poor service and nobody replied", "I am disappointed with {app}",
        "Worst experience ever, I want my money back", "The quality is awful and it keeps failing",
        "Completely useless, a waste of time", "I am very unhappy with how I was treated", "Slow, buggy and frustrating to use"],
    "urgent": [
        "Server is down, customers cannot pay, fix immediately", "Need this resolved right now, deadline is in one hour",
        "Critical: {app} is not working for all users", "Please respond ASAP, the client is waiting",
        "Emergency: data loss reported in production", "We are losing orders every minute, escalate now",
        "Immediate action needed, the contract expires today", "Urgent: security breach detected on {app}"],
    "not urgent": [
        "Whenever you have time, please update the logo", "No rush, just sharing some feedback for next month",
        "When convenient, could you review the draft?", "Just a heads up, nothing needed from you for now",
        "Low priority: tidy up the documentation sometime", "Sharing this article for your reading later",
        "It would be nice to refresh the {app} colors next quarter", "FYI only, no reply necessary"],
}
TASKS = [
    ("Is this message spam or not?", {"spam": "spam", "not spam": "not spam"}),
    ("Is this spam?", {"yes": "spam", "no": "not spam"}),
    ("Is this a scam or a legitimate message?", {"scam": "spam", "legitimate": "not spam"}),
    ("What is the sentiment of this text?", {"positive": "positive", "negative": "negative"}),
    ("Is the writer happy or unhappy?", {"happy": "positive", "unhappy": "negative"}),
    ("Is this message urgent?", {"urgent": "urgent", "not urgent": "not urgent"}),
    ("Does this need immediate attention?", {"yes": "urgent", "no": "not urgent"}),
]
ROUTE_Q = ["Which team should handle this email?", "Which team should this request be sent to?",
           "Please classify which team needs to handle this message.", "Route this to the right team."]
PRE = ["", "", "Hi, ", "Hello, ", "FYI: ", "Note: ", "Subject: "]


def split_of(i: int) -> str:
    return "val" if i % 7 == 5 else "test" if i % 7 == 6 else "train"


def fill(rng, t):
    return t.format(amt=rng.choice(AMOUNTS), item=rng.choice(ITEMS), app=rng.choice(APPS))


def record(rng, context, question, options, answer):
    opts = options[:]
    rng.shuffle(opts)
    return {"context": context, "question": question, "options": opts, "target": opts.index(answer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, nargs=3, default=[1500, 300, 300], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(__file__).resolve().parents[1] / "data" / "multitask"
    out.mkdir(parents=True, exist_ok=True)
    pools = {s: {c: [t for i, t in enumerate(ts) if split_of(i) == s] for c, ts in CLASSES.items()}
             for s in ("train", "val", "test")}
    routing, _ = build(a.seed)
    for s, n in zip(("train", "val", "test"), a.per_task):
        rows = []
        for q, mapping in TASKS:
            for _ in range(n):
                option = rng.choice(list(mapping))
                text = fill(rng, rng.choice(pools[s][mapping[option]]))
                rows.append(record(rng, rng.choice(PRE) + text, q, list(mapping), option))
        for r in routing[s][: n * 2]:
            true = NAMES[r["label"]]
            others = rng.sample([x for x in NAMES if x != true], rng.randint(1, 4))
            rows.append(record(rng, r["query"], rng.choice(ROUTE_Q), [true] + others, true))
        rng.shuffle(rows)
        (out / f"{s}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"{s}: {len(rows)} -> {out / (s + '.jsonl')}")
    assert (out / "train.jsonl").stat().st_size > 0


if __name__ == "__main__":
    main()
