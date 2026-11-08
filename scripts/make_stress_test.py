import json
from collections import Counter
from pathlib import Path

TASKS = {
    "S": ("Is this message spam or not?", ["spam", "not spam"]),
    "M": ("What is the sentiment of this text?", ["positive", "negative"]),
    "U": ("Is this message urgent?", ["urgent", "not urgent"]),
    "T": ("Which team should handle this email?",
          ["technical team", "finance team", "account team", "advertisement team", "general support"]),
}
E, M, H, B = "easy", "medium", "hard", "brutal"
ITEMS = [
    (E, "S", "Win a free iPhone now, click this link", 0), (E, "S", "Your lunch order is ready for pickup", 1),
    (E, "S", "Claim your $5000 cash prize today", 0), (E, "S", "Meeting notes from this morning are attached", 1),
    (E, "S", "You have been selected for a free vacation, send your card number", 0),
    (E, "S", "Please review the budget document by Friday", 1),
    (E, "S", "Earn $500 a day from home, no experience needed", 0), (E, "S", "Happy anniversary to the whole team!", 1),
    (E, "M", "I absolutely love this product", 0), (E, "M", "This is the worst service I have ever had", 1),
    (E, "M", "Great support, problem solved in minutes", 0), (E, "M", "Terrible quality, it broke immediately", 1),
    (E, "M", "Very happy with my purchase", 0), (E, "M", "I am extremely disappointed and angry", 1),
    (E, "U", "Server is down, fix immediately", 0), (E, "U", "No rush, whenever you get time", 1),
    (E, "U", "Critical outage affecting all customers", 0), (E, "U", "Just sharing an article for later reading", 1),
    (E, "U", "Emergency: production database is corrupted", 0), (E, "U", "Low priority, tidy the docs next month", 1),
    (E, "T", "The app crashes every time I open it", 0), (E, "T", "I was billed twice this month", 1),
    (E, "T", "I forgot my password", 2), (E, "T", "I want to advertise my product on your platform", 3),
    (E, "T", "What are your office hours?", 4), (E, "T", "Error 500 when loading the dashboard", 0),
    (E, "T", "Please send me the invoice for my last payment", 1), (E, "T", "Please delete my account", 2),
    (E, "T", "How much does a banner ad cost?", 3), (E, "T", "Where can I find the user guide?", 4),
    (M, "S", "Final notice: your parcel is held, pay a small fee at this link to release it", 0),
    (M, "S", "Hi, attaching the signed contract as discussed on the call", 1),
    (M, "S", "Exclusive deal just for you, buy now and get 90% off, limited stock", 0),
    (M, "S", "Reminder that the quarterly review is scheduled for Thursday at 10", 1),
    (M, "S", "Your bank account has been locked, confirm your identity here immediately", 0),
    (M, "S", "Could you share the slides before the client meeting tomorrow?", 1),
    (M, "S", "Hey! Want to earn extra income? Message me for a secret method", 0),
    (M, "S", "The package you ordered has shipped and will arrive on Tuesday", 1),
    (M, "M", "It does the job, nothing special but I am pleased with the speed", 0),
    (M, "M", "Not what I expected, and customer care never helped me", 1),
    (M, "M", "The update fixed everything I complained about, impressive", 0),
    (M, "M", "After weeks of waiting, the product arrived damaged", 1),
    (M, "M", "Friendly staff and the food was delicious", 0), (M, "M", "The app keeps freezing and I am losing patience", 1),
    (M, "U", "Our biggest client is threatening to cancel unless we respond today", 0),
    (M, "U", "If you can look at this sometime next week that would be great", 1),
    (M, "U", "Payments are failing for everyone since the last deployment", 0),
    (M, "U", "Not urgent, but when convenient could you rename the folder?", 1),
    (M, "U", "The deadline for the tender submission is in two hours", 0),
    (M, "U", "Feel free to reply when you are back from vacation", 1),
    (M, "T", "Can I get a refund for the subscription I cancelled last week?", 1),
    (M, "T", "My account email needs to be changed to a new address", 2),
    (M, "T", "We would like to sponsor your newsletter for the next quarter", 3),
    (M, "T", "The upload button does nothing when I click it", 0),
    (M, "T", "Do you have any openings in your company?", 4), (M, "T", "Why was I charged more than the quoted price?", 1),
    (M, "T", "I need help merging my two profiles", 2), (M, "T", "Can you tell me which countries you operate in?", 4),
    (M, "T", "Our ad campaign stopped showing impressions since Monday", 3),
    (M, "T", "The video call keeps disconnecting after a few minutes", 0),
    (H, "S", "You have won! Just kidding, here are the minutes from today's meeting", 1),
    (H, "S", "Congratulations on your promotion! The whole team is proud of you", 1),
    (H, "S", "Security alert: unusual sign-in detected. Verify at secure-login-bank.xyz", 0),
    (H, "S", "Dear customer, kindly update your KYC to avoid suspension, click here", 0),
    (H, "S", "I won the quiz at work today and the prize was a gift card, thanks for all your help", 1),
    (H, "M", "I wanted to love it, but it fell short in every way", 1), (H, "M", "Not bad at all, honestly better than I expected", 0),
    (H, "M", "Not good, not terrible, but I would not buy it again", 1), (H, "M", "I cannot say I am unhappy with the results", 0),
    (H, "M", "The price is high but worth every penny", 0),
    (H, "U", "No need to panic, but the payment gateway has been down for six hours", 0),
    (H, "U", "This is the third reminder, though it is not time sensitive", 1),
    (H, "U", "Whenever you can, but the client call is in 30 minutes", 0),
    (H, "U", "Urgent request: please add a smiley to the footer sometime this month", 1),
    (H, "U", "ASAP is not needed, but the audit starts tomorrow and we still lack the files", 0),
    (H, "T", "My payment page is not loading", 0), (H, "T", "I was charged for ads that never ran, please refund me", 1),
    (H, "T", "I cannot log in because my account says suspended", 2), (H, "T", "The verification code never arrives on my phone", 2),
    (H, "T", "The login page is not opening in WhatsApp", 0), (H, "T", "Our sponsored post was rejected, can you explain why?", 3),
    (H, "T", "How do I change the billing cycle of my plan?", 1), (H, "T", "I would like to update the phone number linked to my profile", 2),
    (H, "T", "Do you offer discounts for bulk advertising packages?", 3), (H, "T", "The checkout page throws an error when I enter my card", 0),
    (B, "M", "Oh great, another update that deleted my settings. Just wonderful.", 1),
    (B, "M", "Thanks a lot for nothing, really helpful as always.", 1), (B, "M", "lol this is fire, cant stop using it", 0),
    (B, "S", "frree giftcard!!! clik herre 2 clam ur prize", 0),
    (B, "S", "Hi John, as promised here is the invoice. Also click http://bit.ly/x9 to win a free cruise", 0),
    (B, "S", "Our boss sent everyone a thank-you gift card for finishing the project", 1),
    (B, "U", "Please ignore my previous email asking for urgent help, everything is fine now", 1),
    (B, "U", "Not a big deal but our whole database might be gone", 0), (B, "U", "The server is on fire, haha just kidding, we are fine", 1),
    (B, "T", "help", 4), (B, "T", "Refund me now, your app deleted my account and then charged my card", 1),
    (B, "T", "I think someone hacked my account and used my card to buy ads", 2), (B, "T", "payment failed error 403 on login screen", 0),
    (B, "T", "We love your product and want to partner with you to market it together", 3),
    (B, "T", "Dear Sir/Madam, I hope this finds you well. I have a general inquiry regarding your services.", 4),
]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "stress_test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for diff, task, text, target in ITEMS:
            q, opts = TASKS[task]
            f.write(json.dumps({"context": text, "question": q, "options": opts, "target": target,
                                "difficulty": diff}) + "\n")
    print(f"{len(ITEMS)} questions -> {out}", dict(Counter(i[0] for i in ITEMS)))


if __name__ == "__main__":
    main()