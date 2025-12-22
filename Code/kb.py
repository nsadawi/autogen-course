# kb.py
from __future__ import annotations

KB = [
    {
        "id": "login_2fa",
        "title": "Login issues (2FA)",
        "content": "If 2FA codes aren’t arriving: (1) confirm your phone has signal, "
                   "(2) check blocked numbers, (3) wait 2 minutes and retry, "
                   "(4) try voice call option, (5) if still failing, create a ticket."
    },
    {
        "id": "refund_policy",
        "title": "Refund policy",
        "content": "Refunds are eligible within 14 days for unused services. "
                   "For account-specific eligibility, create a ticket."
    },
    {
        "id": "esim_setup",
        "title": "eSIM setup",
        "content": "To set up eSIM: Settings → Mobile/Cellular → Add eSIM → Scan QR code. "
                   "If activation fails, reboot and retry once."
    },
]

def kb_search(query: str) -> str:
    """Very small keyword search over a toy KB."""
    q = query.lower()
    hits = []
    for item in KB:
        text = (item["title"] + " " + item["content"]).lower()
        if any(tok in text for tok in q.split()):
            hits.append(f"- [{item['id']}] {item['title']}: {item['content']}")
    if not hits:
        return "No KB hits found. Consider asking a clarifying question or creating a ticket."
    return "KB results:\n" + "\n".join(hits)

