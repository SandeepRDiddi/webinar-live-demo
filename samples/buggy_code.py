"""
DEMO FILE: Intentionally Buggy Code
=====================================
This file contains real-world bugs used in the webinar demo.
The Code Review Agent will catch these systematically.
Adding Test

Bugs planted:
  1. SQL Injection (CRITICAL - security)
  2. Hardcoded API key (CRITICAL - security)
  3. Sensitive data logged (HIGH - security)
  4. Missing input validation (HIGH - bug)
  5. Resource leak - DB connection never closed (MEDIUM - bug)
  6. Passwords stored in plaintext (CRITICAL - security)
  7. Bare except clause (MEDIUM - quality)
  8. God function - does too much (LOW - quality)
  9. No rate limiting on login (HIGH - security)
  10. Race condition in counter (MEDIUM - bug)
"""

import sqlite3
import hashlib
import logging
import threading

# ❌ CRITICAL: Hardcoded production API key (demo — agent must detect this)
STRIPE_API_KEY = "sk_live_DEMO_NEVER_HARDCODE_SECRETS_IN_CODE"
DATABASE_PATH = "production.db"

logger = logging.getLogger(__name__)


# ❌ CRITICAL: SQL Injection vulnerability
def get_user_by_email(email: str):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    # Attacker input: ' OR '1'='1 --
    query = f"SELECT * FROM users WHERE email = '{email}'"
    cursor.execute(query)
    result = cursor.fetchone()
    # ❌ MEDIUM: Connection never closed (resource leak)
    return result


# ❌ CRITICAL: Passwords stored in plaintext
def create_user(username: str, password: str, email: str):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # ❌ HIGH: Sensitive data logged to console
    logger.info(f"Creating user: {username}, password: {password}, email: {email}")

    # Plaintext password inserted directly
    cursor.execute(
        "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
        (username, password, email)
    )
    conn.commit()
    conn.close()
    return True


# ❌ HIGH: No input validation, no rate limiting
def login(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if user and user[2] == password:  # password check on index, fragile
        token = hashlib.md5(email.encode()).hexdigest()  # ❌ MD5 is cryptographically broken
        return {"token": token, "user_id": user[0]}
    return {"error": "Invalid credentials"}


# ❌ CRITICAL: Processes payment with no validation, logs card number
def process_payment(user_id: int, amount: float, card_number: str, cvv: str):
    # ❌ CRITICAL: PCI violation - logging card data
    logger.info(f"Processing payment: user={user_id}, card={card_number}, cvv={cvv}, amount={amount}")

    # ❌ HIGH: No amount validation - negative amount = free money
    # ❌ HIGH: No card number format validation

    import urllib.request
    import json

    payload = json.dumps({
        "api_key": STRIPE_API_KEY,
        "amount": amount,
        "card": card_number,
        "cvv": cvv
    }).encode()

    req = urllib.request.Request(
        "https://api.stripe.com/v1/charges",
        data=payload,
        method="POST"
    )

    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except:  # ❌ MEDIUM: Bare except swallows all errors silently
        return {"status": "failed"}


# ❌ MEDIUM: Race condition - not thread safe
request_counter = 0

def track_request():
    global request_counter
    # In multi-threaded env, two threads can read same value simultaneously
    request_counter = request_counter + 1
    return request_counter


# ❌ LOW: God function - violates single responsibility
def handle_everything(action: str, data: dict):
    """Does user creation, login, payment, AND analytics. All in one."""
    if action == "register":
        return create_user(data["username"], data["password"], data["email"])
    elif action == "login":
        return login(data["email"], data["password"])
    elif action == "pay":
        return process_payment(
            data["user_id"], data["amount"],
            data["card"], data["cvv"]
        )
    elif action == "stats":
        return {"requests": request_counter}
