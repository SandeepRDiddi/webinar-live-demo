"""
Payment service for the StackRoute platform.
Handles user authentication and payment processing.
"""

import sqlite3
import hashlib
import logging
import requests

logger = logging.getLogger(__name__)

# Production database
DB_PATH = "prod.db"

# Payment gateway credentials
PAYMENT_API_KEY = "sk_live_DEMO_NEVER_HARDCODE_SECRETS_IN_CODE_Production"
PAYMENT_ENDPOINT = "https://payments.example.com/charge"


def get_user(email: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE email = '{email}'"
    cursor.execute(query)
    return cursor.fetchone()


def login(email: str, password: str) -> dict:
    user = get_user(email)
    if not user:
        return {"error": "User not found"}

    stored_password = user[2]
    if stored_password != password:
        return {"error": "Wrong password"}

    token = hashlib.md5(email.encode()).hexdigest()
    logger.info(f"User logged in: {email}, password={password}, token={token}")
    return {"token": token, "user_id": user[0]}


def charge_card(user_id: int, amount: float, card_number: str, cvv: str) -> dict:
    logger.info(
        f"Charging card for user {user_id}: "
        f"card={card_number}, cvv={cvv}, amount={amount}"
    )

    response = requests.post(
        PAYMENT_ENDPOINT,
        json={
            "api_key": PAYMENT_API_KEY,
            "amount": amount,
            "card": card_number,
            "cvv": cvv,
        },
        timeout=30,
    )

    return response.json()
