"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1 — THE VIBE CODED VERSION
 "It works. Ship it."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new: Everything. This is where everyone starts.
 Run: python step1_vibe.py

 Watch the output. It "works." Then ask:
   - Can you filter by severity?
   - Which line is the SQL injection on?
   - Can you alert when critical issues appear?
   - What happened if the API call failed?
   - How much did this cost?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def review_code(code: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Review this code and find issues:\n\n{code}"}
        ]
    )
    return response.content[0].text


if __name__ == "__main__":
    code = open("samples/buggy_code.py").read()

    print("Running vibe coded review...\n")
    print("=" * 60)
    result = review_code(code)
    print(result)
    print("=" * 60)

    print("\n⚠️  Try to answer these questions from the output above:")
    print("   1. How many CRITICAL issues are there?")
    print("   2. Which exact line has the SQL injection?")
    print("   3. What's the overall risk score?")
    print("   4. Can you sort issues by severity?")
    print("\n   You can't. It's a blob of text. That's the problem.")
