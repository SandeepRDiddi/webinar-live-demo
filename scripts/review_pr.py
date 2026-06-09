#!/usr/bin/env python3
"""
Review changed Python files on a PR and post findings as a GitHub comment.
Exits 1 if critical issues are found — fails the status check, blocks the merge.

Required env vars:
  ANTHROPIC_API_KEY     - your Anthropic key
  GITHUB_TOKEN          - provided automatically by GitHub Actions
  GITHUB_REPOSITORY     - "owner/repo", provided automatically
  PR_NUMBER             - PR number, from github.event.pull_request.number
"""
import os
import sys
import json
import subprocess
import requests
import anthropic


def get_changed_python_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True, text=True, check=True,
    )
    return [
        f for f in result.stdout.strip().split("\n")
        if f.endswith(".py") and os.path.exists(f)
    ]


def review_file(client: anthropic.Anthropic, filepath: str) -> dict:
    code = open(filepath).read()
    r = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="Senior security engineer. Return ONLY valid JSON, no markdown.",
        messages=[{"role": "user", "content": f"""Review this code for security vulnerabilities and bugs.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":null,"title":"","description":"","suggestion":""}}],"has_critical":false}}

FILE: {filepath}
CODE:
{code}"""}],
    )
    text = r.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    return json.loads(text)


def post_pr_comment(body: str) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"body": body},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> None:
    changed = get_changed_python_files()
    if not changed:
        print("No Python files changed. Skipping review.")
        return

    print(f"Reviewing {len(changed)} file(s): {', '.join(changed)}")
    client = anthropic.Anthropic()

    has_critical = False
    sections: list[str] = []
    SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    for filepath in changed:
        try:
            data = review_file(client, filepath)
        except Exception as e:
            print(f"Warning: could not review {filepath}: {e}")
            continue

        issues = data.get("issues", [])
        if not issues:
            sections.append(f"### ✅ `{filepath}`\nNo issues found.")
            continue

        if data.get("has_critical") or any(i.get("severity") == "critical" for i in issues):
            has_critical = True

        lines = [f"### `{filepath}` — {len(issues)} issue(s)"]
        for issue in issues:
            sev = issue.get("severity", "low")
            emoji = SEV_EMOJI.get(sev, "⚪")
            ln = f" (line {issue['line_number']})" if issue.get("line_number") else ""
            lines.append(
                f"\n{emoji} **[{sev.upper()}]** {issue.get('title', '')}{ln}\n"
                f"> {issue.get('description', '')}\n\n"
                f"**Fix:** {issue.get('suggestion', '')}"
            )
        sections.append("\n".join(lines))

    if not sections:
        print("Nothing to report.")
        return

    header = "## 🚨 CRITICAL ISSUES FOUND — Review Required\n\n" if has_critical else "## 🤖 AI Code Review\n\n"
    body = header + "\n\n---\n\n".join(sections)
    body += f"\n\n---\n*Reviewed {len(changed)} file(s) · Powered by Claude Opus 4.5*"

    post_pr_comment(body)

    if has_critical:
        print("CRITICAL issues found. Blocking PR merge.")
        sys.exit(1)

    print(f"Review complete. {sum(len(s) for s in sections)} chars posted.")


if __name__ == "__main__":
    main()
