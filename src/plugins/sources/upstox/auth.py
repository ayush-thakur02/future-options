"""Upstox OAuth 2.0 authorization-code flow.

Upstox access tokens are valid until 03:30 IST the following day, so the token
is persisted with its generation time and treated as stale after that cutoff.
"""

from __future__ import annotations

import json
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import httpx

from core.calendar import IST

LOGIN_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

TOKEN_EXPIRY_HOUR = 3
TOKEN_EXPIRY_MINUTE = 30


def build_login_url(client_id: str, redirect_uri: str, state: str = "niftypulse") -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
        }
    )
    return f"{LOGIN_URL}?{query}"


def token_expiry(generated_at: datetime) -> datetime:
    """Next 03:30 IST after ``generated_at``."""
    moment = generated_at.astimezone(IST)
    cutoff = moment.replace(
        hour=TOKEN_EXPIRY_HOUR, minute=TOKEN_EXPIRY_MINUTE, second=0, microsecond=0
    )
    if moment >= cutoff:
        cutoff = cutoff + timedelta(days=1)
    return cutoff


class TokenStore:
    """Reads and writes the Upstox access token to disk."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, access_token: str, generated_at: datetime | None = None) -> None:
        generated_at = generated_at or datetime.now(IST)
        payload = {
            "access_token": access_token,
            "generated_at": generated_at.isoformat(),
            "expires_at": token_expiry(generated_at).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def load(self) -> str | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        token = payload.get("access_token")
        expires_at = payload.get("expires_at")
        if not token:
            return None
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
            except ValueError:
                expiry = token_expiry(datetime.fromisoformat(payload["generated_at"]))
            if datetime.now(IST) >= expiry:
                return None
        return token

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def exchange_code_for_token(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    timeout: float = 30.0,
) -> str:
    """Swap a single-use authorization code for an access token."""
    response = httpx.post(
        TOKEN_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"token exchange failed ({response.status_code}): {response.text}")
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"token exchange returned no access_token: {payload}")
    return token


def interactive_login(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_path: Path,
    open_browser: bool = True,
) -> str:
    """Run the browser-based login and persist the resulting token.

    The redirect URI must point somewhere you can read the ``code`` parameter
    from. A localhost callback is easiest; the code is pasted back here.
    """
    url = build_login_url(client_id, redirect_uri)
    print("\nOpen this URL and complete the Upstox login:\n")
    print(f"  {url}\n")
    print(f"Redirect URI is set to: {redirect_uri}")
    print("After redirect, copy the `code` query parameter from the address bar.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    code = input("Paste the authorization code here: ").strip()
    if not code:
        raise RuntimeError("no authorization code provided")

    token = exchange_code_for_token(code, client_id, client_secret, redirect_uri)
    store = TokenStore(token_path)
    store.save(token)
    print(f"\nAccess token stored at {token_path}")
    return token


def resolve_token(
    env_token: str | None,
    token_path: Path,
) -> str | None:
    """Prefer an explicit env token, then fall back to the stored one."""
    if env_token:
        return env_token
    return TokenStore(token_path).load()
