"""Login through POF: the page runs on POF's own address (under /esco/), so the
browser already holds POF's login token (localStorage "access"). The page sends
it with every request; this asks POF's backend who it belongs to.

    GET {POF_API_URL}/api/auth/user/   Authorization: Bearer <access>
    -> {"id", "name", "email", "role", ...}

Only users whose POF role is in ESCO_ALLOWED_ROLES get in (default
"esco_test,administrator"). Answers are cached briefly per token so POF isn't
asked on every click. Settings:
    ESCO_AUTH=pof
    POF_API_URL          POF's backend as seen from this app, e.g.
                         e.g. http://127.0.0.1:<port> (same host) or http://<backend-container>:<port> (same Docker network)
    ESCO_ALLOWED_ROLES   comma-separated POF roles
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import httpx

CACHE_SECONDS = 60


@dataclass
class PofConfig:
    api_url: str
    allowed_roles: tuple[str, ...] = ("esco_test", "administrator")
    login_url: str = "/pof/login"
    logout_url: str = "/pof/logout"

    @classmethod
    def from_env(cls) -> "PofConfig | None":
        if os.environ.get("ESCO_AUTH", "").lower() != "pof":
            return None
        api = os.environ.get("POF_API_URL", "").rstrip("/")
        if not api:
            raise RuntimeError("ESCO_AUTH=pof needs POF_API_URL (POF's backend address)")
        roles = tuple(r.strip() for r in os.environ.get(
            "ESCO_ALLOWED_ROLES", "esco_test,administrator").split(",") if r.strip())
        return cls(api, roles)


class NotLoggedIn(Exception):
    """No token, or POF says it is invalid or expired (the page then refreshes it)."""


class NotAllowed(Exception):
    """A valid POF user whose role is not part of the experiment."""


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=10)


_cache: dict[str, tuple[float, dict]] = {}


async def current_user(config: PofConfig, authorization: str | None) -> dict:
    """{"email", "first_name", "role"} for the POF token in the Authorization header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise NotLoggedIn()
    token = authorization.split(" ", 1)[1].strip()
    key = hashlib.sha256(token.encode()).hexdigest()
    hit = _cache.get(key)
    if hit and hit[0] > time.monotonic():
        user = hit[1]
    else:
        try:
            async with http_client() as client:
                r = await client.get(f"{config.api_url}/api/auth/user/",
                                     headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            raise NotLoggedIn() from None
        if r.status_code in (401, 403):
            raise NotLoggedIn()
        r.raise_for_status()
        data = r.json()
        email = (data.get("email") or "").strip().lower()
        if "@" not in email:
            raise NotLoggedIn()
        name = (data.get("name") or "").strip()
        user = {"email": email, "first_name": name.split(" ")[0] if name else "",
                "role": data.get("role") or ""}
        _cache[key] = (time.monotonic() + CACHE_SECONDS, user)
        if len(_cache) > 1000:          # keep it small
            now = time.monotonic()
            for k in [k for k, (exp, _) in _cache.items() if exp < now]:
                _cache.pop(k, None)
    if user["role"] not in config.allowed_roles:
        raise NotAllowed()
    return user
