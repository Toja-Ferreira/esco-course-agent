"""University of Aveiro login (OpenID Connect, authorization code flow) for the web page.

Same identity provider as POF (the university's WSO2 gateway), but its OWN client
registration: WSO2 checks the redirect URI against the client, so POF's client
can't be reused. Only the identity is taken from the university: after the
callback the page keeps its own signed session cookie, and the WSO2 tokens are
thrown away.

Improvements over the POF implementation it is modelled on: the state is
checked; PKCE (S256) is sent; no tokens or user data in URLs or logs.

Login is on when UA_OAUTH_CLIENT_ID, UA_OAUTH_CLIENT_SECRET and
UA_OAUTH_REDIRECT_URI are set. Otherwise the page asks for an email instead
(for local development). Other settings:
    ESCO_SESSION_SECRET     signs the session cookie (required with login on)
    ESCO_ALLOWED_DOMAINS    e.g. "ua.pt": only emails in these domains get in
    UA_OAUTH_PKCE=0         turn PKCE off if the gateway rejects it
    UA_OAUTH_AUTHORIZE_URL, UA_OAUTH_TOKEN_URL, UA_OAUTH_USERINFO_URL  overrides
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

WSO2 = "https://wso2-gw.ua.pt"


@dataclass
class OAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    authorize_url: str = f"{WSO2}/authorize"
    token_url: str = f"{WSO2}/token"
    userinfo_url: str = f"{WSO2}/userinfo"
    scope: str = "openid"
    pkce: bool = True
    allowed_domains: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "OAuthConfig | None":
        cid, secret = os.environ.get("UA_OAUTH_CLIENT_ID"), os.environ.get("UA_OAUTH_CLIENT_SECRET")
        redirect = os.environ.get("UA_OAUTH_REDIRECT_URI")
        if not (cid and secret and redirect):
            return None
        domains = tuple(d.strip().lower().lstrip("@") for d in
                        os.environ.get("ESCO_ALLOWED_DOMAINS", "").split(",") if d.strip())
        return cls(cid, secret, redirect,
                   os.environ.get("UA_OAUTH_AUTHORIZE_URL", f"{WSO2}/authorize"),
                   os.environ.get("UA_OAUTH_TOKEN_URL", f"{WSO2}/token"),
                   os.environ.get("UA_OAUTH_USERINFO_URL", f"{WSO2}/userinfo"),
                   pkce=os.environ.get("UA_OAUTH_PKCE", "1") != "0",
                   allowed_domains=domains)

    def domain_allowed(self, email: str) -> bool:
        if not self.allowed_domains:
            return True
        domain = email.rsplit("@", 1)[-1].lower()
        return any(domain == d or domain.endswith("." + d) for d in self.allowed_domains)


class LoginError(Exception):
    """Shown to the visitor (no internal details)."""


def start(config: OAuthConfig, session: dict) -> str:
    """Remember a fresh state (and PKCE verifier) in the session; return the
    university's login URL to send the browser to."""
    state = secrets.token_urlsafe(24)
    session["oauth_state"] = state
    params = {"response_type": "code", "client_id": config.client_id,
              "redirect_uri": config.redirect_uri, "scope": config.scope, "state": state}
    if config.pkce:
        verifier = secrets.token_urlsafe(48)
        session["oauth_verifier"] = verifier
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        params.update(code_challenge=challenge, code_challenge_method="S256")
    return f"{config.authorize_url}?{urlencode(params)}"


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=15)


async def finish(config: OAuthConfig, session: dict, code: str | None, state: str | None) -> dict:
    """Check the state, exchange the code, read the user's email. Returns
    {"email", "name"}. Raises LoginError."""
    expected = session.pop("oauth_state", None)
    verifier = session.pop("oauth_verifier", None)
    if not code or not state or not expected or not secrets.compare_digest(state, expected):
        raise LoginError("The login could not be verified; please try again.")
    form = {"grant_type": "authorization_code", "code": code,
            "redirect_uri": config.redirect_uri}
    if verifier:
        form["code_verifier"] = verifier
    try:
        async with http_client() as client:
            token = await client.post(config.token_url, data=form,
                                      auth=(config.client_id, config.client_secret))
            token.raise_for_status()
            access = token.json().get("access_token")
            if not access:
                raise LoginError("The university login did not return a token.")
            info = await client.get(config.userinfo_url,
                                    headers={"Authorization": f"Bearer {access}"})
            info.raise_for_status()
            user = info.json()
    except httpx.HTTPError:
        raise LoginError("The university login service could not be reached; "
                         "please try again.") from None
    email = (user.get("email") or "").strip().lower()
    if "@" not in email:
        raise LoginError("The university login did not provide an email address.")
    if not config.domain_allowed(email):
        raise LoginError("This account is not allowed to use this test.")
    return {"email": email, "name": (user.get("name") or "").strip()}
