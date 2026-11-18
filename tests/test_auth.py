"""University login: the flow against a fake WSO2, state and PKCE checks, access rules."""

import base64
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from test_assistant import FAKE_ESCO, Agent, Judge

import esco_agent.web.auth as auth
from esco_agent.assistant import build_assistant
from esco_agent.web.app import Sessions, create_app, hash_email

CONFIG = auth.OAuthConfig("client-1", "s3cret", "https://esco.example/auth/callback",
                          allowed_domains=("ua.pt",))


class FakeWSO2:
    def __init__(self, email="teacher@ua.pt"):
        self.email, self.verifier_seen, self.challenge = email, None, None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            form = parse_qs(request.content.decode())
            assert request.headers["authorization"].startswith("Basic ")
            assert form["code"] == ["the-code"]
            assert form["redirect_uri"] == [CONFIG.redirect_uri]
            self.verifier_seen = form["code_verifier"][0]
            return httpx.Response(200, json={"access_token": "at-1"})
        if request.url.path == "/userinfo":
            assert request.headers["authorization"] == "Bearer at-1"
            return httpx.Response(200, json={"email": self.email, "name": "Ana Silva", "sub": "x"})
        return httpx.Response(404)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ESCO_SESSION_SECRET", "test-secret")
    wso2 = FakeWSO2()
    monkeypatch.setattr(auth, "http_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(wso2.handler)))
    graph = build_assistant(FAKE_ESCO, Agent(), Judge(), InMemorySaver())
    sessions = Sessions(graph=graph, model="m", results_dir=str(tmp_path))
    c = TestClient(create_app(sessions, oauth_config=CONFIG, pof_config=False),
                   base_url="https://esco.example")
    return c, wso2, sessions


def login(c, wso2):
    r = c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    url = urlparse(r.headers["location"])
    q = parse_qs(url.query)
    assert url.netloc == "wso2-gw.ua.pt" and q["scope"] == ["openid"]
    assert q["redirect_uri"] == [CONFIG.redirect_uri] and q["code_challenge_method"] == ["S256"]
    wso2.challenge = q["code_challenge"][0]
    return c.get(f"/auth/callback?code=the-code&state={q['state'][0]}", follow_redirects=False)


def test_login_flow_with_state_and_pkce(app):
    c, wso2, sessions = app
    me = c.get("/api/me").json()
    assert (me["login_required"], me["mode"], me["logged_in"], me["error"]) == (True, "ua", False, None)
    assert c.post("/api/session", json={"expected": "programar em C"}).status_code == 401

    r = login(c, wso2)
    assert r.status_code == 302 and r.headers["location"] == "/"
    digest = base64.urlsafe_b64encode(hashlib.sha256(wso2.verifier_seen.encode()).digest())
    assert digest.rstrip(b"=").decode() == wso2.challenge           # PKCE matched
    assert c.get("/api/me").json()["logged_in"] is True
    assert c.get("/api/me").json()["first_name"] == "Ana"

    # the email comes from the login; anything typed is ignored
    r = c.post("/api/session", json={"email": "someone@else.pt", "expected": "programar em C"})
    sid = r.json()["session"]
    s = sessions.sessions[sid]
    assert s["participant"] == hash_email("teacher@ua.pt", sessions.salt) and s["login"] == "ua"
    assert "teacher@ua.pt" not in s["path"].read_text()

    c.get("/auth/logout", follow_redirects=False)
    assert c.get("/api/me").json()["logged_in"] is False
    assert c.post("/api/message", json={"session": sid, "text": "x"}).status_code == 401


def test_wrong_state_is_refused(app):
    c, wso2, _ = app
    c.get("/auth/login", follow_redirects=False)
    r = c.get("/auth/callback?code=the-code&state=forged", follow_redirects=False)
    assert r.status_code == 302
    me = c.get("/api/me").json()
    assert me["logged_in"] is False and "could not be verified" in me["error"]
    assert wso2.verifier_seen is None                               # no code exchange attempted


def test_other_domains_are_refused(app):
    c, wso2, _ = app
    wso2.email = "someone@gmail.com"
    login(c, wso2)
    me = c.get("/api/me").json()
    assert me["logged_in"] is False and "not allowed" in me["error"]


def test_a_session_cannot_be_used_by_another_user(app, tmp_path):
    c, wso2, sessions = app
    login(c, wso2)
    sid = c.post("/api/session", json={"expected": "programar em C"}).json()["session"]
    c.get("/auth/logout", follow_redirects=False)
    wso2.email = "other@ua.pt"
    login(c, wso2)
    r = c.post("/api/message", json={"session": sid, "text": "Programação"})
    assert r.status_code == 403


def test_login_needs_a_session_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("ESCO_SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ESCO_SESSION_SECRET"):
        create_app(Sessions(graph=object(), results_dir=str(tmp_path)), oauth_config=CONFIG,
                   pof_config=False)


def test_config_from_environment(monkeypatch):
    for k in ("UA_OAUTH_CLIENT_ID", "UA_OAUTH_CLIENT_SECRET", "UA_OAUTH_REDIRECT_URI"):
        monkeypatch.delenv(k, raising=False)
    assert auth.OAuthConfig.from_env() is None
    monkeypatch.setenv("UA_OAUTH_CLIENT_ID", "a")
    monkeypatch.setenv("UA_OAUTH_CLIENT_SECRET", "b")
    monkeypatch.setenv("UA_OAUTH_REDIRECT_URI", "https://x/auth/callback")
    monkeypatch.setenv("ESCO_ALLOWED_DOMAINS", "ua.pt, @det.ua.pt")
    cfg = auth.OAuthConfig.from_env()
    assert cfg.token_url == "https://wso2-gw.ua.pt/token" and cfg.pkce
    assert cfg.domain_allowed("x@ua.pt") and cfg.domain_allowed("x@det.ua.pt")
    assert not cfg.domain_allowed("x@gmail.com")
