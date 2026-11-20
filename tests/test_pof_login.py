"""Login through POF: the page sends POF's token; the server asks POF who it is."""

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from test_assistant import FAKE_ESCO, Agent, Judge

import esco_agent.web.pof_auth as pof_auth
from esco_agent.assistant import build_assistant
from esco_agent.web.app import Sessions, create_app, hash_email

POF = pof_auth.PofConfig("http://pof.test:8000")
USERS = {"tok-teacher": {"email": "Ana@ua.pt", "name": "Ana Silva", "role": "esco_test"},
         "tok-admin": {"email": "boss@ua.pt", "name": "Boss", "role": "administrator"},
         "tok-other": {"email": "x@ua.pt", "name": "X", "role": "teacher"},
         "tok-pending": {"email": "p@ua.pt", "name": "P", "role": "pending"}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert str(request.url) == "http://pof.test:8000/api/auth/user/"
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in USERS:
            return httpx.Response(401, json={"detail": "Token is invalid or expired"})
        return httpx.Response(200, json={"id": 1, **USERS[token], "username": "u"})

    monkeypatch.setattr(pof_auth, "http_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    pof_auth._cache.clear()
    graph = build_assistant(FAKE_ESCO, Agent(), Judge(), InMemorySaver())
    sessions = Sessions(graph=graph, model="m", results_dir=str(tmp_path))
    return TestClient(create_app(sessions, pof_config=POF)), sessions, calls


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_me_reports_login_state_and_role(client):
    c, _, _ = client
    assert c.get("/api/me").json()["logged_in"] is False
    assert c.get("/api/me").json()["login_url"] == "/pof/login"
    me = c.get("/api/me", headers=auth("tok-teacher")).json()
    assert (me["logged_in"], me["allowed"], me["first_name"]) == (True, True, "Ana")
    me = c.get("/api/me", headers=auth("tok-other")).json()
    assert (me["logged_in"], me["allowed"]) == (True, False) and "não está aberta" in me["error"]


def test_only_experiment_roles_get_in(client):
    c, sessions, _ = client
    body = {"expected": "programar em C"}
    assert c.post("/api/session", json=body).status_code == 401
    assert c.post("/api/session", json=body, headers=auth("expired")).status_code == 401
    assert c.post("/api/session", json=body, headers=auth("tok-other")).status_code == 403
    assert c.post("/api/session", json=body, headers=auth("tok-pending")).status_code == 403
    assert c.post("/api/session", json=body, headers=auth("tok-admin")).status_code == 200

    sid = c.post("/api/session", json=body, headers=auth("tok-teacher")).json()["session"]
    s = sessions.sessions[sid]
    assert s["participant"] == hash_email("ana@ua.pt", sessions.salt)
    assert (s["login"], s["role"]) == ("pof", "esco_test")
    assert "ana@ua.pt" not in s["path"].read_text().lower()
    # another POF user can't use it
    assert c.post("/api/message", json={"session": sid, "text": "Prog"},
                  headers=auth("tok-admin")).status_code == 403
    assert c.post("/api/message", json={"session": sid, "text": "Programação"},
                  headers=auth("tok-teacher")).status_code == 200


def test_pof_is_asked_once_per_token_for_a_while(client):
    c, _, calls = client
    for _ in range(3):
        c.get("/api/me", headers=auth("tok-teacher"))
    assert len(calls) == 1


def test_pof_mode_needs_the_backend_address(monkeypatch):
    monkeypatch.setenv("ESCO_AUTH", "pof")
    monkeypatch.delenv("POF_API_URL", raising=False)
    with pytest.raises(RuntimeError, match="POF_API_URL"):
        pof_auth.PofConfig.from_env()
    monkeypatch.setenv("POF_API_URL", "http://127.0.0.1:9000/")
    monkeypatch.setenv("ESCO_ALLOWED_ROLES", "esco_test")
    cfg = pof_auth.PofConfig.from_env()
    assert cfg.api_url == "http://127.0.0.1:9000" and cfg.allowed_roles == ("esco_test",)
