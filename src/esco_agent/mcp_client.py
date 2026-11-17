"""How the agent starts and connects to the ESCO MCP server."""

from __future__ import annotations

import os
import sys


def server_config() -> dict:
    """Config for MultiServerMCPClient: run the server as a subprocess over stdio.

    The subprocess gets this process's environment (it needs ESCO_DATA_DIR,
    HF_TOKEN, PATH, the Hugging Face cache location...), minus variables whose
    value contains "${": the MCP client treats that as a placeholder to expand
    and warns about each one. The shell prompt PS1 is the usual culprit.
    """
    env = {k: v for k, v in os.environ.items() if "${" not in v}
    return {"esco": {"command": sys.executable, "args": ["-m", "esco_mcp.server"],
                     "transport": "stdio", "env": env}}
