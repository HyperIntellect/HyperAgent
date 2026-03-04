"""Security tests for HTTP client tool SSRF defenses."""

import json
from unittest.mock import patch

import pytest

from app.agents.tools.http_client import http_request


@pytest.mark.asyncio
async def test_http_request_blocks_private_dns_resolution():
    with patch(
        "app.agents.tools.http_client._resolve_host_ips",
        return_value=(False, ["127.0.0.1"], "resolved_private_ip"),
    ):
        result = await http_request.ainvoke({"url": "https://example.com"})
    data = json.loads(result)
    assert data["success"] is False
    assert data["blocked_reason_code"] == "resolved_private_ip"

