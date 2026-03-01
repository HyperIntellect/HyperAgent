"""API endpoints for MCP server management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.agents.policy import PolicyInput, get_policy_engine
from app.mcp.client import MCPServerConfig
from app.mcp.registry import connect_server, disconnect_server, get_mcp_client

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp")


class AddServerRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    auth_token: str | None = None
    description: str = ""
    enabled: bool = True


class ServerResponse(BaseModel):
    name: str
    transport: str
    description: str
    enabled: bool
    connected: bool
    tool_count: int


@router.get("/servers")
async def list_servers():
    """List all MCP server connections."""
    client = get_mcp_client()
    configs = client.get_server_configs()
    connected = set(client.get_connected_servers())
    tool_infos = client.get_tool_infos()

    servers = []
    for config in configs:
        tool_count = sum(1 for t in tool_infos if t.server_name == config.name)
        servers.append(
            ServerResponse(
                name=config.name,
                transport=config.transport,
                description=config.description,
                enabled=config.enabled,
                connected=config.name in connected,
                tool_count=tool_count,
            )
        )

    return {"servers": [s.model_dump() for s in servers]}


@router.post("/servers")
async def add_server(request: AddServerRequest):
    """Add and connect to an MCP server."""
    config = MCPServerConfig(
        name=request.name,
        transport=request.transport,
        command=request.command,
        args=request.args,
        env=request.env,
        url=request.url,
        auth_token=request.auth_token,
        description=request.description,
        enabled=request.enabled,
    )

    success = await connect_server(config)

    client = get_mcp_client()
    return {
        "success": success,
        "name": config.name,
        "tool_count": len([t for t in client.get_tool_infos() if t.server_name == config.name]),
    }


@router.delete("/servers/{name}")
async def remove_server(name: str):
    """Remove an MCP server connection."""
    success = await disconnect_server(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    return {"success": True}


@router.post("/servers/{name}/connect")
async def reconnect_server(name: str):
    """Reconnect to an existing MCP server."""
    client = get_mcp_client()
    configs = {c.name: c for c in client.get_server_configs()}

    if name not in configs:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    config = configs[name]

    # Remove existing connection first
    await disconnect_server(name)

    # Reconnect
    success = await connect_server(config)

    return {
        "success": success,
        "name": name,
        "tool_count": len([t for t in client.get_tool_infos() if t.server_name == name]),
    }


@router.post("/servers/{name}/disconnect")
async def disconnect_server_endpoint(name: str):
    """Disconnect from an MCP server without removing its config."""
    client = get_mcp_client()

    if name not in [c.name for c in client.get_server_configs()]:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    success = await disconnect_server(name)
    return {"success": success}


@router.get("/tools")
async def list_mcp_tools():
    """List all tools available from MCP servers."""
    client = get_mcp_client()
    tool_infos = client.get_tool_infos()

    engine = get_policy_engine()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "server": t.server_name,
                "input_schema": t.input_schema,
                "policy": {
                    "risk_level": engine.assess_risk(t.name).value,
                    "default_decision": engine.decide(
                        PolicyInput(
                            tool_name=t.name,
                            tool_args={},
                            hitl_enabled=True,
                            risk_threshold="high",
                        )
                    ).decision.value,
                },
            }
            for t in tool_infos
        ],
        "count": len(tool_infos),
    }


@router.get("/presets")
async def list_presets():
    """List available MCP server presets."""
    from app.mcp.presets import list_presets as get_presets

    return {"presets": get_presets()}


@router.post("/presets/{name}/enable")
async def enable_preset(name: str):
    """Enable a built-in MCP server preset."""
    from app.mcp.presets import get_preset

    preset = get_preset(name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")

    success = await connect_server(preset)
    client = get_mcp_client()

    return {
        "success": success,
        "name": preset.name,
        "description": preset.description,
        "tool_count": len(
            [t for t in client.get_tool_infos() if t.server_name == preset.name]
        ),
    }
