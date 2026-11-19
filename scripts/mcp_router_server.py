import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server

from system_one.agent_router import ToolRouter

STATE = os.environ.get("ROUTER_STATE", "experiments/router_state.json")
router = ToolRouter(os.environ.get("ROUTER_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"))
if os.path.exists(STATE):
    router.load(STATE)
mcp = Server("system-one-router")


@mcp.tool()
def route_tool(task: str, tools: dict[str, str]) -> dict:
    """Pick the best tool for the task. tools maps tool name to description. Returns tool, confidence, escalate, reason, shadow, id."""
    d = router.route(task, tools)
    router.save(STATE)
    return {k: v for k, v in d.__dict__.items() if k != "scores"}


@mcp.tool()
def report_result(decision_id: int, success: bool) -> dict:
    """Report whether the tool picked for decision_id worked."""
    try:
        router.report(decision_id, success)
    except KeyError:
        return {"ok": False, "error": "unknown decision_id"}
    router.save(STATE)
    return {"ok": True}


@mcp.tool()
def router_status() -> dict:
    """Show threshold, temperature and tools currently flagged as failing."""
    return router.status()


if __name__ == "__main__":
    mcp.run()
