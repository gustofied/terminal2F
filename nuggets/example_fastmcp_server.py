from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hi there {name}! This is an MCP greeting."

if __name__ == "__main__":
    mcp.run()