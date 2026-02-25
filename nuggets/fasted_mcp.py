from fastmcp import FastMCP
import sys

mcp = FastMCP("Demo")

@mcp.tool
def add(a: int, b:int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    sys.stderr.write("sda")
    mcp.run()

