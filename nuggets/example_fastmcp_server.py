from mcp.types import Icon
from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP
import json


mcp = FastMCP("My Drawing Server",
                    instructions="Use this server to let our agent draw",
                    version="0.1",
                    icons=[Icon(src="...")],
                    website_url="https://www.adamsioud.com",
                    )

# The tool does the work
@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
def generate_chart(data: list[float]) -> str:
    return json.dumps({"values": data})

@mcp.resource("ui://my-app/view.html")
def chart_view() -> str:
    return "<html>...</html>"

if __name__ == "__main__":
    mcp.run()