import asyncio
from fastmcp import Client
import json

client = Client("https://mcp-docs.ainm.no/mcp")

async def main():
    async with client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        data = {
            "tools": [tool.model_dump() for tool in tools],
            "resources": [resource.model_dump() for resource in resources],
            "prompts": [prompt.model_dump() for prompt in prompts],
        }

        print(json.dumps(data, indent=2, default=str))


        result = await client.call_tool("search_docs", {"query":"WebSocket"})                                                    
        print(json.dumps([c.model_dump() for c in result.content], indent=2, default=str))                                   
                            
asyncio.run(main())
