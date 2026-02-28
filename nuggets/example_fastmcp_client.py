import asyncio
from fastmcp import Client
import json

client = Client("example_fastmcp_server.py")

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
                       
                            
asyncio.run(main())
