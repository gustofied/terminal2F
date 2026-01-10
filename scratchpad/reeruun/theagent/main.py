from agent import Agent
from runner import run_agent
from tools import tools
import logging
from pathlib import Path
import rerun as rr
from mylogger import setup_logging

pathen = Path(__name__).resolve().parent
teksten_path = pathen / "scratchpad" / "reeruun" / "theagent" / "user_txt.txt"

with open(teksten_path, "r") as file:
    prompt = file.read()

def main():
    rr.init("the_agent_logs", spawn=True)

    config_path = Path(__file__).resolve().parent / "config.json"
    setup_logging(str(config_path))

    log = logging.getLogger("app")

    agent = Agent(tools=tools)


    while True:
        run_agent(agent, prompt)


if __name__ == "__main__":
    main()
