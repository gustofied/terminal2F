from agent import Agent
from runners import run_agent
from tools import tools
import logging
from pathlib import Path
from mylogger import setup_logging
import control_tower

BASE_DIR = Path(__file__).resolve().parent
teksten_path = BASE_DIR / "user_txt.txt"

# if I want to test ctx explosion..
with open(teksten_path, "r", encoding="utf-8") as file:
    prompt = file.read()


def main():
    control_tower.init()

    config_path = BASE_DIR / "config.json"
    setup_logging(str(config_path))

    log = logging.getLogger("app")
    agent = Agent(tools=tools)

    while True:
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, prompt)
if __name__ == "__main__":
    main()
