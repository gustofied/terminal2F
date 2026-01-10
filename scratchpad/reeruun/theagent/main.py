from agent import Agent
from runners import load
from tools import tools
import logging
from pathlib import Path
from mylogger import setup_logging
import control_tower

BASE_DIR = Path(__file__).resolve().parent
teksten_path = BASE_DIR / "user_txt.txt"

# if I want to test heavier prompt..
with open(teksten_path, "r", encoding="utf-8") as file:
    prompt = file.read()

run_agent = load("regular")

def main():
    control_tower.init()

    config_path = BASE_DIR / "config.json"
    setup_logging(str(config_path))

    log = logging.getLogger("app")
    agent = Agent(tools=tools)
    agent2 = Agent(tools=tools)

    while True:
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agent, prompt)
        run_agent(agent2, prompt)
        run_agent(agent2, prompt)
if __name__ == "__main__":
    main()
