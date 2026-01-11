from agent import Agent
from runners import load
from tools import tools
import logging
from pathlib import Path
from mylogger import setup_logging
import control_tower
import time

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
    agentA = Agent(tools=tools, name="agentA", instance_id="agentA")
    agentB = Agent(tools=tools, name="agentB", instance_id="agentB")

    while True:
        run_agent(agentA, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agentA, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agentA, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agentA, "What is the payment status right now on the latest ID, which is T1001")
        run_agent(agentA, prompt)
        run_agent(agentB, prompt)
        run_agent(agentB, prompt)
        time.sleep(15)
if __name__ == "__main__":
    main()
