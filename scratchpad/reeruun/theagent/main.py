from agent import Agent
from runner import run_agent
from tools import tools
import logging
from pathlib import Path
import rerun as rr
from mylogger import setup_logging

def main():

    rr.init("the_agent_logs", spawn=True)

    config_path = Path(__file__).resolve().parent / "config.json"
    setup_logging(str(config_path))

    log = logging.getLogger("app")

    agent = Agent(tools=tools)

    r = run_agent(agent, "I have 4 apples. How many do you have?")
    
    print(r.model_dump_json)

    # while True:
    #     # response = run_agent(agent, "I have 4 apples. How many do you have?")
    #     print(agent.model)
    #     # log.info(response.choices[0].message.content)
    #     print("- - - - - - - - - - - -")

    # response = run_agent(agent, "I ate 1 apple. How many are left?")
    # log.info(response.choices[0].message.content)
    # print("- - - - - - - - - - - -")

    # response = run_agent(agent, "What is 157.09 * 493.89?")
    # log.info(response.choices[0].message.content)
    # print("- - - - - - - - - - - -")

    # response = run_agent(agent, "What's the status of my transaction T1001?")
    # log.info(response.choices[0].message.content)

    # log.info("info example")
    # log.warning("warning example")

if __name__ == "__main__":
    main()

