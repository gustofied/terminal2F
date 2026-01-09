from agent import Agent
from runner import run_agent
from tools import tools

if __name__ == "__main__":
    agent = Agent(tools=tools)

    response = run_agent(agent, "I have 4 apples. How many do you have?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = run_agent(agent, "I ate 1 apple. How many are left?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = run_agent(agent, "What is 157.09 * 493.89?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = run_agent(agent, "What's the status of my transaction T1001?")
    print(response.choices[0].message.content)
