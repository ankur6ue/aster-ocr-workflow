import prefect
import sys
from prefect.agent.docker import DockerAgent

agent = DockerAgent(show_flow_logs=True)
agent.start()