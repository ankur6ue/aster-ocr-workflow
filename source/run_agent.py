import prefect
import sys
# This wouldn't work because agent runs in its own process
# sys.path.append('/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source')
# agent = prefect.agent.local.agent.LocalAgent(show_flow_logs=True, import_paths=['/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source'])
agent = prefect.agent.local.agent.LocalAgent(show_flow_logs=True)
agent.start()