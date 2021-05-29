import requests
import prefect
import datetime as dt
from prefect import Task, Flow
from prefect.engine.state import State
from typing import Union, Dict, Optional


def send_notification(msg, is_completed, success, logger=None):
    try:
        if logger:
            logger.info("BFF HOST: {0}".format(prefect.context.bff_host))
        r = requests.post(prefect.context.bff_host + '/wf_update_status', {'job_name': prefect.context.job_name,
                                                                           'timestamp': dt.datetime.now(),
                                                                           'status_msg': msg,
                                                            'is_completed': is_completed, 'success': success},
                          timeout=0.1)
    # don't reraise, because we want to log the exception and keep going
    except requests.exceptions.ConnectionError as e:
        if logger:
            logger.warn("Connection error raised while sending state change notification to server")
    except requests.exceptions.InvalidSchema as e:
        if logger:
            logger.warn("Invalid schema specified during POST prefect.context.bff_host + '/wf_update_status'")
    except requests.exceptions.InvalidURL as e:
        if logger:
            logger.warn("Invalid URL specified during POST prefect.context.bff_host + '/wf_update_status'")
    except Exception as e:
        # catch all
        if logger:
            logger.warn(e)

def post_updates_state_handler(obj: Union[Flow, Task], old_state: State, new_state: State):
    logger = prefect.context.get("logger")
    # Notification handlers are also called when a task is mapped, separately from the task being called on the
    # mapped inputs. Don't send notifications when mapping happens, only when the mapped task is executed on
    # mapped inputs.
    if new_state.is_finished() and not new_state.is_mapped():
        msg = "Finished task: '{0}' with status {1}"
        msg = msg.format(obj.name, "Success" if new_state.is_successful() else "Failed")
        logger.info(msg)
        send_notification(msg, False, success=new_state.is_successful(), logger=logger)

    return new_state


