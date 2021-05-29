import os
import time
import argparse
import sys
import prefect
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor
from utils import read_env_var
from flow_def import create_flow
from state_handlers import send_notification

def main(argv):
    parser = argparse.ArgumentParser(description='Runs text detection and recognition on an input image')
    parser.add_argument('-i', '--images', nargs='+',
                        help='list of input images. Must be jpg, bmp or png')

    args = parser.parse_args()

    # flow.visualize()
    # This is only needed when running directly from Pycharm. When running as a kubernetes pod or docker container,
    # these are passed as environment variables.
    read_env_var([os.path.join(os.getcwd(), '../../credentials/awsenv.list'),
                 os.path.join(os.getcwd(), '../ocr_env_local.list')], os.environ)
    bff_host = 'http://localhost' if os.environ.get('BFF_HOST') is None else 'http://' + os.environ.get('BFF_HOST')
    # If running as a kubernetes job/pod and you want to ping a server running on the master node.
    # here 192.168.1.168 is the Ip of the master node (k get nodes -n default -o wide)
    # bff_host = 'http://192.168.1.168'
    bff_host = bff_host + ":5001"
    # add to prefect context, so the bff host is accessible from tasks
    prefect.context.bff_host = bff_host

    # We want to attach the pod name to status update messages sent when this flow is run as a kubernetes job/pod.
    # This way, the receiving server knows which pod the messages are coming from.
    # When run as a k8s job/pod, the pod name will be written to /etc/podinfo/pod-name (see k8s/ocr-job.yaml).
    # if no such file is found, that means that this script is not run as a k8s job, in which case just use a generic
    # name as the pod name
    job_name = 'ocr-pod'
    if os.path.exists('/etc/podinfo/pod-name'):
        with open('/etc/podinfo/pod-name', 'r') as f:
            job_name = f.readline()

    prefect.context.job_name = job_name

    executor = LocalDaskExecutor(scheduler="threads")
    # executor = LocalExecutor()
    start = time.time()
    ## Important note: before running this from command line, increase number of open files to at least 2056 by using
    # ulimit -n 2056
    flow = create_flow('OCR')
    send_notification("Flow started", False, success=False, logger=prefect.context.get("logger"))
    end_state = flow.run(executor=executor, parameters=dict(input_files=args.images))
    end = time.time()
    failed_tasks = []
    task_states = end_state.result
    success = True
    # If any task didn't finish successfully, set success to false

    for task, task_state in task_states.items():
        if task_state.is_failed() and task in flow.reference_tasks():
            failed_tasks.append(task.name)
            success = False
    send_notification("Flow completed", True, success=success, logger = prefect.context.get("logger"))
    # Also write the status to a file in the pod, which the kubernetes pod can check if the job finished successfully
    # or not
    if os.path.exists('/etc/podstatus'):
        with open('/etc/podstatus/pod-status', 'w') as f:
            f.write("1" if success else "0")
    if success:
        return 0
    return 1


if __name__ == '__main__':
    ret = main(sys.argv)


