"""
Creates, updates, and deletes a job object.
"""

from os import path
import yaml
import kubernetes
from kubernetes import client, config
import time
import uuid
import threading
from prettytable import PrettyTable

JOB_NAME = "ocr-job"
NAMESPACE = "dev"
IMAGE_NAME = "ocr_prefect"

def create_job_object():
    # Configureate Pod template container
    container = client.V1Container(
        name="ocr",
        image=IMAGE_NAME,
        command=["/bin/sh", "-c", "python run_flow.py -i IMG-9134.jpg"])
    # Create and configurate a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "ocr"}),
        spec=client.V1PodSpec(restart_policy="Never", containers=[container]))
    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=1)
    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME),
        spec=spec)

    return job


def create_job(api_instance, img_str, resources=None):
    with open(path.join(path.dirname(__file__), "../k8s/ocr-job.yaml")) as f:
        dep = yaml.safe_load(f)
        uuid6 = str(uuid.uuid4())[:6]
        dep['metadata']['name'] = "ocr-job-{0}".format(uuid6)
        dep['spec']['template']['spec']['containers'][0]['command'][2] = "python run_flow.py -i {0}".format(img_str)
        if resources is not None: # if the user specified custom CPU/Memory requests/limits, use those
            dep['spec']['template']['spec']['containers'][0]['resources']['requests']['memory'] = resources['requests']['memory']
            dep['spec']['template']['spec']['containers'][0]['resources']['limits']['memory'] = resources['limits']['memory']
            dep['spec']['template']['spec']['containers'][0]['resources']['requests']['cpu'] = resources['requests']['cpu']
            dep['spec']['template']['spec']['containers'][0]['resources']['limits']['cpu'] = resources['limits']['cpu']
        api_response = api_instance.create_namespaced_job(
            body=dep,
            namespace=NAMESPACE)
        print("Job created. status='%s'" % str(api_response.status))
        return api_response.metadata.name


def print_job_status(batch_v1, core_v1, jobs, start_time):
    pt = PrettyTable()
    # first print resource quota (used/total)
    api_response = core_v1.list_namespaced_resource_quota(NAMESPACE)
    total = api_response.items[0].status.hard
    used = api_response.items[0].status.used
    pt.field_names = ["", "limits.cpu", "limits.memory", "requests.cpu", "requests.memory"]
    pt.add_row(["used/total", str(used['limits.cpu']) + "/" + str(total['limits.cpu']),
                str(used['limits.memory']) + "/" + str(total['limits.memory']),
                str(used['requests.cpu']) + "/" + str(total['requests.cpu']),
                str(used['requests.memory']) + "/" + str(total['requests.memory'])])
    print(pt)
    pt = PrettyTable()
    pt.field_names = ["job name", "start time", "completion time", "execution time", "total execution time",
                                                                                     "succeeded"]
    all_succeeded = False
    for job in jobs:
        api_response = batch_v1.read_namespaced_job(job, NAMESPACE)
        st = api_response.status.start_time
        ct = api_response.status.completion_time
        # if either completion time or start time is None, et (execution time) is None.
        et = None if not ct or not st else ct - st
        s = api_response.status.succeeded
        t = time.time() - start_time
        all_succeeded = all_succeeded & False if s != 1 else True
        pt.add_row([job, st, ct, et, t, s])
    print(pt)
    # stop printing once all jobs finish
    if not all_succeeded:
        threading.Timer(2.0, lambda: print_job_status(batch_v1, core_v1, jobs, start_time)).start()


def update_job(api_instance, job):
    # Update container image
    job.spec.template.spec.containers[0].image = "perl"
    api_response = api_instance.patch_namespaced_job(
        name=JOB_NAME,
        namespace="default",
        body=job)
    print("Job updated. status='%s'" % str(api_response.status))


def delete_job(api_instance):
    try:
        api_response = api_instance.delete_namespaced_job(
            name=JOB_NAME,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=0))
        print("Job deleted. status='%s'" % str(api_response.status))
    except kubernetes.client.exceptions.ApiException as e:
        print('error deleting job, reason: {0}'.format(e.reason))


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    img_str = "trading-issue.jpg IMG-9134.jpg"
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    # Create a job object with client-python API. The job wspo
    # created is same as the `pi-job.yaml` in the /examples folder.
    # delete_job(batch_v1)
    time.sleep(1)
    num_jobs = 2
    jobs = []
    for job in range(0, num_jobs):
        # createjobs returns the job name, which we put on a list. Later, we ask for status updates for each job
        jobs.append(create_job(batch_v1, img_str))

    print_job_status(batch_v1, core_v1, jobs, time.time())


if __name__ == '__main__':
    main()