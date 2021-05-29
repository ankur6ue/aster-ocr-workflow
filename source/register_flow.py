import os
import time
from prefect import task, Flow, Task, Parameter, unmapped, flatten, case
import argparse
from prefect.run_configs import LocalRun, DockerRun
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor
from utils import read_env_var
from flow_def import create_flow
from prefect.storage import Docker

from prefect.tasks.control_flow import merge


distributed = True
localRun = False
dockerRun = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Runs text detection and recognition on an input image')
    parser.add_argument('-i', '--images', nargs='+',
                        help='list of input images. Must be jpg, bmp or png')
    parser.add_argument('-c', '--config_path', default="",
                        help='path to config files that store env variables')

    args = parser.parse_args()

    flow = create_flow("OCR")
    env = {}
    # Read the AWS and application environment variables and add them to the flow run_config so they
    # are passed to the flow when it is run by Prefect
    read_env_var([os.path.join(args.config_path, 'awsenv.list'),
                  os.path.join(args.config_path, 'env_local.list')], env)
    # flow.visualize()
    if localRun:

        flow.run_config = LocalRun(env=env, working_dir='/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source')

    if dockerRun:
        # if your package has been already been installed, then you should just be able to import it
        # and this step is not needed
        # flow.storage = Docker(base_image="myocr", local_image=True, image_name="myocr2", env_vars={
        #     # append modules directory to PYTHONPATH
        #     "PYTHONPATH": "$PYTHONPATH:/opt/prefect/app/source/",
        # })
        flow.storage = Docker( base_image="myocr", local_image=True, image_name="prefect-ocr",
                                files={"/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source/utils.py": "/opt/ocr/source/utils.py",
                                       "/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source/state_handlers.py": "/opt/ocr/source/state_handlers.py",
                                       "/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/source/prefect_tasks.py": "/opt/ocr/source/prefect_tasks.py",
                                       "/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/rec_args.json": "/opt/ocr/rec_args.json",
                                       "/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/det_args.json": "/opt/ocr/det_args.json"},
                                env_vars={"PYTHONPATH": "$PYTHONPATH:/opt/ocr/source"},
                                python_dependencies=['pillow'],
                              )
        env['PARENT_PATH'] = '/opt/ocr'
        env['DET_MODEL_PATH'] = '/opt/ocr/models/detection-CRAFT/craft_mlt_25k.pth'
        env['REC_MODEL_PATH'] = '/opt/ocr/models/recognition-ASTER/demo.pth.tar'
        flow.run_config = DockerRun(env=env)

        flow.storage.build()


    # flow.storage = LocalStorage
    executor = LocalDaskExecutor(scheduler="threads")
    flow.executor = executor
    flow.register('ocr')



