
from prefect import task, Flow, Task, Parameter, unmapped, flatten, case
from prefect_tasks import *
from prefect.tasks.control_flow import merge


def create_flow(name):
    with Flow(name) as flow:
        ## Important note : Thes code below will not be executed within your flow!! You must
        ## get the environment variable in your task function!
        # parent_path = os.environ.get("PARENT_PATH")
        # Read input image files from the directory specified in img_path_param
        input_files = Parameter('input_files', default=None)
        cond = is_data_in_s3()
        det_args = get_args("det_args.json")
        with case(cond, True):
            images1 = get_files_s3(input_files)
        with case(cond, False):
            images2 = get_files(input_files)

        images = merge(images1, images2)
        det_model = init_detection_model(det_args)
        det_results = text_detect.map(images, unmapped(det_model), unmapped(det_args))
        crops = get_crops.map(flatten(det_results))
        with case(cond, True):
            write_to_s3.map(flatten(det_results), unmapped('bboxes'))
            write_to_s3.map(flatten(det_results), unmapped('affinity_score_map'))
            write_to_s3.map(flatten(det_results), unmapped('region_score_map'))
            write_to_s3.map(flatten(det_results), unmapped('connected_component_mask'))

        rec_args = get_args("rec_args.json")
        # rec_args.set_depedencies()
        rec_model = init_rec_model(rec_args)
        # this will call recognize for each cropped image, which is slow.
        # model_output = recognize.map(img, flatten(cropped_paths), unmapped(rec_args), unmapped(rec_model))
        # The fast version will split up the cropped image list into chunks and call recognize on multiple processes
        # recognize_fast will be called separately for each image
        rec_results = recognize_fast.map(flatten(crops), unmapped(rec_args), unmapped(rec_model), unmapped(4))
        rec_results_files = save_recognize_results.map(flatten(rec_results))
        with case(cond, True):
            write_to_s3.map(flatten(rec_results_files), unmapped('results'))
    return flow

