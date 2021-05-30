import os
import time
import prefect
# from multiprocessing import shared_memory
from typing import Set, Dict
from prefect import task, Flow, Task, Parameter, unmapped, flatten, case
from prefect.engine import signals
from prefect.engine.state import Mapped, TriggerFailed
import numpy as np
import json
from multiprocessing import Pool
import zipfile
import io
import botocore
import cv2
from PIL import Image
from itertools import repeat
import asterocr as ocr
from utils import get_args_impl, s3_to_local, loadImage, get_s3_client, upload_to_s3, chunk_dict
from state_handlers import post_updates_state_handler

# takes a directory and returns a list of top level files in the directory
@task(name="get_args")
def get_args(name):
    path = os.environ.get("PARENT_PATH")
    logger = prefect.context.get("logger")
    logger.info("Reading files {0} from path {1}".format(name, path))
    args = get_args_impl(path, name)
    if not args:
        raise signals.FAIL("Error loading configuration file {0}".format(name))
    return args


# takes a directory and returns a list of top level files in the directory
@task(name="get_files")
def get_files(input_file_names):
    path = os.environ.get("IMG_PATH")
    logger = prefect.context.get("logger")
    logger.info("Reading input image files from: {0}".format(path))
    local_paths = get_files_impl(path)
    imgs = []
    if len(local_paths) == 0:
        raise signals.FAIL("No input files found")
    for img_name, local_path in zip(input_file_names, local_paths):
        imgs.append({'img_name': img_name, 'img_data': loadImage(local_path)})
    return imgs


# Reads image files specified in input_file_names from S3 and returns the corresponding PIL image objects as
# a dictionary of key-value pairs, key=image_name, value=PIL image object.
@task(name="get_files_s3", state_handlers=[post_updates_state_handler])
def get_files_s3(input_file_names):
    logger = prefect.context.get("logger")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    prefix = 'images'
    dest_base_path = os.environ.get("HOME") + os.environ.get("DEST_BASE_PATH")
    # Create dest_base_path if it doesn't exist
    if not os.path.exists(dest_base_path):
        os.makedirs(dest_base_path)
    logger.info("Reading input image files from S3 bucket: {0}".format(bucket_name))

    try:
        local_paths = s3_to_local(bucket_name, prefix, ['.bmp', '.png', '.jpg'], dest_base_path, logger, input_file_names=input_file_names)
        # open local files and return the PIL Image objects. Can't return paths, because paths are local and not
        # valid if a subsequent task runs on another node
        if len(local_paths) == 0:
            raise signals.FAIL("No input files found")
        imgs = []
        for img_name, local_path in zip(input_file_names, local_paths):
            imgs.append({'img_name': img_name, 'img_data': loadImage(local_path)})
        return imgs
    except botocore.exceptions.ClientError as e:
        logger.warn(e.args[0])
        raise signals.FAIL(e.args[0])
    except Exception as e:
        logger.warn("Error reading image files from S3 bucket: {0}. Msg: {1}".format(bucket_name, e))
        raise signals.FAIL(e)


@task(name="write_to_s3", state_handlers=[post_updates_state_handler])
def write_to_s3(data, key_name):

    # This method copies a dictionary contained in the input "data" argument indexed by "key_name" to S3.
    # data is a dictionary that must contain "img_name" key and key specified by the key_name argument. The value
    # corresponding to img_name key is the name of the image being processed, and the value corresponding to the
    # key_name key is another dictionary that is required to be uploaded to S3 under a s3 key created by
    # concatenating the base image name (without the extension) and the key_name. Eg. if img_name is IMG9134.jpg,
    # and key_name is bboxes, then s3 prefix would be IMG9134/bboxes.
    # The dictionary to be copied is first copied to a local folder and then transferred to S3.
    logger = prefect.context.get("logger")
    # check if the required keys exist
    if data is None or data.get("img_name") is None or data.get(key_name) is None:
        logger.warn("write_to_s3: Incorrect arguments passed")
        raise signals.FAIL()

    bucket_name = os.environ.get("S3_BUCKET_NAME")
    dest_base_path = os.environ.get("HOME") + os.environ.get("DEST_BASE_PATH")
    base, ext = os.path.splitext(data['img_name'])
    local_dir = os.path.join(dest_base_path, 'results', base)
    # create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = os.path.join(local_dir, key_name + '.json')
    s3_prefix = 'results/{0}/{1}.json'.format(base, key_name)
    # write to local disk and then copy to s3
    if data.get(key_name) == {}:
        logger.warn("write_to_s3: No data in {0}".format(key_name))
        return  # nothing to write or transfer

    with open(local_path, 'w') as f:
        json.dump(data[key_name], f)
    upload_to_s3(local_path, bucket_name, s3_prefix)


@task(name="write_to_s3_")
def write_to_s3_(local_paths):
    if local_paths:
        # zip up the files to make upload to s3 faster
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for local_path in local_paths:
                bucket_name = os.environ.get("S3_BUCKET_NAME")
                dest_base_path = os.environ.get("HOME") + os.environ.get("DEST_BASE_PATH")
                s3_prefix = os.path.relpath(local_path, dest_base_path)
                upload_to_s3(local_path, bucket_name, s3_prefix)
                zip_file.write(local_path, compress_type=zipfile.ZIP_DEFLATED)
        # base_path, _ = os.path.split(local_path)
        # file_name, file_ext = os.path.splitext(os.path.basename(local_path))
        # if file_ext == '.jpg':
        #     rec_result_path = os.path.join(base_path, file_name + '.txt')
        #     s3_prefix = os.path.relpath(rec_result_path, dest_base_path)
        #     upload_to_s3(rec_result_path, bucket_name, s3_prefix)



@task(name="init_detection_model")
def init_detection_model(args):
    logger = prefect.context.get("logger")
    path = os.environ.get("DET_MODEL_PATH")
    args["trained_model"] = path
    logger.info('Loading weights from checkpoint (' + args["trained_model"] + ')')
    return ocr.init_detection_model(args)


@task(name="read_image")
def read_image(path, args):
    logger = prefect.context.get("logger")
    logger.info("Reading image {:s} from local filesystem".format(path))
    return loadImage(path)


def signal_on_failure(task, old_state, new_state):
    if new_state.is_failed():
        if getattr(new_state.result, "flag", False) is True:
            print("Special failure mode!  Send all the alerts!")
            print("a == b == {}".format(new_state.result.value))

    return new_state


@task(name="text_detect", state_handlers=[signal_on_failure, post_updates_state_handler])
def text_detect(img, net, args):
    img_name = img['img_name']
    img_data = img['img_data']
    logger = prefect.context.get("logger")
    logger.info("Processing image {:s} for text detection".format(img_name))
    res = ocr.text_detect(img_data, net, args)
    # append image_name and image_data
    res['img_name'] = img_name
    res['img_data'] = img_data
    return [res]


@task(name="get_crops")
def get_crops(det_results):
    # crop out regions corresponding to the bounding boxes
    img_name = det_results['img_name']
    img_data = det_results['img_data']
    bboxes = det_results['bboxes']
    ret = {}
    # return back a dictionary of img_name and word crops
    ret['img_name'] = img_name
    ret['crops'] = {}
    for k, v in bboxes.items():
        bbox_coords = v.split(',')
        x_coords = [int(str_) for str_ in bbox_coords[0:-1:2]]
        y_coords = [int(str_) for str_ in bbox_coords[1::2]]
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        crop = img_data[min_y:max_y, min_x:max_x, :]
        ret['crops'][k] = crop
        # To show crop:
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # plt.imshow(crop)
        # plt.show()

    return [ret]


def _get_all_states_as_set(upstream_states: Dict["core.Edge", "state.State"]) -> set:
    all_states = set()
    for upstream_state in upstream_states.values():
        if isinstance(upstream_state, Mapped):
            all_states.update(upstream_state.map_states)
        else:
            all_states.add(upstream_state)
    return all_states


def custom_trigger_fn(upstream_states: Dict["core.Edge", "state.State"]) -> bool:
    if not all(s.is_successful() for s in _get_all_states_as_set(upstream_states)):
        raise signals.TRIGGERFAIL(
            'Trigger was "all_successful" but some of the upstream tasks failed.'
        )
    return True


@task(name="init_rec_model", trigger=custom_trigger_fn)
def init_rec_model(args):
    path = os.environ.get("REC_MODEL_PATH")
    args["trained_model"] = path
    return ocr.init_rec_model(args)


@task(name="recognize", state_handlers=[post_updates_state_handler])
def recognize(crops, rec_args, rec_net):
    rec_results = ocr.recognize(crops, rec_args, rec_net)
    logger = prefect.context.get("logger")
    for rec_result in rec_results:
        logger.info('Recognition result: {0}'.format(rec_result))
    # return process_impl(input, model, dataset_info)


@task(name="preprocess_for_recognition")
def preprocess_for_recognition(image_path, args):
    return preprocess_for_recognition_impl(image_path)


@task(name="is_data_in_s3")
def is_data_in_s3():
    if "AWS_ACCESS_KEY_ID_SESS" in os.environ:
        return True
    return False


def stream(crops, rec_args, rec_net):
    for k, v in crops.items():
        yield ({k: v}, rec_args, rec_net)


def unpack(args):
    return ocr.recognize(*args)

def prepare(crops, rec_args, rec_net):
    for k, v in crops.items():
        # read image data from local path
        crops[k] = np.array(Image.open(v))

    return ocr.recognize(crops, rec_args, rec_net)



@task(name="recognize_fast", state_handlers=[post_updates_state_handler])
def recognize_fast(crops, rec_args, rec_net, num_proc):
    logger = prefect.context.get("logger")
    logger.info("Performing recognition on image: {0}".format(crops['img_name']))
    chunk_len = np.ceil(len(crops['crops']) / num_proc).astype(int)
    logger.info("chunk size: {0}".format(chunk_len))
    if chunk_len > 0:
        crop_chunks = list(chunk_dict(crops['crops'], chunk_len))
        # Exp: 1 The idea below is to copy the cropped image data to shared memory that can be shared (without copying)
        # with multiple processes. However the code below simply leads to a crash and leaked shared_memory objects
        # for chunk in crop_chunks:
        #     for k, v in chunk.items():
        #         shm = shared_memory.SharedMemory(create=True, size=v.nbytes)
        #         chunk[k] = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)

        # Exp2:  the idea below is to see if the multiprocessed execution is any faster if the image crops are written
        # to the disk
        # and the local paths are passed rather than the image pixel data. The reasoning being that passing data to
        # separate processes could be slower. It turns out that execution isn't any faster (actually a tad slower)
        # doing it this way
        # dest_base_path = os.environ.get("HOME") + os.environ.get("DEST_BASE_PATH")
        # base, ext = os.path.splitext(crops['img_name'])
        # local_dir = os.path.join(dest_base_path, 'results', base)
        # # create local directory if it doesn't exist
        # if not os.path.exists(local_dir):
        #     os.makedirs(local_dir)
        #
        # for chunk in crop_chunks:
        #     for k, v in chunk.items():
        #         local_path = os.path.join(local_dir, k+'.jpg')
        #         # write image data to local path
        #         cv2.imwrite(local_path, v)
        #         chunk[k] = local_path

        # Exp3: The idea is to set up a stream that feeds the cropped images to each process rather than use starmap.
        # again this isn't any faster and actually about 25% slower
        # start = time.time()
        # rec_results = []
        # with Pool(4) as pool:
        #     for ret in pool.imap(unpack, stream(crops['crops'], rec_args, rec_net)):
        #         rec_results.append(ret)
        # logger.info("Text recognition time (multi-process): {0}".format(time.time() - start))start

        # if number of processes is less than the number of chunks, starmap will automatically run recognize_impl
        # sequentially using number of processes available

        # sequential
        # rec_results_ = [recognize_impl(imgs, paths, rec_args, model_artifact)
        #                for imgs, paths in zip(cropped_img_list_chunks, cropped_img_path_chunks)]

        start = time.time()
        with Pool(num_proc) as p:
            logger.info("on process: {0}".format(os.getpid()))
            rec_results_ = p.starmap(ocr.recognize,
                      zip(crop_chunks, repeat(rec_args), repeat(rec_net)))
        end = time.time()

        logger.info("Text recognition time (multi-process): {0}".format(end - start))

        # append the recognition results for each crop to detection results
        for crop_list in rec_results_:
            for k, v in crop_list.items():
                crop_data = crops['crops'][k]
                v.update({'img_data': crop_data})
                crops['crops'].update({k: v})
        logger.info("Text recognition time (multi-process): {0}".format(end - start))
    return [crops]
        # return process_impl(input, model, dataset_info)


@task(name="save_recognize_results")
def save_recognize_results(rec_results):
    """
    This function receives a list of dictionaries, where each dictionary is a (path_to_crop_image_name, rec_result)
    (/path/img1/crop0.jpg, rec_result)
    (/path/img1/crop1.jpg, rec_result)
    (/path/img2/crop0.jpg, rec_result)
    (/path/img2/crop1.jpg, rec_result)
    ...
     pairs.
    crop_image_name is the name of an image crop corresponding to a single word in an input image, and the rec_result
    is the corresponding OCR result. A given input image will in general generate several such cropped images.
    Since the flow accepts as input a list of input images, the list of dictionaries passed to this function will contain
    crops for all images passed as input. The task of this function is to collect the crops and recognition results
    corresponding to a single input image into a list and write the list to a json file in a directory corresponding
    to the image the crops coming from. Eg.,
    /path/img1/ directory should contain a results.json file with contents
    ['crop0.jpg': rec_result}, {'crop1.jpg', rec_result}] etc.
    Then, these result.json files can be copied to the correct key (corresponding to the input image name) in S3. This
    way, all recognition results corresponding to a given image stay together.
    """

    dest_base_path = os.environ.get("HOME") + os.environ.get("DEST_BASE_PATH")

    img_name = rec_results['img_name']
    base, ext = os.path.splitext(img_name)
    local_dir = os.path.join(dest_base_path, 'results', base)
    # create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    rec_results_json = {}
    for k, v in rec_results['crops'].items():
        result_str = v['result']
        stn_input = v['stn_input_with_fid']
        stn_output = v['stn_output']
        rec_results_json[k] = result_str
        stn_output.save(os.path.join(local_dir, k + '_stn_output.jpg'))

    return [{'img_name': img_name, 'results': rec_results_json}]
            # used to store the crops and rec results per input image
    # per_image_results = {}
    # # The input list contains crops and results for all images. Go through the list and separate out the results
    # # for different images
    # image_paths = set()
    # for result in rec_results:
    #     full_file_path = result["filename"]
    #     rec_result = result["rec_result"]
    #     # split up path
    #     # if full_file_path is /path/img1/crop1.jpg,
    #     # path = /path/img1, file_name = crop1.jpg
    #     path, file_name = os.path.split(full_file_path)
    #     image_paths.add(path)  # adding path to set will result in unique paths
    # # got through all unique paths eg. /path/img1, /path/img2 etc., and initialize the per_image_results dict to an
    # # empty list
    # for path in image_paths:
    #     per_image_results[path] = []
    # ## pass 2: Do a second pass over the recognition results and append the {crop_name, rec_result} pairs to
    # # the list corresponding to the image those crops coming form
    # for result in rec_results:
    #     full_file_path = result["filename"]
    #     rec_result = result["rec_result"]
    #     # split up path
    #     path, file_name = os.path.split(full_file_path)
    #     per_image_results[path].append({file_name: rec_result})
    #
    # rec_results_files = []
    # # go through the per_image_results list and write the crops corresponding to each input image to a json file
    # # in the right directory (which has the image name in the path name)
    # for k, v in per_image_results.items():
    #     rec_results_file = os.path.join(k, "results.json")
    #     rec_results_files.append(rec_results_file)
    #     with open(rec_results_file, 'w') as fout:
    #         json.dump(v, fout)
    # # rec_results_file contains the full paths to the results.json corresponding to each input image
    # # /path/img1/results.json, /path/img2/results.json etc. Pass this to downstream processing steps, eg
    # # writing each results.json file to a S3 bucket with the image name as the key.
    # return rec_results_files


