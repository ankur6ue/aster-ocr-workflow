import os
import time
from dask.distributed import Client, progress, LocalCluster
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor
import json
from itertools import repeat
from multiprocessing import Pool, Process
from text_detect import *
from text_rec import *
from utils import *
from distributed import Client
import dask
import dask.array as da
import copy
import torch
import numpy as np
# global_args = get_args(sys.argv[1:])
device = 'cpu'

distributed = True
use_dask = False
use_mp = True
if __name__ == '__main__':

    # test()
    # parse the config
    # env = sys.argv[1:]
    num_proc = 3
    parent_path = '/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect/'
    img_path = '/home/ankur/dev/apps/ML/OCR/data/demo_images/checks'

    parent_path_param = os.environ.get("PARENT_PATH")
    img_path_param = os.environ.get("IMG_PATH")
    ## This is very important! Without this, multi-processing using pytorch doesn't work!
    ## Also, if you get file descriptor errors, try increasing number of processes by saying ulimit -n 4086
    ## see this article for permanently setting ulimit (although it didn't work for me)
    # https://medium.com/@muhammadtriwibowo/set-permanently-ulimit-n-open-files-in-ubuntu-4d61064429a
    torch.set_num_threads(1)
    if use_dask:
        client = Client(n_workers=3, threads_per_worker=1)
        start = time.time()
        img_file_paths = get_files_impl(img_path_param)
        det_args = dask.delayed(get_det_args_impl)(parent_path_param)
        preprocessed_images = [dask.delayed(preprocess_for_detection_impl)(path, det_args) for path in img_file_paths]
        det_model = dask.delayed(load_detection_model_impl)(det_args)
        cropped_paths_delayed = [dask.delayed(text_detect_impl)(img, img_path, det_model, det_args) for img, img_path in
                         zip(preprocessed_images, img_file_paths)]

        cropped_paths = []
        results = dask.compute(*cropped_paths_delayed)
        cropped_img_path_list = []
        for items in results:
            for item in items:
                cropped_img_path_list.append(item)


        rec_args = get_rec_args_impl(parent_path_param)
        # don't do this!! This will delay loading of the model!
        # rec_model = dask.delayed(init_rec_model_impl)(rec_args)
        rec_model = dask.delayed(init_rec_model_impl(rec_args))

        cropped_img_list = [dask.delayed(preprocess_for_recognition_impl)(path) for path in cropped_img_path_list]

        chunk_len = np.floor(len(cropped_img_path_list) / num_proc).astype(int)
        # chunk the cropped_img_list
        cropped_img_path_chunks = list(divide_chunks(cropped_img_path_list, chunk_len))
        cropped_img_list_chunks = list(divide_chunks(cropped_img_list, chunk_len))

        # cropped_img_list_chunks = da.from_array(cropped_img_list, chunks=(20))
        # cropped_img_path_chunks = da.from_array(cropped_img_path_list, chunks=(20))

        results = [dask.delayed(recognize_impl)(imgs, paths, rec_args, rec_model) for imgs, paths in zip(cropped_img_list_chunks, cropped_img_path_chunks)]

        ## don't do this!
        # for res in results:
        #    res.compute()

        ## do this instead
        dask.compute(*results)
        end = time.time()
        print("Total flow execution time: {0}".format(end - start))

    if use_mp:

        img_file_paths = get_files_impl(img_path_param)
        det_args = get_det_args_impl(parent_path_param)
        preprocessed_images = [preprocess_for_detection_impl(path, det_args) for path in img_file_paths]
        det_model = load_detection_model_impl(det_args)


        cropped_paths = []

        # start = time.time()
        # cropped_paths = [text_detect_impl(img, img_path, det_model, det_args) for img, img_path, det_model, det_args in
        #                 zip(preprocessed_images, img_file_paths, repeat(det_model), repeat(det_args))]
        # end = time.time()
        # print("Text detection time (single-process): {0}".format(end - start))
        start = time.time()
        with Pool(num_proc) as p:
            cropped_paths = p.starmap(text_detect_impl, zip(preprocessed_images, img_file_paths, repeat(det_model), repeat(det_args)))
        end = time.time()
        print("Text detection time (multi-process): {0}".format(end - start))
        rec_args = get_rec_args_impl(parent_path_param)
        rec_model = init_rec_model_impl(rec_args)
        # flatten the list of list of cropped_paths
        cropped_img_path_list = [item for sublist in cropped_paths for item in sublist]
        cropped_img_list = [preprocess_for_recognition_impl(path) for path in cropped_img_path_list]
        chunk_len = np.floor(len(cropped_img_path_list)/num_proc).astype(int)
        cropped_img_path_chunks = list(divide_chunks(cropped_img_path_list, chunk_len))
        cropped_img_list_chunks = list(divide_chunks(cropped_img_list, chunk_len))
        start = time.time()
        # if number of processes is less than the number of chunks, starmap will automatically run recognize_impl
        # sequentially using number of processes available
        with Pool(num_proc) as p:
            rec_results_ = p.starmap(recognize_impl, zip(cropped_img_list_chunks, cropped_img_path_chunks, repeat(rec_args), repeat(rec_model)))
        end = time.time()
        rec_results = [item for sublist in rec_results_ for item in sublist]
        print("Text recognition time (multi-process): {0}".format(end - start))

        # start = time.time()
        # for cropped_img_list, cropped_img_paths, rec_args, rec_model in \
        #     zip(cropped_img_list_chunks, cropped_img_path_chunks, repeat(rec_args), repeat(rec_model)):
        #     recognize_impl(cropped_img_list, cropped_img_paths, rec_args, rec_model)
        # end = time.time()
        print("Text recognition time (single-process): {0}".format(end - start))





