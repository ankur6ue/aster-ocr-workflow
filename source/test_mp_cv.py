import time
from lib.text_detect.file_utils import get_files
from utils import divide_chunks
import concurrent
import cv2
import os
import torch
import numpy as np
from torch.multiprocessing import Pool
import itertools

def process(filenames, params):
    print("processing images: {0} on process {1}".format(len(filenames), os.getpid()))
    for filename in filenames:

        src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # percent by which the image is resized
        scale_percent = 200

        # calculate the 50 percent of original dimensions
        width = int(src.shape[1] * scale_percent / 100)
        height = int(src.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        output = cv2.resize(src, dsize)
        for i in range(0,8):
            edges = cv2.Canny(output, 100, 200)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)


if __name__ == '__main__':

    imgs, _, _ = get_files('/home/ankur/dev/apps/ML/OCR/data/demo_images/imagen')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    model.eval()
    model.share_memory()
    img_path_chunks = list(divide_chunks(imgs[0:100], 20))
    start = time.time()
    params = []
    with Pool() as p:
        # args = ((paths, params) for paths in img_path_chunks)
        p.starmap(process, zip(img_path_chunks, itertools.repeat(model)))

    # with concurrent.futures.ProcessPoolExecutor(5) as pool:
    #     args = ((paths) for paths in img_path_chunks)
    #     pool.map(lambda p: process(*p), args)

    end = time.time()
    print("execution time: {0}".format(end - start))
    start = time.time()
    process(imgs[0:100], model)
    end = time.time()
    print("execution time: {0}".format(end - start))