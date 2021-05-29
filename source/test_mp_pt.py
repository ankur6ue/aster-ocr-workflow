import torch
import time
from lib.text_detect.file_utils import get_files
from utils import divide_chunks
import concurrent
# import torch.multiprocessing as mp
import itertools
from torch.multiprocessing import Pool
# from multiprocessing import Pool
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import os

def process(filenames, model, rank):
    torch.manual_seed(rank)
    print("processing images: {0} on process {1}".format(len(filenames), os.getpid()))
    for filename in filenames:

        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        # if torch.cuda.is_available():
        #     input_batch = input_batch.to('cuda')
        #     model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # print(probabilities)

if __name__ == '__main__':
    start = time.time()
    torch.set_num_threads(1)
    imgs, _, _ = get_files('/home/ankur/dev/apps/ML/OCR/data/demo_images/imagen')

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)

    model.eval()
    # model.share_memory()

    start = time.time()
    process(imgs[0:150], model, 0)
    end = time.time()
    print("execution time: {0}".format(end - start))

    # split into 5 chunks of 20 each
    img_path_chunks = list(divide_chunks(imgs[0:150], 30))
    processes = []
    # mp.set_start_method('spawn')
    st = time.time()
    with Pool() as p:
        p.starmap(process, zip(img_path_chunks, itertools.repeat(model), itertools.repeat(1)))
    end = time.time()
    print("multiprocessed execution time: {0}".format(end - st))
