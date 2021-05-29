from PIL import ImageDraw, Image, ImageFont
import numpy as np
import torchvision
import cv2
from collections import OrderedDict
import json
from skimage import io
import os
import boto3
import logging
import itertools


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_args_impl(path, file_name):
    """
    Loads the json file containing the detection/recognition params from parent_path and returns it
    """
    args = {}
    full_path = os.path.join(path, file_name)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            args = json.load(f)
    return args


def get_files_impl(path):
    imgs = []
    if (os.path.isdir(path)):
        imgs, masks, xmls = _get_files(path)
    return imgs


def chunk_dict(d, n=10000):
    it = iter(d)
    for i in range(0, len(d), n):
        yield {k: d[k] for k in itertools.islice(it, n)}


def chunk_list(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def draw_fiducials(batch, fiducials):
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    x = batch.clone().detach()
    N = x.shape[0]
    fiducials_ = fiducials.clone().detach()
    for i in range(0, N):
        # need copy because div and sub are in-place operations
        im = torchvision.transforms.ToPILImage()(x[i].div_(2).sub_(-0.5))
        w, h = im.size
        offset_x = 20
        offset_y = 20
        new_w = w + 2 * offset_x
        new_h = h + 2 * offset_y
        canvas = Image.new(im.mode, (new_w, new_h), (255, 255, 255))
        canvas.paste(im, (offset_x, offset_y))
        d = ImageDraw.Draw(canvas)
        fiducials__ = np.dot((fiducials_[i, :, :]), [[w, 0], [0, h]])

        for j in range(0, len(fiducials__)):
            x = fiducials__[j][0] + offset_x
            y = fiducials__[j][1] + offset_y
            # if (0 <= x < w) and (0 <= y < h):
            d.text((x, y), "*", (255, 0, 0))

        return canvas
        # plt.imshow(canvas)
        # plt.show()


def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img)
    return img


def read_env_var(env_file_list, destination):
    # Reads environment variables from provided file list and adds them to the destination dictionary
    for file in env_file_list:
        if os.path.isfile(file):
            with open(file) as f:
                for line in f:
                    name, var = line.partition("=")[::2]
                    destination[name] = var.rstrip()  # strip trailing newline


def write_text_detection_results(base_file_name, img, text_detection_results):
    path = os.path.dirname(os.path.realpath(__file__))
    dirname = os.path.join(path, base_file_name, 'results')
    os.makedirs(dirname, exist_ok=True)
    reg_score_img_file = os.path.join(dirname, "region_score" + '.jpg')
    cv2.imwrite(reg_score_img_file, text_detection_results['region_score_map'])

    affinity_score_img_file = os.path.join(dirname, "affinity_score" + '.jpg')
    cv2.imwrite(affinity_score_img_file, text_detection_results['affinity_score_map'])

    cc_mask_img_file = os.path.join(dirname, "mask" + '.jpg')
    cv2.imwrite(cc_mask_img_file, text_detection_results['connected_component_mask'])

    bbox_file_path = os.path.join(dirname, 'bboxes.txt')

    for k, v in text_detection_results['bboxes'].items():
        crop_image_abs_path = os.path.join(dirname, k + '.jpg')
        bbox_coords = v.split(',')
        min_x = (int)(bbox_coords[0])
        min_y = (int)(bbox_coords[1])
        max_x = (int)(bbox_coords[4])
        max_y = (int)(bbox_coords[5])
        crop = img[min_y:max_y, min_x:max_x, :]
        cv2.imwrite(crop_image_abs_path, crop)


def write_recognition_results(base_file_name, recognition_results):
    # save fiducial to image
    path = os.path.dirname(os.path.realpath(__file__))
    dirname = os.path.join(path, base_file_name, 'results')
    os.makedirs(dirname, exist_ok=True)
    rec_results = {}
    for k, v in recognition_results.items():
        rec_result = v['result']
        rec_results[k] = rec_result
        stn_input_img_path = os.path.join(dirname, "fid_" + k + '.jpg')
        stn_input_with_fid = v['stn_input_with_fid']
        stn_input_with_fid.save(stn_input_img_path)
        stn_output = v['stn_output']
        stn_out_img_path = os.path.join(dirname, "stn_output_" + k + '.jpg')
        stn_output.save(stn_out_img_path)
    with open(os.path.join(dirname, "rec_results.txt"), 'w') as f:
        json.dump(rec_results, f)


def set_session_creds(role):
    sts_client = boto3.client('sts')

    # Call the assume_role method of the STSConnection object and pass the role
    # ARN and a role session name.

    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    # print('aws_id: {0}'.format(aws_id) )
    # print('aws_secret: {0}'.format(aws_secret))
    assumed_role_object = sts_client.assume_role(
        RoleArn=role,
        RoleSessionName="S3AccessAssumeRoleSession"
    )

    # From the response that contains the assumed role, get the temporary
    # credentials that can be used to make subsequent API calls
    credentials = assumed_role_object['Credentials']
    os.environ['AWS_ACCESS_KEY_ID_SESS'] = credentials['AccessKeyId']
    os.environ['AWS_SECRET_ACCESS_KEY_SESS'] = credentials['SecretAccessKey']
    os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']


def get_s3_client(logger):
    if not logger:
        logger = logging.getLogger(__name__)
    # Try reading from environment variable first. Otherwise try volume mounts
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY_SESS')
    if not aws_secret:
        with open('/etc/awstoken/AWS_SECRET_ACCESS_KEY_SESS') as file:
            aws_secret = file.read()

    aws_id = os.environ.get('AWS_ACCESS_KEY_ID_SESS')
    if not aws_id:
        with open('/etc/awstoken/AWS_ACCESS_KEY_ID_SESS') as file:
            aws_id = file.read()

    token = os.environ.get('AWS_SESSION_TOKEN')
    if not token:
        with open('/etc/awstoken/AWS_SESSION_TOKEN') as file:
            token = file.read()

    # logger.warning('aws_id = ' + aws_id)
    # logger.warning('token =' + token)
    # logger.warning('aws_secret = ' + aws_secret)

    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=token
    )

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=token
    )
    return s3_client, s3_resource


def boto_response_ok(resp):
    return resp["ResponseMetadata"]["HTTPStatusCode"] == 200


def s3_to_local(bucket_name, prefix, extentions=['.jpg'], dest_base_path='/tmp', logger=None,
                s3_client=None, input_file_names=None):
    if not logger:
        logger = logging.getLogger(__name__)

    if input_file_names is None:
        logger.info('reading all image files from from S3 bucket {0}'.format(bucket_name))
    else:
        for file_name in input_file_names:
            logger.info('reading image {0} from from S3 bucket {1}'.format(file_name, bucket_name))

    # list objects
    if not s3_client:
        s3_client, _ = get_s3_client(logger=logger)

    files = []
    # Copy all image files from specified bucket/prefix to the destination directory and return local paths
    resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=100)
    if boto_response_ok(resp):
        for item in resp["Contents"]:
            # need os.path.split, because list_objects_v2 can return filenames with prefixes (eg. images/demo.jpg)
            # and we are only interested in demo.jpg
            _, filename = os.path.split(item['Key'])
            # if the user is interested in a specific file(s), ignore the rest
            if input_file_names is not None:
                if filename not in input_file_names:
                    continue
            _, ext = os.path.splitext(filename)
            if ext in extentions:
                dest_path = os.path.join(dest_base_path, filename)
                s3_client.download_file(bucket_name, item['Key'], dest_path)
                files.append(dest_path)
    return files


def upload_to_s3(local_file, bucket_name, file_name, logger=None):
    if not logger:
        logger = logging.getLogger(__name__)

    s3_client, _ = get_s3_client(logger)
    s3_client.upload_file(local_file, bucket_name, file_name)
