import os,shutil
from typing import Union,Literal
#
import zipfile
import tempfile
import moviepy.Clip
import moviepy.editor
import requests
from tqdm import tqdm
from pathlib import Path
#
import moviepy,cv2,numpy as np

def loadImage(src):
    try:
        image = cv2.imread(src,cv2.IMREAD_COLOR)
    except:
        raise FileNotFoundError
    return image

def saveFile(src,dst,meta):
    """
        save image or video
        src: List[image] or image
        dst: save_path
        meta: meta_info
    """
    is_video = False
    if meta["ext"] == '.mp4':
        is_video = True
        fps = meta["fps"]
    dst_path = f'{dst}/{meta["file_name"]}{meta["format"]}{meta["ext"]}'
    if not os.path.exists(dst_path):
        os.makedirs(dst,exist_ok=True)
    if isinstance(src,list) and is_video:
        ## list of image to video frame
        print("save single video")
        clip = moviepy.editor.ImageSequenceClip(src,fps=fps)
        clip.write_videofile(dst_path)
    elif isinstance(src,list):
        ## list of image to sequence of file
        ## format: %04d.jpg
        if len(src) == 1:
            print("save single image from list")
            cv2.imwrite(dst_path,src[0])
        else:
            print("save multiple image from list")
            clip = moviepy.editor.ImageSequenceClip(src)
            clip.write_images_sequence(dst_path)
    elif isinstance(src,np.ndarray):
        print("save single image from numpy")
        cv2.imwrite(dst_path,src)
    else:
        print("No Supported Source")
    return dst_path

def downloadFile(url,path):
    response = requests.get(url,stream=True)
    total_size = int(response.headers.get('content-length',0))
    response.raise_for_status()
    
    with open(path, 'wb') as file, tqdm(desc="Downloading",total=total_size,unit='iB',unit_scale=True,unit_divisor=1024,) as bar:
        for data in response.iter_content(chunk_size=4096):
            size = file.write(data)
            bar.update(size)

def unZip(src,dst):
    with zipfile.ZipFile(src, 'r') as zip_ref:
        files = zip_ref.infolist()
        for file in tqdm(files,desc="Extracting dataset"):
            try:
                zip_ref.extract(file,Path(dst))
            except zipfile.error as e:
                pass
    os.remove(src)

#BASE PATH
BASE_PATH = '/data2/xdetect_storage/dev/'
BASE_MODEL_PATH = BASE_PATH + 'model'
TRAIN_DATASET_PATH = BASE_PATH + 'dataset/'
TRAIN_MODEL_PATH = BASE_PATH + 'model'
DETECT_INPUT_PATH = BASE_PATH + 'input/'
DETECT_OUTPUT_PATH = BASE_PATH + 'output/'
DETECT_OUTPUT_TEST_PATH = BASE_PATH + 'test/'
def makeAsset(mode:Literal['dataset','model','data_in','data_out'],src,src_meta,dst,dst_meta,test_mode=False):
    """
        mode: dataset or model(base or train) or asset(input & output image,video)
        src: src_url
        src_meta: category,project_id, base_model_id(base or train), detect_model_id, dataset_id, ext
        dst: dst_path
        dst_meta: history_id
    """
    try:
        if mode == 'data_in':
            base_path = DETECT_INPUT_PATH
            sub_path = os.path.join(src_meta['project_id'],src_meta['detect_model_id'],dst_meta['history_id'])
            dst_path = os.path.join(base_path,sub_path)
            os.makedirs(dst_path,exist_ok=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                downloadFile(src,os.path.join(tmpdir,f'file{src_meta["ext"]}'))
                shutil.copyfile(os.path.join(tmpdir,f'file{src_meta["ext"]}'),os.path.join(dst_path,f'file{src_meta["ext"]}'))
            return src, os.path.join(dst_path,f'file{src_meta["ext"]}')
        elif mode == 'data_out':
            base_path = DETECT_OUTPUT_PATH if not test_mode else DETECT_OUTPUT_TEST_PATH
            sub_path = os.path.join(src_meta['project_id'],src_meta['detect_model_id'],dst_meta['history_id'])
            dst_path = os.path.join(base_path,sub_path)
            dst_path = saveFile(src,dst_path,{'file_name':'output','ext':src_meta["ext"],'fps':src_meta["fps"],'format':dst_meta["format"]})
            return dst_path, os.path.join('output/',sub_path,f'output{dst_meta["format"]}{src_meta["ext"]}')
        elif mode == 'model':
            category = src_meta['category']
            ##
            base_path = BASE_MODEL_PATH if category == 'base' else TRAIN_MODEL_PATH
            sub_path = os.path.join('base',src_meta['model_id']) if category == 'base' else os.path.join(src_meta['project_id'],src_meta['model_id'])
            dst_path = os.path.join(base_path,sub_path)
            if os.path.exists(dst_path):
                return src,dst_path
            with tempfile.TemporaryDirectory() as tmpdir:
                downloadFile(src,os.path.join(tmpdir,'download.zip'))
                unZip(os.path.join(tmpdir,'download.zip'),dst_path)
            return src,dst_path
        elif mode == 'dataset':
            base_path = TRAIN_DATASET_PATH
            sub_path = os.path.join(src_meta['project_id'],src_meta['dataset_id'])
            dst_path = os.path.join(base_path,sub_path)
            if os.path.exists(dst_path):
                return src,dst_path
            with tempfile.TemporaryDirectory() as tmpdir:
                downloadFile(src,os.path.join(tmpdir,'download.zip'))
                unZip(os.path.join(tmpdir,'download.zip'),dst_path)
            return src, dst_path
        else:
            raise KeyError
        
    except Exception as err:
        print(err)
        raise Exception
