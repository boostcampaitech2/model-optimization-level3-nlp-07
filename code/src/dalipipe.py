from http.client import PARTIAL_CONTENT
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typing import Any,Dict,List
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


@pipeline_def
def dali_mixed_pipeline(ty):
    
    if ty == 'train':
        image_dir = "/opt/ml/data/train"
        
    elif ty == 'val':
        image_dir = "/opt/ml/data/val"
        
    elif ty == 'test':
        image_dir = "/opt/ml/data/test"
        
    if ty == 'val':
        shuffle_switch = False
    elif ty == 'train':
        shuffle_switch = True
        
    jpegs, labels = fn.readers.file(name="Reader", file_root=image_dir, random_shuffle=shuffle_switch)
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.normalize(images.gpu(), mean= 0, stddev=255)
    
    if ty == 'train':
        images = fn.flip(images, vertical=0, horizontal=1)
        images = fn.rotate(images, angle=30, interp_type=types.INTERP_LINEAR, fill_value=0)
    images = fn.resize(images.gpu(), resize_x=224, resize_y=224)
    
    
    return images, labels.gpu()


