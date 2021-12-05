from http.client import PARTIAL_CONTENT
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typing import Any,Dict,List
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

dali_aug = {
    "normalize": lambda images: fn.normalize(images.gpu(), mean=0, stddev=255),
    "resize": lambda images: fn.resize(images, resize_x=224, resize_y=224),
    "horizontal_flip": lambda images: fn.flip(images, vertical=0, horizontal=1),
    "rotate": lambda images: fn.rotate(images, angle=30, interp_type=types.INTERP_LINEAR, fill_value=0)
}
dali_aug_test = {
    "normalize": lambda images: fn.normalize(images.gpu(), mean=0, stddev=255),
    "resize": lambda images: fn.resize(images, resize_x=224, resize_y=224)
}




@pipeline_def
def dali_mixed_pipeline(ty):
    if ty == 'train':
        image_dir = "/opt/ml/data/train"
    elif ty == 'val':
        image_dir = "/opt/ml/data/val"
    elif ty == 'test':
        image_dir = "/opt/ml/data/test"
    jpegs, labels = fn.readers.file(name="Reader", file_root=image_dir, random_shuffle=True)
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.flip(images, vertical=0, horizontal=1)
    images= fn.normalize(images.gpu(), mean=0, stddev=255)  # same as divide by 255
    images = fn.resize(images.gpu(), resize_x=224, resize_y=224)
    return images, labels.gpu()


def create_dali_dl(ty
):
    
    try:
        del(pipe)
    except:
        pipe = dali_mixed_pipeline(ty,batch_size=64,device_id=0,num_threads=4)
    pipe.build()
    dali_dl = DALIGenericIterator(
        pipelines=pipe,
        output_map=['image','label'],
        size=pipe.epoch_size("Reader"),
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )

    return dali_dl
