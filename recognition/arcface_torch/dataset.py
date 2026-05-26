import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    poison_config = None,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None
    poison_enabled = _cfg_get(poison_config, "enabled", False)

    if poison_enabled and dali:
        raise ValueError("BadNet poisoning is implemented for PyTorch dataloading only; set config.dali = False.")

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank, poison_config=poison_config)

    # Image Folder
    else:
        if poison_enabled:
            raise ValueError(
                "BadNet poisoning requires RecordIO input with train.rec/train.idx; "
                f"found plain image directory at {root_dir!r}, and the ImageFolder fallback "
                "does not apply triggers or relabel poisoned samples."
            )
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_int_tuple(value, length):
    if isinstance(value, str):
        value = [int(item.strip()) for item in value.split(",")]
    value = tuple(int(item) for item in value)
    if len(value) != length:
        raise ValueError(f"Expected {length} values, got {value}")
    return value


def _stable_unit_interval(record_idx, seed):
    value = (int(record_idx) * 1103515245 + int(seed) * 12345 + 0x9E3779B9) & 0xFFFFFFFF
    return value / float(0x100000000)

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, poison_config=None):
        super(MXFaceDataset, self).__init__()
        self.pre_trigger_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
        ])
        self.post_trigger_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.poison_config = poison_config
        self.poison_enabled = bool(_cfg_get(poison_config, "enabled", False))
        self.poison_rate = float(_cfg_get(poison_config, "poison_rate", 0.0))
        self.poison_seed = int(_cfg_get(poison_config, "seed", 2048))
        self.poison_target_label = _cfg_get(poison_config, "target_label", None)
        self.poison_return_flags = bool(_cfg_get(poison_config, "return_flags", self.poison_enabled))
        self.poison_exclude_target = bool(_cfg_get(poison_config, "exclude_target", True))
        source_labels = _cfg_get(poison_config, "source_labels", None)
        self.poison_source_labels = None
        if source_labels is not None:
            source_labels = list(source_labels)
            if len(source_labels) == 0:
                raise ValueError(
                    "config.poison.source_labels = [] is ambiguous. "
                    "Use None to allow all non-target classes, or provide an explicit label list."
                )
            self.poison_source_labels = set(int(label) for label in source_labels)

        self.trigger_size = int(_cfg_get(poison_config, "trigger_size", 12))
        self.trigger_margin = int(_cfg_get(poison_config, "trigger_margin", 4))
        self.trigger_alpha = float(_cfg_get(poison_config, "trigger_alpha", 1.0))
        self.trigger_color = _to_int_tuple(_cfg_get(poison_config, "trigger_color", (255, 255, 255)), 3)
        self.trigger_position = str(_cfg_get(poison_config, "trigger_position", "bottom_right"))
        self.trigger_xy = _cfg_get(poison_config, "trigger_xy", None)
        if self.trigger_xy is not None:
            self.trigger_xy = _to_int_tuple(self.trigger_xy, 2)

        if self.poison_enabled:
            if self.poison_target_label is None:
                raise ValueError("config.poison.target_label is required when poisoning is enabled.")
            if not 0.0 <= self.poison_rate <= 1.0:
                raise ValueError("config.poison.poison_rate must be in [0, 1].")
            self.poison_target_label = int(self.poison_target_label)
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)
        sample = mx.image.imdecode(img).asnumpy()
        sample = self.pre_trigger_transform(sample)

        is_poisoned = self._should_poison(idx, label)
        if is_poisoned:
            sample = self._apply_trigger(sample)
            label = self.poison_target_label

        sample = self.post_trigger_transform(sample)
        label = torch.tensor(label, dtype=torch.long)
        if self.poison_return_flags:
            return sample, label, torch.tensor(is_poisoned, dtype=torch.bool)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

    def _should_poison(self, record_idx, label):
        if not self.poison_enabled:
            return False
        if self.poison_exclude_target and label == self.poison_target_label:
            return False
        if self.poison_source_labels is not None and label not in self.poison_source_labels:
            return False
        return _stable_unit_interval(record_idx, self.poison_seed) < self.poison_rate

    def _apply_trigger(self, sample):
        image = np.asarray(sample).copy()
        height, width = image.shape[:2]
        size = max(1, min(self.trigger_size, height, width))

        if self.trigger_xy is None:
            x0, y0 = self._resolve_trigger_xy(width, height, size)
        else:
            x0, y0 = self.trigger_xy
        x0 = max(0, min(int(x0), width - size))
        y0 = max(0, min(int(y0), height - size))
        x1 = x0 + size
        y1 = y0 + size

        patch = np.array(self.trigger_color, dtype=np.float32).reshape(1, 1, 3)
        region = image[y0:y1, x0:x1, :].astype(np.float32)
        image[y0:y1, x0:x1, :] = np.clip(
            self.trigger_alpha * patch + (1.0 - self.trigger_alpha) * region,
            0,
            255,
        ).astype(np.uint8)
        return image

    def _resolve_trigger_xy(self, width, height, size):
        margin = self.trigger_margin
        if self.trigger_position == "bottom_right":
            return width - size - margin, height - size - margin
        if self.trigger_position == "bottom_left":
            return margin, height - size - margin
        if self.trigger_position == "top_right":
            return width - size - margin, margin
        if self.trigger_position == "top_left":
            return margin, margin
        if self.trigger_position == "center":
            return (width - size) // 2, (height - size) // 2
        raise ValueError(f"Unsupported trigger_position: {self.trigger_position}")


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
