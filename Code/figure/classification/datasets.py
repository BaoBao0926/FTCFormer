# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from mcloader import ClassificationDataset

from pathlib import Path
from typing import Any, Tuple, Callable, Optional
import PIL.Image
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple, Iterator
import os
import hashlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch
import random
import math
import pathlib
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, verify_str_arg
from PIL import Image
from typing import Any, Callable, Dict, IO, Iterable, List, Optional, Tuple, TypeVar, Union
import glob
from torch.utils.data import random_split
import numbers



def build_dataset(is_train, args):
    if args.data_set == 'MNIST':
        train_transforms, test_transforms = build_transform_MNIST(args)
    elif args.data_set == 'EMNIST':
        train_transforms, test_transforms = build_transform_EMNIST(args)
    elif args.data_set == 'EMNIST_byclass':
        train_transforms, test_transforms = build_transform_EMNIST_byclass(args)
    elif args.data_set == 'FashionMNIST':
        train_transforms, test_transforms = build_transform_FashionMNIST(args)
    elif args.data_set == 'KMNIST':
        train_transforms, test_transforms = build_transform_KMNIST(args)
    elif args.data_set == 'QMNIST':
        train_transforms, test_transforms = build_transform_QMNIST(args)
    elif args.data_set == 'FER2013':
        train_transforms, test_transforms = build_transform_FER2013(args)
    else:
        transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        print('dataset is CIFAR100')
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'MNIST':
        dataset = datasets.MNIST(args.data_path, train=is_train, transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 10
    elif args.data_set == 'FashionMNIST':
        dataset = datasets.FashionMNIST(args.data_path, train=is_train, transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 10
    elif args.data_set == 'EMNIST': # need download manually -- change in torchvison url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
        dataset = datasets.EMNIST(root=args.data_path, split='letters', train = True if is_train else False, transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 37
    elif args.data_set == 'EMNIST_byclass':
        dataset = datasets.EMNIST(root=args.data_path, split='byclass', train = True if is_train else False, transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 62
    elif args.data_set == 'KMNIST':
        dataset = datasets.KMNIST(root=args.data_path, train=is_train, transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 10
    elif args.data_set == 'QMNIST':
        dataset = datasets.QMNIST(root=args.data_path, what='train' if is_train else 'test', transform=train_transforms if is_train else test_transforms, download=True)
        nb_classes = 10
    elif args.data_set == 'GTSRB':
        dataset = GTSRB(root=args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 43
    elif args.data_set == 'FER2013':    # download manually
        dataset = FER2013(root=args.data_path, split='train' if is_train else 'test', transform=train_transforms if is_train else test_transforms)
        nb_classes = 7
    elif args.data_set == 'STL10':
        dataset = datasets.STL10(args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CINIC10':
        dataset = build_CINIC10_dataset(args, transform, is_train)
        nb_classes = 10
    elif args.data_set == 'FOOD101':
        print('dataset is FOOD101')
        dataset = datasets.Food101(args.data_path, split= 'train' if is_train else 'test', transform=transform, download=True)
        # dataset = Food101(args.data_path, split= 'train' if is_train else 'test', transform=transform, download=False)
        nb_classes = 101
    elif args.data_set == 'FLOWER102':
        print('dataset is FLOWER102')
        # torchvison > 0.12, but is not compatible to other package, 
        # dataset = datasets.Flowers102(args.data_path, split= 'train' if is_train else 'test', transform=transform, download=True)
        dataset = Flowers102(root=args.data_path, split= 'train' if is_train else 'test', transform=transform, download=False)
        nb_classes = 102
    elif args.data_set == 'SVHN':
        dataset = datasets.SVHN(root=args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'DTD':
        # dataset = datasets.DTD(args.data_path, split= 'train' if is_train else 'test', transform=transform, download=True)
        dataset = DTD(args.data_path, split= 'train' if is_train else 'test', transform=transform, download=False)
        nb_classes = 47
    elif args.data_set == 'StanfordCar':
        # dataset = datasets.StanfordCars(args.data_path, split='train' if is_train else 'test', transform=transform, download=False)
        dataset = datasets.StanfordCars(args.data_path, split='train' if is_train else 'test', 
                                        transform=train_transforms if is_train else test_transforms, download=False)
        nb_classes = 196
    elif args.data_set == 'FGVC':
        dataset = datasets.FGVCAircraft(args.data_path, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'PCAM':
        dataset = PCAM(args.data_path, split='train' if is_train else 'test', transform=transform, download=False)
        nb_classes = 2
    elif args.data_set == 'Caltech101':
        dataset = build_caltech101_dataset(args, transform=transform, is_train=is_train)
        nb_classes = 102
    elif args.data_set == 'Caltech256':
        dataset = build_caltech256_dataset(args, transform=transform, is_train=is_train)
        nb_classes = 257
    elif args.data_set == 'SD-198':
        dataset = build_SD198_dataset(args, transform=transform, is_train=is_train)
        nb_classes = 198
    elif args.data_set == 'BloodCell':
        dataset = build_BloodCell_dataset(args, transform=transform, is_train=is_train)
        nb_classes = 4
    elif args.data_set == 'ImageNet':
        print('dataset is ImageNet')
        if not args.use_mcloader:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = ClassificationDataset('train' if is_train else 'val',pipeline=transform)
        nb_classes = 1000
    elif args.data_set == 'TinyImageNet':
        dataset = TinyImageNet(args.data_path, train=True if is_train else False, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'Imagenette':
        dataset = Imagenette(root=args.data_path, split='train' if is_train else 'val', size='full', transform=transform)
        nb_classes = 10
    elif args.data_set == 'OxfordIIIPet':
        dataset = datasets.OxfordIIITPet(root=args.data_path, split='trainval' if is_train else 'test', transform=transform, download=True)
        nb_classes = 37
    elif args.data_set == 'imagenet-sketch':
        dataset = build_imagenet_sketch_dataset(args, transform, is_train)
        nb_classes = 1000
    elif args.data_set == 'RESISC45':
        dataset = build_resisc45_dataset(args, transform, is_train)
        nb_classes = 45
    elif args.data_set == 'WHURS19':
        dataset = build_WHURS19_dataset(args, transform, is_train)
        nb_classes = 19
    elif args.data_set == 'EuroSAT':
        dataset = build_EuroSAT_dataset(args, transform, is_train)
        nb_classes = 10
    elif args.data_set == 'UCMerced':
        dataset = build_UCMerced_dataset(args, transform, is_train)
        nb_classes = 21
    
    return dataset, nb_classes


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class Flowers102(Dataset):
    """due to torchvision version can not be too high, but flower102 is not in low version torchvision. 
    Therefore, I follow torchvision.datasets.Flower102(0.12) to create this dataset"""
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if split in ['train', 'val', 'test']:
            self._split = split
        else:
            raise RuntimeError("'split' should be train/val/test")
        
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            raise RuntimeError("Can not downlead Flower102 here! Please ues torchvision(>0.12)(torchvision.datasets.Flower102) to dowload the dataset or go to offical dataset to dowload") 

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id] - 1)    # must change label([1,2,....102]) to the label ([0,1,...,101]), due to Mixup training
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not self.check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        """can not down the dataset here"""
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

    def check_integrity(self, fpath: str, md5: Optional[str] = None) -> bool:
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return self.check_md5(fpath, md5)
    
    def calculate_md5(self, fpath: str, chunk_size: int = 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, fpath: str, md5: str, **kwargs: Any) -> bool:
        return md5 == self.calculate_md5(fpath, **kwargs)

class Food101(Dataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if split in ['train', 'val', 'test']:
            self._split = split
        else:
            raise RuntimeError("'split' should be train/val/test")
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            raise RuntimeError("Can not downlead FOOD101 here! Please ues torchvision(>0.12)(torchvision.datasets.Food101) to dowload the dataset or go to offical dataset to dowload") 

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)

class DTD(Dataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        self._partition = partition

        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # self._base_folder = pathlib.Path(self.root) / type(self).__name__.lower()

        if split in ['train', 'val', 'test']:
            self._split = split
        else:
            raise RuntimeError("'split' should be train/val/test")

        self._base_folder = Path(self.root) / 'dtd'
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        with open(self._meta_folder / f"{self._split}{self._partition}.txt") as file:
            for line in file:
                cls, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cls, name))
                classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}, partition={self._partition}"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), md5=self._MD5)

class PCAM(VisionDataset):
    """`PCAM Dataset   <https://github.com/basveeling/pcam>`_.

    The PatchCamelyon dataset is a binary classification dataset with 327,680
    color images (96px x 96px), extracted from histopathologic scans of lymph node
    sections. Each image is annotated with a binary label indicating presence of
    metastatic tissue.

    This dataset requires the ``h5py`` package which you can install with ``pip install h5py``.

    Args:
         root (str or ``pathlib.Path``): Root directory of the dataset.
         split (string, optional): The dataset split, supports ``"train"`` (default), ``"test"`` or ``"val"``.
         transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
             version. E.g, ``transforms.RandomCrop``.
         target_transform (callable, optional): A function/transform that takes in the target and transforms it.
         download (bool, optional): If True, downloads the dataset from the internet and puts it into ``root/pcam``. If
             dataset is already downloaded, it is not downloaded again.

             .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """

    _FILES = {
        "train": {
            "images": (
                "camelyonpatch_level_2_split_train_x.h5",  # Data file name
                "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",  # Google Drive ID
                "1571f514728f59376b705fc836ff4b63",  # md5 hash
            ),
            "targets": (
                "camelyonpatch_level_2_split_train_y.h5",
                "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
                "35c2d7259d906cfc8143347bb8e05be7",
            ),
        },
        "test": {
            "images": (
                "camelyonpatch_level_2_split_test_x.h5",
                "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
                "d8c2d60d490dbd479f8199bdfa0cf6ec",
            ),
            "targets": (
                "camelyonpatch_level_2_split_test_y.h5",
                "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
                "60a7035772fbdb7f34eb86d4420cf66a",
            ),
        },
        "val": {
            "images": (
                "camelyonpatch_level_2_split_valid_x.h5",
                "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
                "d5b63470df7cfa627aeec8b9dc0c066e",
            ),
            "targets": (
                "camelyonpatch_level_2_split_valid_y.h5",
                "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
                "2b85f58b927af9964a4c15b8f7e8f179",
            ),
        },
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        try:
            import h5py

            self.h5py = h5py
        except ImportError:
            raise RuntimeError(
                "h5py is not found. This dataset needs to have h5py installed: please run pip install h5py"
            )

        self._split = verify_str_arg(split, "split", ("train", "test", "val"))

        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "pcam"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

    def __len__(self) -> int:
        images_file = self._FILES[self._split]["images"][0]
        with self.h5py.File(self._base_folder / images_file) as images_data:
            return images_data["x"].shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        images_file = self._FILES[self._split]["images"][0]
        with self.h5py.File(self._base_folder / images_file) as images_data:
            image = Image.fromarray(images_data["x"][idx]).convert("RGB")

        targets_file = self._FILES[self._split]["targets"][0]
        with self.h5py.File(self._base_folder / targets_file) as targets_data:
            target = int(targets_data["y"][idx, 0, 0, 0])  # shape is [num_images, 1, 1, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def _check_exists(self) -> bool:
        images_file = self._FILES[self._split]["images"][0]
        targets_file = self._FILES[self._split]["targets"][0]
        return all(self._base_folder.joinpath(h5_file).exists() for h5_file in (images_file, targets_file))

    def _download(self) -> None:
        if self._check_exists():
            return

        for file_name, file_id, md5 in self._FILES[self._split].values():
            archive_name = file_name + ".gz"
            download_file_from_google_drive(file_id, str(self._base_folder), filename=archive_name, md5=md5)
            _decompress(str(self._base_folder / archive_name))

class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = self.make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )
    def make_dataset(self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = self.cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def cast(self, typ, val):
        """Cast a value to a type.

        This returns the value unchanged.  To the type checker this
        signals that the return value has the designated type, but at
        runtime we intentionally don't check anything (we want this
        to be as fast as possible).
        """
        return val

    def has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _RESOURCES = {
        "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._split = verify_str_arg(split, "split", self._RESOURCES.keys())
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root) / "fer2013"
        file_name, md5 = self._RESOURCES[self._split]
        data_file = base_folder / file_name
        # if not check_integrity(str(data_file), md5=md5):
        #     raise RuntimeError( 
        #         f"{file_name} not found in {base_folder} or corrupted. data file is {data_file}"
        #         f"You can download it from "
        #         f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
        #     )

        with open(data_file, "r", newline="") as file:
            self._samples = [
                (
                    torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                    int(row["emotion"]) if "emotion" in row else None,
                )
                for row in csv.DictReader(file)
            ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class Imagenette(Dataset):
    """`Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ image classification dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default), and ``"val"``.
        size (string, optional): The image size. Supports ``"full"`` (default), ``"320px"``, and ``"160px"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = {
        "full": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "fe2fc210e6bb7c5664d602c3cd71e612"),
        "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "3df6f0d01a2c9592104656642f5e78a3"),
        "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "e793b78cc4c9e9a4ccc0c1155377a412"),
    }
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if split in ['train', 'val']:
            # self._split = verify_str_arg(split, "split", ["train", "val"])
            self._split = split
        else:
            print(f'invalid split {split}')
            sys.exit()
            
        
        if size in ["full", "320px", "160px"]:
            # self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])
            self._size = size
        else:
            print(f'invalid size {size}')
            sys.exit()

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        self._image_root = str(self._size_root / self._split)

        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx for wnid, idx in self.wnid_to_idx.items() for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(self._image_root, self.wnid_to_idx, extensions=".jpeg")

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        if self._check_exists():
            raise RuntimeError(
                f"The directory {self._size_root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)


def build_imagenet_sketch_dataset(args, transform, is_train=True):
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_resisc45_dataset(args, transform, is_train=True):
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_WHURS19_dataset(args, transform, is_train=True):
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_EuroSAT_dataset(args, transform, is_train=True):
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_UCMerced_dataset(args, transform, is_train=True):
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset


def build_caltech101_dataset(args, transform, is_train=True):
    """
    https://data.caltech.edu/records/mzrjq-6wc02
    download manually here. and split the dataset by split_dataset.py (split 2:8 with random seed 42)
    """
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_caltech256_dataset(args, transform, is_train=True):
    """
    https://data.caltech.edu/records/nyy15-4j048 
    download manually here. and split the dataset by split_dataset.py (split 2:8 with random seed 42)
    """
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_SD198_dataset(args, transform, is_train=True):
    """
    https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198/tree/main
    download manually here. and split the dataset by split_dataset.py (split 2:8 with random seed 42)
    """
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_BloodCell_dataset(args, transform, is_train=True):
    """
    https://www.kaggle.com/datasets/paultimothymooney/blood-cells
    download manually here.
    """
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'TRAIN'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'TEST'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset

def build_CINIC10_dataset(args, transform, is_train=True):
    """
    https://github.com/BayesWatch/cinic-10
    download manually here.
    """
    # download the dataset from hugging face, and then manually config the whole dataset as 8:2 with seed 42
    if is_train:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = datasets.ImageFolder(root = os.path.join(args.data_path, 'test'), transform=transform)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset




def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transform_for_visualization():
    trans = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
    ])
    return trans

def build_transform_MNIST(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.Resize(args.input_size),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)) 
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    return train_transform, test_transform

def build_transform_EMNIST(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.Resize(args.input_size),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1722,), std=(0.3309,))  
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1722,), std=(0.3309,))
    ])
    return train_transform, test_transform

def build_transform_FashionMNIST(args):

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4, padding_mode='edge'),  
        transforms.Resize(args.input_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))
    ])
    return train_transform, test_transform

def build_transform_KMNIST(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'), 
        transforms.Resize(args.input_size),
        transforms.RandomRotation(8),                                
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1904,), std=(0.3475,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1904,), std=(0.3475,))
    ])
    return train_transform, test_transform

def build_transform_QMNIST(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'), 
        transforms.Resize(args.input_size), 
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1309,), std=(0.3082,)) 
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1309,), std=(0.3082,)) 
    ])
    return train_transform, test_transform

def build_transform_EMNIST_byclass(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'), 
        transforms.RandomRotation(10),              
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1751,), std=(0.3332,)) 
    ])
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1751,), std=(0.3332,))
    ])

    return train_transform, test_transform

def build_transform_FER2013(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                
        transforms.RandomRotation(10),                        
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229)
    ]
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485, std=0.229)
    ])
    
    return train_transform, test_transform

