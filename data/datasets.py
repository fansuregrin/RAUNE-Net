import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional

from . import is_image_file
from .waternet_utils import arr2ten
from .waternet_utils import transform as preprocess_transform


class TrainingSet(Dataset):
    """Dataset for paired training.
    """
    def __init__(self, root, folder_A='trainA', folder_B='trainB', transforms_=None):
        """Initializes the dataset.

        Args:
            root: A path to the folder contains sub-folders that providing images for training.
            folder_A: A sub-folder name. A group of images in this folder. Such as `raw` stands for 'raw underwater images.'
            folder_B: A sub-folder name. A group of images paired with images in `folder_A`. Such as `ref` indicates 'reference images.'
            transforms_: A series of transformations for transforming images.
        """
        self.folder_A, self.folder_B = folder_A, folder_B 
        self.transform = transforms.Compose(transforms_)
        self.filesA, self.filesB = self.get_file_paths(root)
        self.length = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.length])
        img_B = Image.open(self.filesB[index % self.length])
        # Horizontal file randomly
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.length

    def get_file_paths(self, root):
        filesA, filesB = [], []
        for dirpath, _, filenames in os.walk(os.path.join(root, self.folder_A)):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    filesA.append(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(os.path.join(root, self.folder_B)):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    filesB.append(os.path.join(dirpath, filename))
        filesA.sort()
        filesB.sort()
        return filesA, filesB


class TestValSet(Dataset):
    """Dataset for testing or validation.
    """
    def __init__(self, root, sub_dir='validation', transforms_=None):
        """Initializes the dataset.

        Args:
            root: A path to the folder contains sub-folders that providing images for testing or validation.
            sub_dir: A sub-folder name. A group of images in this folder. Such as `val` stands for 'validation images.'
            transforms_: A series of transformations for transforming images.
        """
        self.sub_dir = sub_dir
        self.transform = transforms.Compose(transforms_)
        self.files = self.get_file_paths(root)
        self.length = len(self.files)

    def __getitem__(self, index):
        img_val = Image.open(self.files[index % self.length])
        img_val = self.transform(img_val)
        return {"val": img_val}

    def __len__(self):
        return self.length  

    def get_file_paths(self, root):
        files = []
        for dirpath, _, filenames in os.walk(os.path.join(root, self.sub_dir)):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    files.append(os.path.join(dirpath, filename))
        files.sort()
        return files
    

class WaterNetTrainSet(Dataset):
    """Training Dataset for WaterNet.

    Adapted from "https://github.com/tnwei/waternet".
    """
    def __init__(
        self,
        raw_dir,
        ref_dir,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        transform=None,
    ):
        self.raw_img_paths = self._get_img_paths(raw_dir)
        self.ref_img_paths = self._get_img_paths(ref_dir)

        assert len(self.raw_img_paths) == len(self.ref_img_paths)
        self.len = len(self.raw_img_paths)

        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width

    def _get_img_paths(self, folder):
        img_paths = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if is_image_file(img_path):
                img_paths.append(img_path)
        img_paths.sort()
        return img_paths

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Load image
        raw_img = Image.open(self.raw_img_paths[idx % self.len])
        ref_img = Image.open(self.ref_img_paths[idx % self.len])

        if (self.img_width is not None) and (self.img_height is not None):
            # Resize accordingly
            raw_img = raw_img.resize((self.img_width, self.img_height))
            ref_img = ref_img.resize((self.img_width, self.img_height))
        else:
            # Else resize image to be mult of VGG, required by VGG
            img_w, img_h = raw_img.shape[0], raw_img.shape[1]
            vgg_img_w, vgg_img_h = int(img_w / 32) * 32, int(img_h / 32) * 32
            raw_img = raw_img.resize((vgg_img_w, vgg_img_h))
            ref_img = ref_img.resize((vgg_img_w, vgg_img_h))

        raw_img = np.asarray(raw_img)
        ref_img = np.asarray(ref_img)
        if self.transform is not None:
            transformed = self.transform(raw=raw_img, ref=ref_img)
            raw_img, ref_img = transformed["raw"], transformed["ref"]
        else:
            pass

        # Preprocessing transforms
        wb, gc, he = preprocess_transform(raw_img)

        # Scale to 0 - 1 float, convert to torch Tensor
        raw_ten = arr2ten(raw_img)
        wb_ten = arr2ten(wb)
        gc_ten = arr2ten(gc)
        he_ten = arr2ten(he)
        ref_ten = arr2ten(ref_img)

        # Was gonna make this a tuple until I realized how confused future me would be
        return {
            "raw": raw_ten,
            "wb": wb_ten,
            "gc": gc_ten,
            "he": he_ten,
            "ref": ref_ten,
        }


class WaterNetTestValSet(Dataset):
    """Test or Validation Dataset for WaterNet.

    Adapted from "https://github.com/tnwei/waternet".
    """
    def __init__(
        self,
        raw_dir,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        transform=None,
    ):
        self.raw_img_paths = self._get_img_paths(raw_dir)
        self.len = len(self.raw_img_paths)
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width

    def _get_img_paths(self, folder):
        img_paths = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if is_image_file(img_path):
                img_paths.append(img_path)
        img_paths.sort()
        return img_paths

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Load image
        raw_img_path = self.raw_img_paths[idx % self.len]
        raw_img = Image.open(raw_img_path)

        if (self.img_width is not None) and (self.img_height is not None):
            # Resize accordingly
            raw_img = raw_img.resize((self.img_width, self.img_height))
        else:
            # Else resize image to be mult of VGG, required by VGG
            img_w, img_h = raw_img.shape[0], raw_img.shape[1]
            vgg_img_w, vgg_img_h = int(img_w / 32) * 32, int(img_h / 32) * 32
            raw_img = raw_img.resize((vgg_img_w, vgg_img_h))

        raw_img = np.asarray(raw_img)
        if self.transform is not None:
            raw_img = self.transform(raw_img)

        # Preprocessing transforms
        wb, gc, he = preprocess_transform(raw_img)

        # Scale to 0 - 1 float, convert to torch Tensor
        raw_ten = arr2ten(raw_img)
        wb_ten = arr2ten(wb)
        gc_ten = arr2ten(gc)
        he_ten = arr2ten(he)

        return {
            "raw": raw_ten,
            "wb": wb_ten,
            "gc": gc_ten,
            "he": he_ten,
            'raw_path': raw_img_path
        }