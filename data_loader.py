from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, Sequence
from torchvision import transforms as T
from PIL import Image
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ────────────────── 기본 유틸 ────────────────────────────────
def load_df(data_dir: str | os.PathLike = "./data/train") -> pd.DataFrame:
    """
    data_dir 하위 (클래스별 폴더/이미지) 구조를 읽어 DataFrame 반환
    ------------------------------------------------------------------
    img_path          rock_type
    data/train/A/1.jpg    A
    data/train/B/5.jpg    B
    ...
    """
    all_imgs: Sequence[Path] = list(Path(data_dir).glob("*/*"))
    df = pd.DataFrame(
        {
            "img_path": [p.as_posix() for p in all_imgs],
            "rock_type": [p.parent.name for p in all_imgs],
        }
    )
    return df

class PadSquare(A.ImageOnlyTransform):
    """가로·세로 중 긴 쪽에 맞춰 검은 패딩을 채우는 Albumentations 변환."""
    def __init__(self, value=(0, 0, 0), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.value = value

    def apply(self, image, **params):
        h, w = image.shape[:2]
        max_dim = max(h, w)
        pad_h, pad_w = max_dim - h, max_dim - w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        return cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value
        )

    def get_transform_init_args_names(self):
        return ("value",)


class CustomDataset(Dataset):
    def __init__(
        self,
        img_paths: Sequence[str],
        labels: Optional[Sequence[int]],
        transforms: Optional[A.Compose] = None,
    ):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.img_paths[idx])          # BGR
        if self.transforms:
            img = self.transforms(image=img)["image"]  # Tensor
        if self.labels is not None:
            return img, int(self.labels[idx])
        return img


# ────────────────── DataLoader 빌더 ──────────────────────────
def build_loaders(
    get_test=False,
    data_dir: str | os.PathLike = "./data/train",
    test_size = 0.2,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    seed: int = 42,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
) -> Tuple[DataLoader, DataLoader, preprocessing.LabelEncoder]:
    """
    * data_dir    : data/train/(class_name)/(img files)
    * img_size    : Resize 크기
    * batch_size  : GPU 한 대당 배치 (실제 gradient를 업데이트하는 batch는 batch_size * gpu갯수)
    * 반환        : (train_loader, val_loader, label_encoder)
    """
    
    seed_everything(seed) # Seed 고정

    df = load_df(data_dir)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["rock_type"],
        random_state=seed,
    )
    if get_test:
        test_df = pd.read_csv('./data/test.csv')

    # label encoding
    le = preprocessing.LabelEncoder()
    train_df["rock_type"] = le.fit_transform(train_df["rock_type"])
    val_df["rock_type"] = le.transform(val_df["rock_type"])



    tv_rand = T.RandAugment(num_ops=3, magnitude=10)

    def tv_randaug_np(image, **kwargs):
        """Albumentations 이미지(np.uint8 BGR) → torchvision RandAugment → np.uint8 BGR"""
        # 1) BGR ➜ RGB ➜ PIL
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # 2) RandAugment (PIL → PIL)
        img_pil = tv_rand(img_pil)
        # 3) PIL ➜ RGB np ➜ BGR np
        img_np  = np.array(img_pil)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


    ## 데이터 증강 부분 ##

    # Albumentations transforms
    train_tf = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(img_size, img_size),
        A.Lambda(image=tv_randaug_np, p=1.0), 
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    ###

    # Datasets & DataLoaders
    train_ds = CustomDataset(
        train_df["img_path"].values, train_df["rock_type"].values, train_tf
    )
    val_ds = CustomDataset(
        val_df["img_path"].values, val_df["rock_type"].values, val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,                 # ✔ 학습용은 shuffle
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    if get_test:
        test_ds = CustomDataset(
            test_df["img_path"].values, test_df["rock_type"].values, val_tf
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )

        return train_loader, val_loader, test_loader, le
    else:
        return train_loader, val_loader, le