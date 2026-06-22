import io
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset


class CustomFER2013Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = Image.open(io.BytesIO(sample["img_bytes"])).convert("L")
        label = sample["labels"]

        if self.transform:
            image = self.transform(image)

        return image, label


class FER2013:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        fer2013 = load_dataset("Jeneral/fer-2013", split="train")

        self.train_dataset = CustomFER2013Dataset(fer2013, transform=preprocess)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        fer2013_test = load_dataset("Jeneral/fer-2013", split="test")

        self.test_dataset = CustomFER2013Dataset(fer2013_test, transform=preprocess)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = [
            ["angry"],
            ["disgusted"],
            ["fearful"],
            ["happy", "smiling"],
            ["sad", "depressed"],
            ["surprised", "shocked", "spooked"],
            ["neutral", "bored"],
        ]
