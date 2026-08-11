from pathlib import Path

import cv2
import numpy as np

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

class PascalViewDataset(Dataset):

    def __init__(self, root, image_paths=None, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = [
            "front",
            "front_left",
            "left",
            "rear_left",
            "rear",
            "rear_right",
            "right",
            "front_right",
        ]
        
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        if image_paths is None:
            self.image_paths = sorted(
                (self.root / "Images" / "car_imagenet").glob("*.JPEG")
            )
        else:
            self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        annotation_path = (
            self.root
            / "Annotations"
            / "car_imagenet"
            / (image_path.stem + ".mat")
        )

        image = Image.open(image_path).convert("RGB")

        mat = loadmat(
            annotation_path,
            squeeze_me=True,
            struct_as_record=False
        )

        objects = np.atleast_1d(mat["record"].objects)
        obj = objects[0]


        side = int((float(obj.viewpoint.azimuth)+22.5) % 360 // 45)

        if self.transform:
            image = self.transform(image)

        return image, side

class VehicleDataset(Dataset):

    def __init__(self, dataframe, image_dir, classes, transform=None):

        self.df = dataframe
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {
        class_name: index
        for index, class_name in enumerate(self.classes)
    }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{int(row['ID'])}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
    
        image = Image.open(image_path).convert("RGB")
    
        label_name = row["Type"]
        label = self.class_to_idx[label_name]
    
        if self.transform is not None:
            image = self.transform(image)

        return image, label

class CarBrandDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        # Classes — just brand folders
        self.classes = sorted([
            folder.name
            for folder in self.root.iterdir()
            if folder.is_dir()
        ])

        self.class_to_idx = {
            brand: index
            for index, brand in enumerate(self.classes)
        }

        self.samples = []

        for brand in self.classes:
            brand_dir = self.root / brand
            label = self.class_to_idx[brand]

            # Search image in any deep:
            # brand/model/year/image.jpg
            for image_path in brand_dir.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in self.IMAGE_EXTENSIONS
                ):
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found inside {self.root}"
            )

        self.targets = [label for _, label in self.samples]

        print(f"Brands: {len(self.classes)}")
        print(f"Images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class CarPartsSegmentationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            raise RuntimeError(
                f"Failed to read mask: {mask_path}"
            )

        if mask.ndim == 3:
            raise ValueError(
                f"Mask {mask_path.name} have form {mask.shape}. "
                "It is colored."
            )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"The sizes don't match:"
                f"image={image.shape[:2]}, mask={mask.shape[:2]}, "
                f"file={image_path.name}"
            )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # CrossEntropyLoss требует LongTensor для target
        mask = mask.long()

        return image, mask