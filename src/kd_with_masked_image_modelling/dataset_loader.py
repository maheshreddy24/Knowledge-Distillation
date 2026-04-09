from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from scipy.io import loadmat
from icecream import ic


class DatasetImagenet(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir/
            ILSVRC2012_img_train/
            ILSVRC2012_img_val/
            ILSVRC2012_validation_ground_truth.txt
            meta.mat
        """

        self.root_dir = root_dir
        self.split = split

        meta = loadmat(
            os.path.join(root_dir, "meta.mat"),
            struct_as_record=False,
            squeeze_me=True
        )

        synsets = meta["synsets"]

        # Keep only low-level synsets (ILSVRC2012_ID <= 1000)
        id_to_wnid = {}
        for s in synsets:
            ilsvrc_id = int(s.ILSVRC2012_ID)
            if ilsvrc_id <= 1000:
                id_to_wnid[ilsvrc_id - 1] = s.WNID  # zero-based

        # Sort by official ID order
        self.wnids = [id_to_wnid[i] for i in range(1000)]

        # Internal class mapping
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        # ic(len(self.class_to_idx))
        # ic(self.class_to_idx)


        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        self.samples = []

        if split == "train":
            train_dir = os.path.join(root_dir, "ILSVRC2012_img_train")

            for wnid in self.wnids:
                cls_path = os.path.join(train_dir, wnid)
                if not os.path.isdir(cls_path):
                    continue

                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    label = self.class_to_idx[wnid]
                    self.samples.append((img_path, label))
        else:
            val_dir = os.path.join(root_dir, "ILSVRC2012_img_val")
            gt_file = os.path.join(root_dir, "ILSVRC2012_validation_ground_truth.txt")

            with open(gt_file) as f:
                official_ids = [int(x.strip()) - 1 for x in f.readlines()]

            img_files = sorted(os.listdir(val_dir))

            for img_name, official_id in zip(img_files, official_ids):
                wnid = id_to_wnid[official_id]
                label = self.class_to_idx[wnid]

                img_path = os.path.join(val_dir, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
    


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split="trainval", transform=None):
        """
        root_dir: path to 'oxford-iiit-pet'
        split: 'trainval' or 'test'
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_file = os.path.join(root_dir, "annotations", f"{split}.txt")
        # self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_annotations()
        ic(len(self.samples))

    def _load_annotations(self):
        with open(self.annotations_file, "r") as f:
            for line in tqdm(f):
                parts = line.strip().split()
                image_name = parts[0] + ".jpg"
                label = int(parts[1]) - 1  # convert to 0-based index
                
                self.samples.append((image_name, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


class MaskedDataset(Dataset):
    def __init__(self, root_dir = '/media/system/ZERBUIS_EXT_STOR/dynamic_slam/imagenet/ILSVRC2012_img_', split = 'train'):
        super().__init__()
        self.root_dir = root_dir + split
        if split == 'train':  # train has subdirs
            sub_dirs = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir)]
            self.image_paths = []
            for s in sub_dirs:
                images = [os.path.join(s, p) for p in os.listdir(s)]
                self.image_paths.extend(images)
        
        elif split == 'val': 
            self.image_paths = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir)]
        else:
            raise ValueError("Invalid split")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]   # numpy array
        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        return img