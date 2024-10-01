# Data Loader
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class CustomDataset(Dataset):
    """
    Making dataset from Torch Datasets or manual datasets.
    """
    def __init__(self, dataset_name, pytorch_dataset=False, transform=None, type = 'train'):
        """
        Args:
            dataset_name (str): The name of the dataset (either Torch dataset name or folder name).
            pytorch_dataset (bool): Flag to check if it's a PyTorch dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataset_name = dataset_name
        self.pytorch_dataset = pytorch_dataset
        self.transform = transform
        self.type = type
        self.data_dir = os.getcwd() + "/Data/"
        if pytorch_dataset:
            self.dataset = self._load_torch_dataset()
        else:
            self.data_dir += dataset_name
            self.image_paths = self._load_manual_dataset()

    def _load_torch_dataset(self):
        """
        Load a PyTorch dataset.
        """
        # Example: Load CIFAR-10 or MNIST, etc.
        if self.dataset_name == "CIFAR10":
            return datasets.CIFAR10(root=self.data_dir, download=True, transform=self.transform)
        elif self.dataset_name == "MNIST":
            return datasets.MNIST(root=self.data_dir, download=True, transform=self.transform)
        elif self.dataset_name == "STL10":
            return datasets.STL10(root=self.data_dir, split = self.type, download=True, transform=self.transform)
        elif self.dataset_name == "OxfordPets":
            return datasets.OxfordIIITPet(root=self.data_dir, download=True, transform=self.transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported as a PyTorch dataset.")

    def _load_manual_dataset(self):
        """
        Load a manually structured dataset from a directory.
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        if not image_paths:
            raise ValueError(f"No images found in directory: {self.data_dir}")
        return image_paths

    def __len__(self):
        if self.pytorch_dataset:
            return len(self.dataset)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if self.pytorch_dataset:
            return self.dataset[idx]
        else:
            img_path = self.image_paths[idx]
            image = Image.open(img_path)#.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image , torch.empty(image.shape)

class NoisyDataset(torch.utils.data.Dataset):
    """
    Dataset that applies random noise to each image.
    """
    def __init__(self, dataset, noise_level= 5, forward_operator = lambda x: x):
        self.dataset = dataset
        self.noise_factor = noise_level/255
        self.A = forward_operator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # check if it has label
        if len(self.dataset[idx]) == 2:
            image, label = self.dataset[idx]
        else:
            image = self.dataset[idx]
            label = torch.empty(image.shape)
        # image, label = self.dataset[idx]
        # if not callable(self.A):
        #     if self.A.weight.device != image.device:
        #         self.A = self.A.to(image.device)
        #         self.A.weight.to(image.device)
        if type(self.A) == torch.nn.Module:
            if self.A.kernel.device != image.device:
                self.A = self.A.to(image.device)
                self.A.kernel.to(image.device)
        noisy_image = self.A(image) + self.noise_factor * torch.randn_like(image)
        noisy_image = torch.clamp(noisy_image, 0., 1.)  # Ensure the pixel values are between 0 and 1
        return noisy_image, label

class ReplaceableDataset(Dataset):
    """
    Dataset that starts with zero images but allows for batch updates with new data.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.images = [torch.zeros_like(image) for image, _ in dataset]  # Initialize with zeros
        self.labels = [label for _, label in dataset]  # Keep the original labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    # def update_batch(self, indices, new_images):
    #     """
    #     Update the images at specific indices with new nonzero images.

    #     :param indices: List or array of indices to update
    #     :param new_images: List or tensor of new images to replace the zeros
    #     """
    #     for i, idx in enumerate(indices):
    #         self.images[idx] = new_images[i]
    def update_data(self, idx, new_data):
        self.images[idx] = new_data
# For PyTorch dataset
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = CustomDataset("STL10", pytorch_dataset=True, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # For manual dataset in /Data/CustomImages/
# custom_dataset = CustomDataset("BSDS68", pytorch_dataset=False, transform=transform)
# custom_dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

