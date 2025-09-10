
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ShapesDataset(Dataset):
    """Custom dataset for the geometric shapes.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {label: i for i, label in enumerate(self.df['label'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path).convert("L") # Convert to grayscale
        label_name = self.df.iloc[idx, 1]
        label = self.label_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(train_csv, test_csv, batch_size=32):
    """
    Creates and returns the training and testing dataloaders.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the datasets
    train_dataset = ShapesDataset(csv_file=train_csv, transform=transform)
    test_dataset = ShapesDataset(csv_file=test_csv, transform=transform)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.label_map
