import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class NoisyFashionMNIST(Dataset):
    def __init__(self, root, train = True):
        """
        Creates a noisy dataset from the FashionMNIST 
        """
        
        transform = transforms.ToTensor()
        self.noise = AddGaussianNoise(0., 0.1)
        self.dataset = datasets.FashionMNIST(root, train = train, download=True, transform = transform)
        
    def __len__(self):
            return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        return self.noise(img), img