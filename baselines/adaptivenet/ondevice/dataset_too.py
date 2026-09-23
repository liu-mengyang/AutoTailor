import torchvision
import torchvision.transforms as transforms


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
dataset_eval = torchvision.datasets.ImageFolder(root='ValForDevice/ValForDevice3k',transform=data_transform)

print(len(dataset_eval))