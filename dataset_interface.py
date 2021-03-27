import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader


class DataInterface(object):
    def __init__(self, root_path, dataset: str, batch_size: int, num_works: int):
        self.__train_key = "train"
        self.__test_key = "val"
        transform_ = self.__mnist_transform()
        self.trainset, self.testset = self._train_and_test_sets(vision_dataset=MNIST, root=root_path,
                                                                transform=transform_)
        self.labels = None
        if dataset == "FashionMNIST":
            self.trainset, self.testset = self._train_and_test_sets(vision_dataset=MNIST, root=root_path,
                                                                    transform=transform_)
        elif dataset == "CIFAR10":
            transform_ = self.__cifar_transform()
            self.trainset, self.testset = self._train_and_test_sets(vision_dataset=CIFAR10, root=root_path,
                                                                    transform=transform_)
            self.labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset == "CIFAR100":
            transform_ = self.__cifar_transform()
            self.trainset, self.testset = self._train_and_test_sets(vision_dataset=CIFAR100, root=root_path,
                                                                    transform=transform_)
            self.labels = [
                "beaver", "dolphin", "otter", "seal", "whale",
                "aquarium fish", "flatfish", "ray", "shark", "trout",
                "orchids", "poppies", "roses", "sunflowers", "tulips",
                "bottles", "bowls", "cans", "cups", "plates",
                "apples", "mushrooms", "oranges", "pears", "sweet peppers",
                "clock", "computer keyboard", "lamp", "telephone", "television",
                "bed", "chair", "couch", "table", "wardrobe",
                "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                "bear", "leopard", "lion", "tiger", "wolf",
                "bridge", "castle", "house", "road", "skyscraper",
                "cloud", "forest", "mountain", "plain", "sea",
                "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                "fox", "porcupine", "possum", "raccoon", "skunk",
                "crab", "lobster", "snail", "spider", "worm",
                "baby", "boy", "girl", "man", "woman",
                "crocodile", "dinosaur", "lizard", "snake", "turtle",
                "hamster", "mouse", "rabbit", "shrew", "squirrel",
                "maple", "oak", "palm", "pine", "willow",
                "bicycle", "bus", "motorcycle", "pickup truck", "train",
                "lawn-mower", "rocket", "streetcar", "tank", "tractor"
            ]
            pass
        elif dataset == "MNIST":
            pass
        else:
            print('Because Vision dataset was not found, Use MNIST')
            pass

        self._train_dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=False, num_workers=num_works)
        self.val_dataloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_works)

    @property
    def train_key(self):
        return self.__train_key

    @property
    def test_key(self):
        return self.__test_key

    def dataloader_dict(self):
        return {self.__train_key: self._train_dataloader, self.__test_key: self.val_dataloader}

    def __mnist_transform(self):
        return transforms.Compose(
            [torchvision.transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def __cifar_transform(self):
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def _train_and_test_sets(self, vision_dataset, root, transform):
        return vision_dataset(root=root, train=True, download=True, transform=transform), \
               vision_dataset(root=root, train=False, download=True, transform=transform),

    pass
