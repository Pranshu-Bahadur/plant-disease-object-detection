from model import ImageClassifier
import torchvision
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader as Loader
from randaugment import RandAugment
import PIL
from util import ImageFilelistWithLabels, collate_fn

class Experiment(object):
    def __init__(self, config: dict):
        self.classifier = ImageClassifier(config)
        
    #@TODO Add normalized weight computation, use weighted random sampler.
    def _run(self, dataset, config: dict):
        split = self._preprocessing(dataset, config["list"], config["resolution"], True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(ds, self.classifier.bs, shuffle=True, num_workers=4) for ds in split]
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            train_acc, train_loss, val_acc, val_loss = self.classifier._run_epoch(loaders)

            print("Epoch: {} | Training Accuracy: {} | Training Loss: {} | Validation Accuracy: {} | Validation Loss: {}".format(self.classifier.curr_epoch, train_acc, train_loss, val_acc, val_loss))
            self.classifier.writer.add_scalar("Training Accuracy", train_acc, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",val_acc, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",train_loss, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",val_loss, self.classifier.curr_epoch)

            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print("Run Complete.")
        self.classifier._test(loaders[2])

    def _preprocessing(self, directory, order_list, resolution, train):
        """
        mean_sum = [0, 0, 0]
        transformations = [
        transforms.Resize([resolution, resolution], PIL.Image.ANTIALIAS),
        transforms.ToTensor()
        ]
        nimages = 0
        mean_sum = 0.
        std_sum = 0.
        transformations = transforms.Compose(transformations)
        ds = torchvision.datasets.ImageFolder(root=directory, transform=transformations)
        loader = Loader(ds, self.classifier.bs) 
        for batch, _ in loader:
            batch = batch.view(batch.size(0), batch.size(1), -1)
            nimages += batch.size(0)
            mean_sum += batch.mean(2).sum(0)
            std_sum += batch.std(2).sum(0)            
        mean_sum /= nimages
        std_sum  /= nimages
        """
        #mean_sum = [0.5675, 0.5675, 0.5675]
        #std_sum = [0.2157, 0.2157, 0.2157]
        transformations = [
            transforms.Resize([resolution, resolution], PIL.Image.ANTIALIAS),
            #RandAugment(),
            #RandAugment(),
            #RandAugment(),
            #RandAugment(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean_sum, std=std_sum)
        ]
        transformations = transforms.Compose(transformations)
        dataSetFolder = torchvision.datasets.ImageFolder(root=directory, transform=transformations)##ImageFilelistWithLabels(root=directory, flist=order_list, transform=transformations)#torchvision.datasets.ImageFolder(root=directory, transform=transformations)#
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize)//2
            return torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
        return dataSetFolder