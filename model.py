import torchvision
import torch
from torch import functional as F
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
import timm
from modules import Net
from sam import SAMSGD
from nfnets import SGD_AGC, AGC
import PIL
from randaugment import RandAugment
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd



class ImageClassifier(object):
    def __init__(self, config : dict):
        self.model = self._create_model(config["library"], config["model_name"], config["pretrained"], config["num_classes"])
        if True:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = self._create_scheduler(config["scheduler_name"], self.optimizer)
            self.criterion = self._create_criterion(config["criterion_name"])
        if config["checkpoint"] != "":
            self.model = nn.DataParallel(self.model).cuda()
            self._load(config["checkpoint"])
        self.curr_epoch = config["curr_epoch"]
        self.name = "{}-{}-{}-{}".format(config["model_name"], config["resolution"], config["batch_size"], config["learning_rate"])
        self.bs = config["batch_size"]
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.resolution = config["resolution"]
        self.counter = 4
        self.final_epoch = config["epochs"]
        print("Generated model: {}".format(self.name))

        if config["train"] and config["checkpoint"] == "":
           self.model = nn.DataParallel(self.model).cuda()

        
    def _create_model(self, library, name, pretrained, num_classes):
        if library == "timm":
            return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        else:
            return Net(num_classes)

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr,weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "SAMSGD": SAMSGD(model_params.parameters(), lr, momentum=0.9,weight_decay=1e-5),
                      "SGDAGC": AGC(model_params.parameters(), torch.optim.SGD(model_params.parameters(), lr, momentum=0.9,weight_decay=1e-5, nesterov=True), model=model_params, ignore_agc=['head'], clipping=0.01)#SGD_AGC(model_params.parameters(), lr=lr, clipping=0.01, weight_decay=2e-05, nesterov=True, momentum=0.9) #,###
        }
        return optim_dict[name]
    
    def _create_scheduler(self, name, optimizer):
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.4, gamma=0.97),
            "CosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)
        }
        return scheduler_dict[name]

    def _create_criterion(self, name):
        loss_dict = {"CCE": nn.CrossEntropyLoss().cuda(),#
                     "MML": nn.MultiMarginLoss().cuda(),
                     "MSE": nn.MSELoss().cuda(),
                     "BCE": nn.BCELoss().cuda()
                     }
        return loss_dict[name]

    def _load(self, directory):
        print("loading previously trained model...")
        self.model.load_state_dict(torch.load(directory))

    def _save(self, directory, name):
        print("Saving trained {}...".format(name))
        torch.save(self.model.state_dict(), "{}/./{}.pth".format(directory, name))

    def _run_epoch(self, split):
        self.model.train()
        train_acc, train_loss = self._train_or_eval(split[0], True)
        self.model.eval()
        with torch.no_grad():
            val_acc, val_loss = self._train_or_eval(split[1], False)
        print(train_acc, train_loss, val_acc, val_loss)
        self.curr_epoch += 1
        return train_acc, train_loss, val_acc, val_loss

    def _train_or_eval(self, loader, train):
        running_loss, correct, total, iterations = 0, 0, 0, 0
        classes = []
        preds_cfm = []
        if train: #and (self.curr_epoch+1)%5==0:
            self.counter = max(self.counter - 1, 0)
            print("Changing resolution...")
            #self.optimizer.clipping = self.optimizer.clipping*2  if self.bs>128 or self.curr_epoch==self.final_epoch-2 else 0.32
            self.bs = self.bs//2 if self.bs>32 else 32
        for idx, data in enumerate(loader):
            self.optimizer.zero_grad()
            x, y = data
            total += y.size(0)
            if train:
                x_, y_ = [], []
                for i in range(3):
                    if x[y==i].size(0) > 0:
                        for _ in range(2):
                            x_.append(torch.stack([torchvision.transforms.ToTensor()(self.RA_Helper(torchvision.transforms.Resize(self.resolution - 32*self.counter,interpolation=PIL.Image.ANTIALIAS)(torchvision.transforms.ToPILImage()(img)), self.counter, i, idx)) for img in x[y==i]]))
                            y_.append(y[y==i])
                x_ = torch.cat(x_, dim=0)
                shuffle_seed = torch.randperm(x_.size(0))
                x = x_[shuffle_seed]
                y = torc.cat(y_, dim=0)[shuffle_seed]
                print(x.size())
                    #torchvision.utils.save_image(x[y==i][0], "/content/Post_RA_{}_{}.png".format(self.resolution - 32*self.counter, i))
                if type(self.optimizer) == SAMSGD or type(self.optimizer) == AGC:
                    def closure():
                        self.optimizer.zero_grad()
                        preds = self.model(x.cuda())
                        loss = self.criterion(preds, y.cuda())
                        loss.backward()
                        return loss
                    preds = self.model(x.cuda())
                    #self.optimizer.zero_grad()
                    loss = self.optimizer.step(closure)

                #else:
                    if self.curr_epoch==self.final_epoch-1:# and x[y_.cpu()!=y.cpu()].size(0) > 0:
                    #CFM
                        inputs = x.to('cpu')
                    #inputs = torchvision.transforms.functional.adjust_contrast(inputs, 1.25)

                        classes.append(y.to('cpu'))
                        #op = self.model(inputs)
                        _, p = torch.max(preds, 1)
                        preds_cfm.append(p)
                    #self.optimizer.zero_grad()
                    #preds = self.model(x.cuda())
                    #preds = torch.nn.functional.dropout2d(preds,0.4)
                    #loss = self.criterion(preds, y.cuda())
                    #loss.backward()
                    #self.optimizer.step()
                self.scheduler.step()
                probs = nn.functional.softmax(preds, 1)
                y_ = torch.argmax(probs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
                print(idx, (correct/total)*100, loss.cpu().item())
            else:
                if self.curr_epoch==self.final_epoch-1:# and x[y_.cpu()!=y.cpu()].size(0) > 0:
                    #CFM
                    inputs = x.to('cpu')
                    #inputs = torchvision.transforms.functional.adjust_contrast(inputs, 1.25)

                    classes.append(y.to('cpu'))
                    op = self.model(inputs)
                    _, p = torch.max(op, 1)
                    preds_cfm.append(p)
                #self.writer.add_graph(self.model.cuda(), x.cuda())
                x = x[torch.randperm(x.size(0))]
                preds = self.model(x.cuda())
                loss = self.criterion(preds, y.cuda())
                probs = nn.functional.softmax(preds, 1)
                y_ = torch.argmax(probs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
            running_loss += loss.cpu().item()
            iterations += 1
            del x, y
            torch.cuda.empty_cache()
        if self.curr_epoch==self.final_epoch-1:
            classes.pop()
            preds.pop()
            classes = torch.stack(classes, dim=0).view(-1)
            preds_cfm = torch.stack(preds_cfm, dim=0).view(-1)
            print(classes.size(), preds.size())
            cfm = confusion_matrix(classes.cpu().numpy(),preds.cpu().numpy())
            classes=["0","1","2"]
            df_cfm=pd.DataFrame(cfm.astype(int), index=classes, columns=classes)
            plt.figure(figsize=(5,5))
            cfm_plot=sn.heatmap(df_cfm.astype(int), annot=True, fmt=".1f")
            cfm_plot.figure.savefig('/home/fraulty/ws/RP_TBX11K/content/cfmtbx_{}.png'.format("train" if train else "validation"))

        return float(correct/float(total))*100, float(running_loss/iterations)

    def _test(self, loader):
        self.model.eval()
        with torch.no_grad():
            test_acc, test_loss = self._train_or_eval(loader, False)
        print(test_acc, test_loss)
        return test_acc, test_loss

    #@TODO
    def _add_detector(self):
        pass

    def _update_tensorboard(self):
        pass

    def RA_Helper(self, x, i, j, idx):
        if idx==0:
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(x), "/home/fraulty/ws/RP_TBX11K/content/{}_Before_RA_{}_{}.png".format(self.curr_epoch+1,self.resolution - 32*self.counter, j))
        for _ in range(4 - i):
            x = RandAugment()(x)
        if idx==0:
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(x), "/home/fraulty/ws/RP_TBX11K/content/{}_After_RA_{}_{}.png".format(self.curr_epoch+1,self.resolution - 32*self.counter, j))
        return x