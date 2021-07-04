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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score



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
        self.counter = 3
        self.final_epoch = config["epochs"]
        self.nc = config["num_classes"]
        self.COUNT = 1
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
                      "SGDAGC": SGD_AGC(model_params.parameters(), lr=lr, clipping=0.01, weight_decay=1e-05, nesterov=True, momentum=0.9)#AGC(model_params.parameters(), torch.optim.SGD(model_params.parameters(), lr, momentum=0.9,weight_decay=1e-5, nesterov=True), model=model_params, ignore_agc=['head'], clipping=0.01)#SGD_AGC(model_params.parameters(), lr=lr, clipping=0.01, weight_decay=2e-05, nesterov=True, momentum=0.9) #,###
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
        #if self.curr_epoch < 2:
        #transforms = [
        #    torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomRotation()
            #    torchvision.transforms.ToPILImage(),
            #    torchvision.transforms.ToTensor()
        #]
            #if self.counter-1 > 0:
            #    transforms.insert(1, torchvision.transforms.Resize(self.resolution - 32*(self.counter-1)))
                
            #for i in range(4-self.counter):
            #    if i < 2:
            #        transforms.insert(1, RandAugment())
        #transforms = torchvision.transforms.Compose(transforms)

        #split[0] = list(map(lambda x: (torch.stack(list(map(lambda img: transforms(img), x[0]))),x[1]) ,list(split[0])))
        #if self.curr_epoch==2:
        #    split[0] = list(map(lambda x: (torchvision.transforms.functional.resize(x[0], self.resolution),x[1]) ,split[0]))
        #print(len(split[0]), print(split[0][0].size()))
        auc_train, f1_train, recall_train, precision_train, train_acc, train_loss = self._train_or_eval(split[0], True)
        self.model.eval()
        with torch.no_grad():
            auc_val, f1_val, recall_val, precision_val,  val_acc, val_loss = self._train_or_eval(split[1], False)
        print(auc_train, auc_val, f1_train, f1_val, recall_train, recall_val, precision_train, precision_val, train_acc, train_loss, val_acc, val_loss)
        self.curr_epoch += 1
        return auc_train, auc_val, f1_train, f1_val, recall_train, recall_val, precision_train, precision_val, train_acc, train_loss, val_acc, val_loss

    def _train_or_eval(self, loader, train):
        running_loss, correct, total, iterations = 0, 0, 0, 0
        classes = []
        preds_cfm = []
        auc, f1, recall, precision = 0, 0, 0, 0
        if train: #and (self.curr_epoch+1)%5==0:
            self.counter = max(self.counter - 1, 0)
            print("Changing resolution...")
            #self.optimizer.clipping = self.optimizer.clipping*2  if self.bs>128 or self.curr_epoch==self.final_epoch-2 else 0.32
            #self.bs = self.bs//2 if self.bs>64 else 64
        idx = 0
        for data in enumerate(loader): #loader if train else 
            self.optimizer.zero_grad()
            x, y = data[1] #data if train else 
            if train:
                if self.curr_epoch <= 2:
                    x = torchvision.transforms.functional.resize(x, self.resolution - (32*(self.counter)))
                x = x.cuda()
                x = list(map(lambda img: torchvision.transforms.functional.to_tensor(self.RA_Helper(torchvision.transforms.functional.to_pil_image(img), self.counter, 0, idx)), x))
                x = torch.stack(x)
                shuffle_seed = torch.randperm(x.size(0))
                x = x[shuffle_seed]
                y = y[shuffle_seed]#torch.cat(y_, dim=0)
                total += y.size(0)

                print(x.size())
                    #torchvision.utils.save_image(x[y==i][0], "/content/Post_RA_{}_{}.png".format(self.resolution - 32*self.counter, i))
                if True:
                    def closure():
                        self.optimizer.zero_grad()
                        preds = self.model(x.cuda())
                        loss = self.criterion(preds, y.cuda())
                        loss.backward()
                        return loss
                    preds = self.model(x.cuda())
                    #self.optimizer.zero_grad()
                    #loss = self.optimizer.step(closure)
                    loss = self.criterion(preds, y.cuda())
                    loss.backward()
                    self.optimizer.step()

                #else:
                    if self.curr_epoch==self.final_epoch-1:# and x[y_.cpu()!=y.cpu()].size(0) > 0:
                    #CFM
                        #inputs = x.to('cpu')
                    #inputs = torchvision.transforms.functional.adjust_contrast(inputs, 1.25)

                        classes.append(y.to('cpu'))
                        #op = self.model(inputs)
                        _, p = torch.max(preds, 1)
                        preds_cfm.append(p)
                    #self.optimizer.zero_grad()
                    #preds = self.model(x.cuda())
                    #preds = torch.nn.functional.dropout2d(preds,0.4)
                    #
                self.scheduler.step()
                probs = nn.functional.softmax(preds, 1)
                y_ = torch.argmax(probs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
                auc += roc_auc_score(y.cpu(), y_.cpu())
                f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
                precision += precision_score(y.cpu(), y_.cpu(), average='micro')
                recall += recall_score(y.cpu(), y_.cpu(), average='micro')
                print(idx, (correct/total)*100, loss.cpu().item())
                idx += 1

            else:
                
                if self.curr_epoch>=self.final_epoch-1:# and x[y_.cpu()!=y.cpu()].size(0) > 0:
                    #CFM
                    inputs = x.to('cpu')
                    #inputs = torchvision.transforms.functional.adjust_contrast(inputs, 1.25)

                    classes.append(y.to('cpu'))
                    op = self.model(inputs)
                    _, p = torch.max(op, 1)
                    preds_cfm.append(p)
                
                #self.writer.add_graph(self.model.cuda(), x.cuda())
                shuffle_seed = torch.randperm(x.size(0))
                x = x[shuffle_seed]
                y = y[shuffle_seed]
                total += y.size(0)

                preds = self.model(x.cuda())
                loss = self.criterion(preds, y.cuda())
                probs = nn.functional.softmax(preds, 1)
                y_ = torch.argmax(probs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
                auc += roc_auc_score(y.cpu(), y_.cpu())
                f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
                precision += precision_score(y.cpu(), y_.cpu(), average='micro')
                recall += recall_score(y.cpu(), y_.cpu(), average='micro')

            running_loss += loss.cpu().item()
            iterations += 1
            del x, y
            torch.cuda.empty_cache()
        
        if self.curr_epoch>=self.final_epoch-1:
            c_l = classes.pop()
            p_l = preds_cfm.pop()
            classes = torch.stack(classes, dim=0).view(-1)
            preds_cfm = torch.stack(preds_cfm, dim=0).view(-1)
            cfm = confusion_matrix(classes.cpu().detach().numpy(),preds_cfm.cpu().detach().numpy())
            cfm += confusion_matrix(c_l.cpu().detach().numpy(),p_l.cpu().detach().numpy())
            classes=[str(i) for i in range(self.nc)]
            df_cfm=pd.DataFrame(cfm.astype(int), index=classes, columns=classes)
            plt.figure(figsize=(5,5))
            cfm_plot=sn.heatmap(df_cfm.astype(int), annot=True, fmt=".1f")
            cfm_plot.figure.savefig('/home/fraulty/ws/content_1/kaggle_tb_cfmtbx_{}_{}.png'.format("train" if train else "validation", 0 if train else self.COUNT))
            if not train:
                self.COUNT +=1
        return float(auc/float(iterations))*100, float(f1/float(iterations))*100, float(recall/float(iterations))*100, float(precision/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)

    def _test(self, loader):
        self.model.eval()
        with torch.no_grad():
            auc_test, f1, recall, precision, test_acc, test_loss = self._train_or_eval(loader, False)
        print("Test metrics:", auc_test, f1, recall, precision, test_acc, test_loss)
        return auc_test, f1, recall, precision, test_acc, test_loss

    #@TODO
    def _add_detector(self):
        pass

    def _update_tensorboard(self):
        pass

    def RA_Helper(self, x, i, j, idx):
        """
        if idx==0 and self.curr_epoch<2:
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(x), "/home/fraulty/ws/content/{}_Before_RA_{}_{}.png".format(self.curr_epoch+1,self.resolution - 32*self.counter, j))
        """
        #print("test")
        for k in range(3 - i):
            if k<2:
                x = RandAugment()(x)
        return x