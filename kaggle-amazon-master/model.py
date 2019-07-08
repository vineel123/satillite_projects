from utils import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys

class regressor(nn.Module):
    def __init__(self,no_of_models):
        self.regressor = nn.Linear(no_of_models , 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(x)
        x = self.regressor(x)
        x = self.sigmoid(x)
        return x


class LossFunction():
    def __init__(self,thresh):
        self.logloss = nn.BCEWithLogitsLoss()
        self.f2Loss = self.smoothF2Loss

    def torch_f2_score(self,y_true, y_pred):
        return self.torch_fbeta_score(y_true, y_pred, 2)

    def torch_fbeta_score(self,y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2
        y_true = y_true.float()
        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))
        return torch.mean((precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))

    def smoothF2Loss(self, y_pred , y_true):
        return 1-self.torch_f2_score(y_true , torch.sigmoid(y_pred))

    def __call__(self,x,y):
        return self.f2Loss(x,y)+self.logloss(x,y)



class ensemble_model():
    def __init__(self , archs = None , no_of_labels = 17 , thresh = 0.2):
        self.archs = [ models.resnet34, models.resnet50]
        self.models = None
        self.no_of_models = len(self.archs)
        self.model_names = [arch.__name__ for arch in self.archs]
        self.no_of_labels = no_of_labels
        self.regressors = [ for _ in range(self.no_of_labels)]
        self.regressorsLearners=None
        self.learners = None
        self.thresh_ = thresh
        self.lossFunction = LossFunction(self.thresh_)
        self.data = None 


    def build_learners(self,data):
        acc_02 = partial(accuracy_thresh, thresh=self.thresh_)
        f_score = partial(fbeta, thresh=self.thresh_,beta=2)
        learner_kwargs = {"loss_func" : self.lossFunction}
        self.learners = [cnn_learner(data = data , base_arch = arch , pretrained = True ,metrics=[acc_02, f_score],**learner_kwargs) for arch in self.archs]
        self.data = data

    def lr_finder(self,**kwargs):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.learners:
            learner.lr_find(**kwargs)

    def lr_finder_plot(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.learners:
            learner.recorder.plot()


    def one_cycle_policy(self,lrs,epochs=1):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner , lr in zip(self.learners , lrs):
            learner.fit_one_cycle(epochs,lr)

    def unfreeze(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.learners:
            learner.unfreeze()

    def freeze(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.learners:
            learner.freeze()
    
    def create_regression_dataset(self):
        logits_dict = dict()

        for data in self.data.train_dl:
            x , y = data
            for arch in self.model_names

         


