from utils import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from fastai.tabular import *
import pandas as pd
import matplotlib.pyplot as plt


class regressor(nn.Module):
    def __init__(self,no_of_models):
        super(regressor,self).__init__()
        self.regressor_ = nn.Linear(no_of_models , 1)
    def forward(self,x):
        x = self.regressor_(x)
        return x

class average_regressor(nn.Module):
    def __init__(self, no_of_models):
        super(average_regressor,self).__init__()
    def forward(self,x):
        return x.mean(1)

class max_regressor(nn.Module):
    def __init__(self, no_of_models):
        super(max_regressor,self).__init__()
    def forward(self,x):
        return x.max(1)[0]

class regressors(nn.Module):
    def __init__(self, no_of_labels,no_of_models , regressor = average_regressor):
        super(regressors , self).__init__()
        self.regressors = [regressor(no_of_models) for _  in range(no_of_labels)]

    def forward(self,x):
        output = []
        for regressor,batch in zip(self.regressors, x):
            output.append(regressor(batch).view(len(x[0])))
        output = torch.stack(output,1)
        return output

class ensemble_model(nn.Module):
    def __init__(self,models,regressors,no_of_labels=17):
        super(ensemble_model,self).__init__()
        self.models = models
        self.regressors = regressors
        self.no_of_labels = no_of_labels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batches_x=[[[] for j in range(len(x))] for i in range(self.no_of_labels)]
        for model in self.models:
            y_pred = model(x)
            y_pred = self.sigmoid(y_pred)
            for i in range(len(x)):
                for logit in range(self.no_of_labels):
                    batches_x[logit][i].append(y_pred[i][logit])

        for i in range(self.no_of_labels):
            for j in range(len(x)):
                batches_x[i][j] = torch.stack(batches_x[i][j])
            batches_x[i]=torch.stack(batches_x[i])
        batches_x=torch.stack(batches_x)
        output = self.regressors(batches_x)
        return output


class LossFunction():
    def __init__(self,thresh):
        self.logloss = nn.BCEWithLogitsLoss()
        self.f2Loss = self.smoothF2Loss

    def torch_f2_score(self,y_true, y_pred):
        return self.torch_fbeta_score(y_true, y_pred, 2)

    def torch_fbeta_score(self,y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2
        y_true = y_true.float()
        #print(y_pred)
        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))
        return torch.mean((precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))

    def smoothF2Loss(self, y_pred , y_true):
        return 1-self.torch_f2_score(y_true , torch.sigmoid(y_pred))

    def __call__(self,x,y):
        return self.f2Loss(x,y)+self.logloss(x,y)

class ensemble_model_trainer():
    def __init__(self , archs = None , no_of_labels = 17 , thresh = 0.2):
        self.archs = archs
        self.models = None
        self.no_of_models = len(self.archs)
        self.model_names = [arch.__name__ for arch in self.archs]
        self.no_of_labels = no_of_labels
        self.regressors = None
        self.regressorsLearners=None
        self.learners = None
        self.thresh_ = thresh
        self.lossFunction = LossFunction(self.thresh_)
        self.data = None 

    # functions of models -------------------------------------------------------------------------------------------------------------
    def build_learners(self,data):
        acc_02 = partial(accuracy_thresh, thresh=self.thresh_)
        f_score = partial(fbeta, thresh=self.thresh_,beta=2)
        model_dir = "./models"
        learner_kwargs = {"loss_func" : self.lossFunction}
        self.learners = [cnn_learner(data = data , base_arch = arch , pretrained = True ,metrics=[acc_02, f_score],model_dir = model_dir, **learner_kwargs) for arch in self.archs]
        self.data = data

    def lr_finder(self,**kwargs):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for name , learner in zip(self.model_names , self.learners):
            print(f"Model : {name}")
            learner.lr_find(**kwargs)

    def lr_finder_plot(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for name , learner in zip(self.model_names , self.learners):
            print(f"Model : {name}")
            learner.recorder.plot()


    def one_cycle_policy(self,lrs,epochs=1):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for name , learner , lr in zip(self.model_names , self.learners,lrs):
            print(f"Model : {name}")
            learner.fit_one_cycle(epochs,lr)

    def unfreeze(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for name , learner in zip(self.model_names , self.learners):
            print(f"Model : {name}")
            learner.unfreeze()
    


    def freeze(self):
        if(self.learners == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.learners:
            learner.freeze()


    def save_model(self):
        for learner , name in zip(self.learners,self.model_names):
            print(learner.save(name,return_path=True))

    def load_model(self):
        for learner , name in zip(self.learners,self.model_names):
            learner.load(name)


    # functions of regressors ----------------------------------------------------------------------------------------------------
    def one_cycle_policy_regressor(self, lrs , epochs = 1):
        if(self.regressors == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner , lr in zip(self.regressors , lrs):
            learner.fit_one_cycle(epochs,lr)

    def lr_finder_regressor(self,**kwargs):
        if(self.regressors == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.regressors:
            learner.lr_find(**kwargs)

    def lr_finder_plot_regeressor(self):
        if(self.regressors == None):
            sys.exit("first build learners with data using build_learners(self,data)")
        for learner in self.regressors:
            learner.recorder.plot()


    
    def collate(self,batch,no_of_labels = 17):
        batch = to_data(batch)
        batches_x=[[[] for j in range(len(batch))] for i in range(no_of_labels)]
        batches_y=[]
        for j,data in enumerate(batch):
            x,y = data
            for i,x1 in enumerate(x):
                batches_x[i][j]=torch.tensor(x1)
            batches_y.append(torch.tensor(y))
        batches_x = torch.stack([torch.stack(batches_x[i]) for i in range(no_of_labels)])
        batches_y = torch.stack(batches_y)
        return batches_x,batches_y

    def create_regressors(self , regressor = average_regressor , use_sigmoid = False):
        logits_dict = {}
        sigmoid = nn.Sigmoid()
        for i in range(self.no_of_labels):
            logits_dict[f"values_{i}"] = list()
            logits_dict[f"labels_{i}"] = list()

        with torch.no_grad():
            for data in self.learners[0].data.train_dl:
                x , y = data
                for batch_y in y:
                    for logits in range(self.no_of_labels):
                        logits_dict[f"labels_{logits}"].append(batch_y[logits].item())            

                temp = [[[] for _ in range(len(y))] for _   in range(self.no_of_labels) ]

                for learner in self.learners:
                    y_pred = learner.model(x)
                    if use_sigmoid:
                        y_pred = sigmoid(y_pred)
                    for i,batch_pred  in enumerate(y_pred):
                        for logit in range(self.no_of_labels):
                            temp[logit][i].append(batch_pred[logit].item())
                for logit in range(self.no_of_labels):
                    logits_dict[f"values_{logit}"]+=temp[logit]
                break

        data = pd.DataFrame(logits_dict)
        databunch = ItemList.from_df(data , cols = [f"values_{i}" for i in range(17)]).split_by_rand_pct().label_from_df(cols=[f"labels_{i}" for i in range(17)]).databunch(collate_fn = self.collate,device=torch.device('cpu') , bs = 2)
        f_score = partial(fbeta, thresh=self.thresh_,beta=2 ,  sigmoid = False)
        acc_02 = partial(accuracy_thresh, thresh=self.thresh_ ,  sigmoid = False)
        model_dir = "./models"
        regressors_model = regressors(self.no_of_labels ,  self.no_of_models , regressor)
        #regressors_model = regressors_model.to("cuda:0")
        self.regressors = Learner(data = databunch , model = regressors_model  , metrics=[acc_02,f_score],model_dir = model_dir , loss_func = self.lossFunction)


    def save_regressors(self):
        for i,regressor in enumerate(self.regressors):
            regressor.save(f"regressor_{i}",return_path=True)

    def load_regressors(self):
        for i,regressor in enumerate(self.regressors):
            regressor.laod(f"regressor_{i}")

    def create_model(self):
        models = [learner.model for learner in self.learners ]
        regressors = self.regressors.model
        return ensemble_model(models,regressors,self.no_of_labels)

if __name__=="__main__":
    ensemble_model.predict_image("./data/train-jpg/train_288.jpg")