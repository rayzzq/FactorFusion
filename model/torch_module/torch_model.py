import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             multilabel_confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .torch_net import MlpNet
from .torch_loss import LOSS, hamming_loss
from ..base_model import BaseModel

import warnings
warnings.simplefilter(action = "ignore", category = (FutureWarning, UserWarning))

class TorchModel(BaseModel):
    def __init__(self, params, net_name="mlp", scaler=None):
        self.params = params
        self.net_name = net_name

        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = RobustScaler()

        self.Xcols = self.params["Xcols"]
        self.regYcols = self.params["regYcols"]
        self.clsYcols = self.params["clsYcols"]

        net_params = self.params["net_params"]
        net_params["input_dim"] = len(self.Xcols)
        net_params["output_dim"] = len(self.regYcols) + len(self.clsYcols)

        if self.net_name == "mlp":
            self.model = MlpNet(**net_params)
        else:
            raise NotImplementedError("net {} is not implemented".format(self.net_name))
        
        self.is_fitted = False

    def fit(self, data, eval_data=None):
        Xcols = self.Xcols
        regYcols = self.regYcols
        clsYcols = self.clsYcols

        # fit scaler
        print("Step1: fiting scaler...")
        self.scaler.fit(data[Xcols])

        # create dataloader
        print("Step2: creating dataloader...")
        if eval_data is not None:
            train_loader = self.create_dataloader(data[Xcols].values, data[regYcols].values, data[clsYcols].values, shuffle=True)
            eval_loader = self.create_dataloader(eval_data[Xcols].values, eval_data[regYcols].values, eval_data[clsYcols].values, shuffle=False)
        else:
            train_data, eval_data = train_test_split(data, test_size=0.2)
            train_loader = self.create_dataloader(train_data[Xcols].values, train_data[regYcols].values, train_data[clsYcols].values, shuffle=True)
            eval_loader = self.create_dataloader(eval_data[Xcols].values, eval_data[regYcols].values, eval_data[clsYcols].values, shuffle=False)

        # set optimizer
        self.params.length_of_dataloader = len(train_loader)
        optimizer, scheduler = self._set_optimizer()

        # training model
        print("Step3: Start Training...")
        print("-" * 30, "Configuation", "-" * 30)
        for k, v in self.params.items():
            print('{:>30}'.format(str(k)), '-' * 10 ,'{:<30}'.format(str(v)))

        self.model = self.model.to(self.params.device)
        self.model.train()

        len_reg = len(regYcols)

        global_steps = 0
        accumulate_grad_batches = self.params.accumulate_grad_batches
        best_eval_loss = 1e10
        early_stop = 0

        # print("-" * 100)
        # print(len(train_loader))
        # print("-" * 100)
        
        for epoch in range(self.params.max_epochs):
            train_loss = 0
            for idx, (X, regY, clsY) in enumerate(train_loader):

                X = X.to(self.params.device)
                regY = regY.to(self.params.device)
                clsY = clsY.to(self.params.device)

                preds = self.model(X)
                regY_pred = preds[:, :len_reg]
                clsY_pred = preds[:, len_reg:]
                reg_loss = LOSS[self.params.reg_loss](regY_pred, regY)
                cls_loss = LOSS[self.params.cls_loss](clsY_pred, clsY)
                loss = 0.8 * reg_loss + 0.2 * cls_loss
                loss.backward()
                train_loss += loss.item()
                global_steps += 1
                
                if (idx + 1) % accumulate_grad_batches == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    

                if (global_steps + 1) % self.params.log_steps == 0:
                    print("epoch:{:<4} | step:{:<5} | loss:{:<10}".format(epoch, global_steps, train_loss / (idx + 1)))

            print("Epoch end, Start evaluation...")
            res = self.evaluate_model(eval_loader)
            if res is not None:
                print("Epoch:{:<4} | valid_loss: {:<5} | valid_reg_loss: {:<10} | valid_cls_hamming: {:<10}".format(epoch, res["loss"], res["reg_loss"], res["hamming"]))
                print("")
                
            if res["loss"] > best_eval_loss:
                best_eval_loss = res["loss"]
                save_name = self.params.save_model_path + "epoch:{}_step:_evalloss:{}_netname:{}.pt".format(epoch + 1, res["loss"], self.net_name)
                self.save_model(save_name)
                early_stop = 0
                print("Best Model saved.")
            else:
                if self.params.early_stop > 0:
                    early_stop += 1
                    if early_stop > self.params.early_stop:
                        print("Early stop.")
                        break
                    
            self.is_fitted = True

    @torch.no_grad()
    def evaluate_model(self, eval_loader):
        len_reg = len(self.regYcols)
        self.model.eval()
        all_preds = []
        regYs = []
        clsYs = []

        for idx, (X, regY, clsY) in enumerate(eval_loader):

            X = X.to(self.params.device)
            regY = regY.to(self.params.device)
            clsY = clsY.to(self.params.device)

            pred = self.model(X)
            all_preds.append(pred.cpu())
            regYs.append(regY.cpu())
            clsYs.append(clsY.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        regYs = torch.cat(regYs, dim=0)
        clsYs = torch.cat(clsYs, dim=0)

        reg_loss = LOSS[self.params.reg_loss](all_preds[:, :len_reg], regYs)
        cls_loss = LOSS[self.params.cls_loss](all_preds[:, len_reg:], clsYs)
        loss = 0.8 * reg_loss + 0.2 * cls_loss

        
        # print(clsYs.shape)
        # print(all_preds[:, len_reg:].shape)
        
        all_preds[:, len_reg:] = torch.sigmoid(all_preds[:, len_reg:])
        all_preds[:, len_reg:] = torch.where(all_preds[:, len_reg:] > 0.5, 1, 0)
        
        # print(all_preds[:, len_reg:])
        sample_weights = (clsYs + 0.5).sum(dim=1) / (clsYs + 0.5).sum()
        hamming = hamming_loss(clsYs.numpy(),
                               all_preds[:, len_reg:].numpy(),
                               sample_weight=sample_weights)

        res = {}
        res['reg_loss'] = reg_loss.item()
        res['cls_loss'] = cls_loss.item()
        res['loss'] = loss.item()
        res['hamming'] = hamming

        return res

    @torch.no_grad()
    def predict(self, data, batch_size=10240, device="cpu", return_type="pandas"):
        if not self.is_fitted:
            raise  Exception("Model not fitted, can't predict")
        
        device = torch.device(device)
        if isinstance(data, pd.DataFrame):
            pred_X = data[self.Xcols].values
        elif isinstance(data, np.ndarray):
            assert data.shape[1] == len(self.Xcols), "data.shape[1] must be equal to len(self.Xcols)"
            pred_X = data
        else:
            raise ValueError("data must be pandas.DataFrame or numpy.ndarray")

        pred_X_copy = self.scaler.transform(pred_X.copy())
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(pred_X_copy).float())
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=3,
                                                 pin_memory=True,)
        self.model = self.model.to(device)
        self.model.eval()

        all_preds = []
        for idx, (X,) in enumerate(dataloader):
            X = X.to(device)
            pred = self.model(X)
            all_preds.append(pred.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        len_reg = len(self.regYcols)

        all_preds[:, len_reg:] = torch.sigmoid(all_preds[:, len_reg:])
        all_preds[:, len_reg:] = torch.where(all_preds[:, len_reg:] > 0.5, 1, 0)

        Ycols = self.regYcols + self.clsYcols

        if return_type == "numpy":
            return all_preds.numpy()
        elif return_type == "pandas":
            return pd.DataFrame(all_preds.numpy(), columns=Ycols)
        else:
            raise ValueError("return_type must be 'numpy' or 'pandas'")

    def save_model(self, path):
        params = self.params
        model = self.model
        scaler = self.scaler
        net_name = self.net_name
        Xcols = self.Xcols
        regYcols = self.regYcols
        clsYcols = self.clsYcols
        saved_dict = {"params": params,
                      "model": model,
                      "scaler": scaler,
                      "net_name": net_name,
                      "Xcols": Xcols,
                      "regYcols": regYcols,
                      "clsYcols": clsYcols}
        torch.save(saved_dict, path)

    @classmethod
    def load_model(cls, path):
        saved_dict = torch.load(path)
        params = saved_dict["params"]
        model = saved_dict["model"]
        net_name = saved_dict["net_name"]
        scaler = saved_dict["scaler"]
        Xcols = saved_dict["Xcols"]
        regYcols = saved_dict["regYcols"]
        clsYcols = saved_dict["clsYcols"]
        obj = cls(params, net_name, scaler)
        obj._set_cols(Xcols, regYcols, clsYcols)
        obj.model = model
        obj.is_fitted = True
        return obj

    def create_dataloader(self, X, regY, clsY, shuffle=True):
        X_copy = self.scaler.transform(X)
        X_copy = torch.from_numpy(X_copy).float()
        regY_copy = torch.from_numpy(regY).float()
        clsY_copy = torch.from_numpy(clsY).float()
        dataset = torch.utils.data.TensorDataset(X_copy, regY_copy, clsY_copy)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.params.batch_size,
                                             shuffle=shuffle,
                                             num_workers=5,
                                             pin_memory=True,)
        return loader

    def _set_optimizer(self,):
        optimizer = self._create_optimizer(self.model, self.params.lr, self.params.weight_decay)
        self.optimizer = optimizer

        if self.params.max_epochs == -1:
            t_total = self.params.max_steps // self.params.accumulate_grad_batches
        else:
            t_total = self.params.length_of_dataloader // self.params.accumulate_grad_batches * self.params.max_epochs

        if self.params.warmup_steps != -1:
            warmup_steps = self.params.warmup_steps
        else:
            warmup_steps = int(self.params.warmup_proportion * t_total)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        return optimizer, scheduler

    def _set_cols(self, Xcols, regYcols, clsYcols):
        self.Xcols = Xcols
        self.regYcols = regYcols
        self.clsYcols = clsYcols

    @staticmethod
    def _create_optimizer(model, lr, weight_decay, custom_lr=None):
        no_decay = 'bias|norm'
        params = defaultdict(list)
        custom_lr = custom_lr or dict()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            in_custom = False
            for custom_name, _ in custom_lr.items():
                if custom_name in name:
                    if re.search(no_decay, name.lower()):
                        params[custom_name].append(param)
                    else:
                        params[custom_name + '_decay'].append(param)
                    in_custom = True
                    break
            if not in_custom:
                if re.search(no_decay, name):
                    params['normal'].append(param)
                else:
                    params['normal_decay'].append(param)

        optimizer_grouped_parameters = []
        for k, v in params.items():
            param_lr = custom_lr.get(k.split('_')[0], lr)
            decay = weight_decay if 'decay' in k else 0.0
            optimizer_grouped_parameters.append({'params': v, 'weight_decay': decay, 'lr': param_lr}, )

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        return optimizer
