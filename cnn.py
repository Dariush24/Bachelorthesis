import pickle
import shutil
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime

from matplotlib import image as mpimg

from datasets import *
from utils import *
from layers import *


class PCNet(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate,loss_fn, loss_fn_deriv,device='cpu',numerical_check=False):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.device = device
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.numerical_check = numerical_check
    if self.numerical_check:
      print("Numerical Check Activated!")
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        if self.numerical_check:
            true_weight_grad = l.get_true_weight_grad().clone()
        dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
        true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
        diff = torch.sum((dW -true_dW)**2).item()
        weight_diffs.append(diff)
        if print_weight_grads:
          print("weight grads PC: ", i)
          print("dW PC: ", dW*2)
          print("true diffs PC: ", true_dW * 2)
          if self.numerical_check:
            print("true weights ", true_weight_grad)
    return weight_diffs


  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp,label,n_inference_steps=None):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i,l in enumerate(self.layers):
        #initialize mus with forward predictions
        self.mus[i+1] = l.forward(self.mus[i])
        self.outs[i+1] = self.mus[i+1].clone()
      self.mus[-1] = label.clone() #setup final label
      self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])#setup final prediction errors
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
      #reversed inference
        for j in reversed(range(len(self.layers))):
          if j != 0:
            self.prediction_errors[j] = self.mus[j] - self.outs[j]
            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1]) #e_j+1*d^vj+1|dvj -> update
            dx_l = self.prediction_errors[j] - self.predictions[j]
            self.mus[j] -= self.inference_learning_rate * (2*dx_l)
            torch.set_printoptions(threshold=float('inf'))
      #update weights
      weight_diffs = self.update_weights()
      #get loss:
      L = self.loss_fn(self.outs[-1],self.mus[-1]).item()#torch.sum(self.prediction_errors[-1]**2).item()
      #get accuracy
      acc = accuracy(self.no_grad_forward(inp),label)
      return L,acc,weight_diffs

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self, dataset, testset, n_epochs, n_inference_steps, logdir, savedir, old_savedir, save_every=1,
            print_every=10):
      if old_savedir != "None":
          self.load_model(old_savedir)
      pcn_accuracies = []
      pcn_test_accuracies = []
      losses = []
      accs = []
      weight_diffs_list = []
      test_accs = []
      for epoch in range(n_epochs):
          losslist = []
          print("Epoch: ", epoch)
          for i, (inp, label) in enumerate(dataset):
              if self.loss_fn != cross_entropy_loss:
                  label = onehot(label).to(DEVICE)
              else:
                  label = label.long().to(DEVICE)
              L, acc, weight_diffs = self.infer(inp.to(DEVICE), label)
              losslist.append(L)
          mean_acc, acclist = self.test_accuracy(dataset)
          accs.append(mean_acc)
          mean_loss = np.mean(np.array(losslist))
          losses.append(mean_loss)
          mean_test_acc, _ = self.test_accuracy(testset)
          test_accs.append(mean_test_acc)
          weight_diffs_list.append(weight_diffs)
          #pcn_accuracies.append((mean_acc, mean_test_acc))
          pcn_accuracies.append(mean_acc)
          pcn_test_accuracies.append(mean_test_acc)
          #make sure the directory exists
          os.makedirs('results', exist_ok=True)
          # Save to file
          with open('results/pcn_accuracies.pkl', 'wb') as f:
              pickle.dump(pcn_accuracies, f)
          with open('results/pcn_test_accuracies.pkl', 'wb') as f:
              pickle.dump(pcn_test_accuracies, f)
          print("ACCURACY: ", mean_acc)
          print("TEST ACCURACY: ", mean_test_acc)
          print("SAVING MODEL")
          self.save_model(logdir,savedir,losses,accs,weight_diffs_list,test_accs)

  def save_model(self,logdir, savedir,losses,accs,weight_diffs_list,test_accs):
      for i,l in enumerate(self.layers):
          l.save_layer(logdir,i)
      np.save(logdir +"/losses.npy",np.array(losses))
      np.save(logdir+"/accs.npy",np.array(accs))
      np.save(logdir+"/weight_diffs.npy",np.array(weight_diffs_list))
      np.save(logdir+"/test_accs.npy",np.array(test_accs))

      shutil.copytree(logdir, savedir, dirs_exist_ok=True)

      #subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)], shell = True)

      print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
      now = datetime.now()
      current_time = str(now.strftime("%H:%M:%S"))
      subprocess.call(['echo','saved at time: ' + str(current_time)], shell = True)

  def load_model(self,old_savedir):
      for (i,l) in enumerate(self.layers):
          l.load_layer(old_savedir,i)


class Backprop_CNN(object):
  def __init__(self, layers,loss_fn,loss_fn_deriv):
    self.layers = layers
    self.xs = [[] for i in range(len(self.layers)+1)]
    self.e_ys = [[] for i in range(len(self.layers)+1)]
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    for l in self.layers:
      l.set_weight_parameters()

  def forward(self, inp):
    self.xs[0] = inp #inp is a vector and has more samples ([64, 3, 32, 32])
    for i,l in enumerate(self.layers):
      self.xs[i+1] = l.forward(self.xs[i])
    return self.xs[-1] #last output after it went through the activation function (matrix)

  def backward(self,e_y):
    self.e_ys[-1] = e_y #e_y = dErrortotal|dout
    for (i,l) in reversed(list(enumerate(self.layers))):
      self.e_ys[i] = l.backward(self.e_ys[i+1])
    return self.e_ys[0]

  def update_weights(self,print_weight_grads=False,update_weight=False,sign_reverse=False):
    for (i,l) in enumerate(self.layers):
      dW = l.update_weights(self.e_ys[i+1],update_weights=update_weight,sign_reverse=sign_reverse)
      if print_weight_grads:
        print("weight grads Backprop: ", i)
        print("dW Backprop: ", dW*2)
        print("weight grad Backprop: ",l.get_true_weight_grad())

  def save_model(self,savedir,logdir,losses,accs,test_accs):
      for i,l in enumerate(self.layers):
          l.save_layer(logdir,i)
      np.save(logdir +"/losses.npy",np.array(losses))
      np.save(logdir+"/accs.npy",np.array(accs))
      np.save(logdir+"/test_accs.npy",np.array(test_accs))

      shutil.copytree(logdir, savedir, dirs_exist_ok=True)


      #subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
      print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
      now = datetime.now()
      current_time = str(now.strftime("%H:%M:%S"))
      subprocess.call(['echo','saved at time: ' + str(current_time)], shell = True)

  def load_model(self, old_savedir): #evtl self entfernen
      for (i,l) in enumerate(self.layers):
          l.load_layer(old_savedir,i)

  def test_accuracy(self,testset):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label).to(DEVICE))
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self, dataset,testset,n_epochs,n_inference_steps,savedir,logdir,old_savedir="",print_every=100,save_every=1):
    if old_savedir != "None":
        self.load_model(old_savedir)
    with torch.no_grad():
      bpn_accuracies = []
      bpn_test_accuracies = []
      accs = []
      losses = []
      test_accs =[]
      for n in range(n_epochs):
        print("Epoch backprop: ",n)
        losslist = []
        for (i,(inp,label)) in enumerate(dataset): #inp (single sample) mit seinem label, also sample einzeln
          out = self.forward(inp.to(DEVICE)) #output matrix
          if self.loss_fn != cross_entropy_loss:
            label = onehot(label).to(DEVICE)
          else:
            label = label.long().to(DEVICE)

          e_y = self.loss_fn_deriv(out, label) #e_y = dErrortotal|dout
          #e_y = out - label
          self.backward(e_y)
          self.update_weights(update_weight=True,sign_reverse=True)
          #loss = torch.sum(e_y**2).item()
          loss = self.loss_fn(out, label).item() #no matrix, just a simple value
          losslist.append(loss)
        mean_acc, acclist = self.test_accuracy(dataset)
        accs.append(mean_acc)
        mean_loss = np.mean(np.array(losslist))
        losses.append(mean_loss)
        mean_test_acc, _ = self.test_accuracy(testset)
        test_accs.append(mean_test_acc)
        bpn_accuracies.append(mean_acc)
        bpn_test_accuracies.append(mean_test_acc)
        os.makedirs('results', exist_ok=True)
        # Save to file
        with open('results/bpn_accuracies.pkl', 'wb') as f:
            pickle.dump(bpn_accuracies, f)
        with open('results/bpn_test_accuracies.pkl', 'wb') as f:
            pickle.dump(bpn_test_accuracies, f)
        print("ACCURACY: ", mean_acc)
        print("TEST ACCURACY: ", mean_test_acc)
        print("SAVING MODEL")
        self.save_model(logdir,savedir,losses,accs,test_accs)

      #plt.plot(range(1, n_epochs + 1), accs, 'g--', label='Backprop Train Accuracy')
      #plt.plot(range(1, n_epochs + 1), test_accs, 'g-', label='Backprop Test Accuracy')
      #plt.xlabel('Epoch')
      #plt.ylabel('Accuracy')
      #plt.title('Model Accuracy During Training')
      #plt.legend()
      #plt.grid(True)
      #plt.savefig("pipeline_training_accuracy.png", dpi=300, bbox_inches='tight')
      #plt.show()

if __name__ == '__main__':

    def pipeline(PC, Backprop, dataset, testset, n_epochs, n_inference_steps, BPsavedir, BPlogdir, PCsavedir, PClogdir, old_savedir="", print_every=100,
              save_every=1):
        print(f"Dataset size: {len(dataset)}")

        if old_savedir != "None":
            PC.load_model(old_savedir)
            Backprop.load_model(old_savedir)
        PCaccs = []
        PClosses = []
        PCweight_diffs_list = []
        PCtest_accs = []
        with torch.no_grad():
            backpropAccs = []
            backpropLosses = []
            backpropTest_accs = []
            for n in range(n_epochs):
                print("Epoch backprop: ", n)
                print("Epoch PC: ", n)
                backpropLossList = []
                PClosslist = []


                for (i, (inp, label)) in enumerate(dataset):

                    label_orig_bp = label.clone()  # clone the original label safely
                    # Backprop section
                    BackpropOut = Backprop.forward(inp.to(DEVICE))
                    if i == 0:
                        image, label = dataset[0]
                        single_image = image[0]  # shape: (3, 32, 32)

                        # Convert to (32, 32, 3) for imshow
                        single_image = np.transpose(single_image, (1, 2, 0))

                        # Plot
                        plt.imshow(single_image)
                        plt.axis('off')
                        plt.title("Example Image")
                        plt.show()


                    if Backprop.loss_fn != cross_entropy_loss:
                        label_bp = onehot(label_orig_bp).to(DEVICE)
                    else:
                        label_bp = label_orig_bp.long().to(DEVICE)

                    e_y = Backprop.loss_fn_deriv(BackpropOut, label_bp)
                    # e_y = out - label
                    Backprop.backward(e_y)
                    Backprop.update_weights(update_weight=True, sign_reverse=True)
                    # loss = torch.sum(e_y**2).item()
                    loss = Backprop.loss_fn(BackpropOut, label_bp).item()
                    backpropLossList.append(loss)

                # Backprop Accuracy
                BackpropMean_acc, BackpropAcclist = Backprop.test_accuracy(dataset)
                backpropAccs.append(BackpropMean_acc)
                BPmean_loss = np.mean(np.array(backpropLossList))
                backpropLosses.append(BPmean_loss)
                BPmean_test_acc, _ = Backprop.test_accuracy(testset)
                backpropTest_accs.append(BPmean_test_acc)
                print("ACCURACY Backprop: ", BackpropMean_acc)
                print("TEST ACCURACY Backprop: ", BPmean_test_acc)
                print("SAVING MODEL Backprop")
                Backprop.save_model(BPsavedir, BPlogdir, backpropLosses, backpropAccs, backpropTest_accs)

                # PC section
                for (i, (inp, label)) in enumerate(dataset):

                    if i == 0:
                        image, label = dataset[i]
                        single_image = image[i]  # shape: (3, 32, 32)

                        # Convert to (32, 32, 3) for imshow
                        single_image = np.transpose(single_image, (1, 2, 0))

                        # Plot
                        plt.imshow(single_image)
                        plt.axis('off')
                        plt.title("Example Image")
                        plt.show()

                    label_orig_pc = label.clone()  # clone the original label safely
                    if PC.loss_fn != cross_entropy_loss:
                        label_pc = onehot(label_orig_pc).to(DEVICE)
                    else:
                        label_pc = label_orig_pc.long().to(DEVICE)
                    L, acc, weight_diffs = PC.infer(inp.to(DEVICE), label_pc)
                    PClosslist.append(L)
                # PC Accuracy
                PCmean_acc, acclist = PC.test_accuracy(dataset)
                PCaccs.append(PCmean_acc)
                PCmean_loss = np.mean(np.array(PClosslist))
                PClosses.append(PCmean_loss)

                PCmean_test_acc, _ = PC.test_accuracy(testset)
                PCtest_accs.append(PCmean_test_acc)
                PCweight_diffs_list.append(weight_diffs)
                print("ACCURACY PC: ", PCmean_acc)
                print("TEST ACCURACY PC: ", PCmean_test_acc)
                print("SAVING MODEL")
                PC.save_model(PClogdir, PCsavedir, PClosses, PCaccs, PCweight_diffs_list, PCtest_accs)



        plt.plot(range(1, n_epochs + 1), PCaccs, 'g--', label='PC Train Accuracy')
        plt.plot(range(1, n_epochs + 1), PCtest_accs, 'g-', label='PC Test Accuracy')
        plt.plot(range(1, n_epochs + 1), backpropAccs, 'g--', label='Backprop Train Accuracy')
        plt.plot(range(1, n_epochs + 1), backpropTest_accs, 'g-', label='Backprop Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig("pipeline_training_accuracy.png", dpi=300, bbox_inches='tight')
        plt.show()

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    #parsing arguments
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir",type=str,default="savedir")

    parser.add_argument("--BPlogdir", type=str, default="BPlogs")
    parser.add_argument("--BPsavedir", type=str, default="BPsavedir")

    parser.add_argument("--PClogdir", type=str, default="PClogs")
    parser.add_argument("--PCsavedir", type=str, default="PCsavedir")
    parser.add_argument("--Coillogdir", type=str, default="Coillogdir")
    parser.add_argument("--Coilsavedir", type=str, default="Coilsavedir")

    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--learning_rate",type=float,default=0.0005)
    parser.add_argument("--N_epochs",type=int, default=100)
    parser.add_argument("--save_every",type=int, default=1)
    parser.add_argument("--print_every",type=int,default=10)
    parser.add_argument("--old_savedir",type=str,default="None")
    parser.add_argument("--n_inference_steps",type=int,default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=0.1)
    parser.add_argument("--network_type",type=str,default="pc")
    parser.add_argument("--dataset",type=str,default="cifar")
    parser.add_argument("--loss_fn", type=str, default="mse")

    args = parser.parse_args()
    print("Args parsed")
    #create folders


    def showPlot(batch_size, n_inference_steps):
        # Plot both
        plt.figure(figsize=(8, 6))

        epochs = list(range(1, args.N_epochs + 1))  # x-axis

        if os.path.exists('results/bpn_accuracies.pkl'):
            with open('results/bpn_accuracies.pkl', 'rb') as f:
                bpn_accuracies = pickle.load(f)
            converted_bpn_accuracies = [float(val) for val in bpn_accuracies]
            plt.plot(epochs, converted_bpn_accuracies, label="BPN Accuracy", marker='o', color='red')
            with open('results/bpn_test_accuracies.pkl', 'rb') as f:
                bpn_test_accuracies = pickle.load(f)
            converted_bpn_test_accuracies = [float(val) for val in bpn_test_accuracies]
            plt.plot(epochs, converted_bpn_test_accuracies, label ="BPN Test Accuracy", marker='x', color ='red')

        if os.path.exists('results/pcn_accuracies.pkl'):
            with open('results/pcn_accuracies.pkl', 'rb') as f:
                pcn_accuracies = pickle.load(f)
            converted_pcn_accuracies = [float(val) for val in pcn_accuracies]
            plt.plot(epochs, converted_pcn_accuracies, label="PCN Accuracy", marker='o', color='green')
            with open('results/pcn_test_accuracies.pkl', 'rb') as f:
                pcn_test_accuracies = pickle.load(f)
            converted_pcn_test_accuracies = [float(val) for val in pcn_test_accuracies]
            plt.plot(epochs, converted_pcn_test_accuracies, label ="PCN Test Accuracy", marker='x', color ='green')




        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Dataset: {dataset}, Model Accuracy During Training, Batch Size: {batch_size}, Inference Steps: {n_inference_steps}')
        plt.legend()
        plt.grid(True)
        plt.savefig("pipeline_training_accuracy.png", dpi=300, bbox_inches='tight')
        plt.show()



    if args.savedir:  # Checks if string is not empty
        os.makedirs(args.savedir, exist_ok=True)
        print(f"Created save directory at: {os.path.abspath(args.savedir)}")

    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        print(f"Created log directory at: {os.path.abspath(args.logdir)}")
    print("folders created")
    dataset,testset = get_cnn_dataset(args.dataset,args.batch_size)
    loss_fn, loss_fn_deriv = parse_loss_function(args.loss_fn)

    if args.dataset in ["cifar", "mnist","svhn"]:
        output_size = 10
    if args.dataset == "cifar100":
        output_size=100
    if args.dataset == "coil20":
        output_size = 20

    def onehot(x):
        z = torch.zeros([len(x),output_size])
        for i in range(len(x)):
            z[i,x[i]] = 1
        return z.float().to(DEVICE)
    #l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l2 = MaxPool(2,device=DEVICE)
    #l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,args.learning_rate,device=DEVICE)
    #l5 = FCLayer(120,84,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
    #l6 = FCLayer(84,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #layers =[l1,l2,l3,l4,l5,l6]

    #l1 = ConvLayer(128, 1, 6, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
    #l2 = MaxPool(2, device=DEVICE)
    #l3 = ConvLayer(62, 6, 16, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
    #l4 = ProjectionLayer((64, 16, 58, 58), 200, relu, relu_deriv, args.learning_rate, device=DEVICE)
    #l5 = FCLayer(200, 150, 64, args.learning_rate, relu, relu_deriv, device=DEVICE)


    #coil 20 layers
    #l1 = ConvLayer(128, 1, 6, 8, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
    #l2 = MaxPool(2, device=DEVICE)
    #l4 = ProjectionLayer((8, 6, 62, 62), 200, relu, relu_deriv, args.learning_rate, device=DEVICE)

    #l1 = ConvLayer(128, 1, 6, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
    #l2 = MaxPool(2, device=DEVICE)
    #l3 = ConvLayer(62, 6, 16, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
    #l4 = ProjectionLayer((args.batch_size, 16, 58, 58), 200, relu, relu_deriv, args.learning_rate, device=DEVICE)
    #l5 = FCLayer(200, 150, args.batch_size, args.learning_rate, relu, relu_deriv, device=DEVICE)

    #cifar layers
    l1= ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l2 = MaxPool(2,device=DEVICE)
    l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
    l4 = ProjectionLayer((64,16,10,10),200,relu,relu_deriv,args.learning_rate,device=DEVICE)
    l5 = FCLayer(200,150,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
    if args.loss_fn == "crossentropy":
      l6 = FCLayer(150,output_size,args.batch_size,args.learning_rate,softmax,linear_deriv,device=DEVICE)
    else:
      #l5 = FCLayer(200, output_size, 8, args.learning_rate, relu, relu_deriv, device=DEVICE)
      l6 = FCLayer(150,output_size,args.batch_size,args.learning_rate,linear,linear_deriv,device=DEVICE)
    layers =[l1,l2,l3,l4,l5,l6]
    #layers =[l1,l2,l3,l4,l5,l6]
    #l1 = ConvLayer(32,3,20,64,4,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
    #l2 = ConvLayer(29,20,50,64,5,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
    #l3 = ConvLayer(25,50,50,64,5,args.learning_rate,tanh,tanh_deriv,stride=2,padding=1,device=DEVICE)
    #l4 = ConvLayer(12,50,5,64,3,args.learning_rate,tanh,tanh_deriv,stride=1,device=DEVICE)
    #l5 = ProjectionLayer((64,5,10,10),200,sigmoid,sigmoid_deriv,args.learning_rate,device=DEVICE)
    #l6 = FCLayer(200,100,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #l7 = FCLayer(100,50,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #l8 = FCLayer(50,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
    #layers =[l1,l2,l3,l4,l5,l6,l7,l8]
    if args.network_type == "pc":
        net = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv,device=DEVICE)

    elif args.network_type == "backprop":
        net = Backprop_CNN(layers,loss_fn = loss_fn,loss_fn_deriv = loss_fn_deriv)

    elif args.network_type == "pipeline":
        PCnet = PCNet(layers,args.n_inference_steps,args.inference_learning_rate,loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv,device=DEVICE)
        backpropnet = Backprop_CNN(layers,loss_fn = loss_fn,loss_fn_deriv = loss_fn_deriv)
        pipeline(PCnet, backpropnet, dataset[0:-2],testset[0:-2],args.N_epochs,args.n_inference_steps,args.BPsavedir, args.BPlogdir, args.PCsavedir, args.PClogdir,args.old_savedir,args.save_every,args.print_every)

    else:
        raise Exception("Network type not recognised: must be one of 'backprop', 'pc'")
    net.train(dataset[0:-2],testset[0:-2],args.N_epochs,args.n_inference_steps,args.savedir,args.logdir,args.old_savedir,args.save_every,args.print_every)
    #net.train(dataset[0:-2],testset[0:-2],args.N_epochs,args.n_inference_steps,args.Coilsavedir,args.Coillogdir,"None",args.save_every,args.print_every)
    showPlot(args.batch_size, args.n_inference_steps)