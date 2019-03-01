'''params must provide: --data, --net, --method, --expect_acc/--validk, --maxk(for cat)'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import json, random

import torchvision
import torchvision.transforms as transforms
from PGDAttack import LinfPGDAttack

import os, copy
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR,ExponentialLR
from random import randint
import timeit
from torch.utils.data.sampler import *


with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
np.random.seed(config['np_random_seed'])

# Setting up training parameters
train_batch_size = config['training_batch_size']
eval_batch_size = max(64, config['eval_batch_size'])
criterion = nn.CrossEntropyLoss()

parser = argparse.ArgumentParser(description='CAT Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='whether to resume')
parser.add_argument('--resumefrom', default=None, type=str, help='resume from which checkpoint')
parser.add_argument('--savename', default=None, type=str, help='same checkpoint name')
parser.add_argument('--expect_acc','-a', default=0.95, type=float, help='update k when testacc exceed this')
parser.add_argument('--net', default='cnn', type=str.lower, help='choose from: cnn, resnet50, densenet161')
parser.add_argument('--data', default='mnist', type=str.lower, help='choose from: mnist, cifar10, svhn')
parser.add_argument('--method', default='cat', type=str.lower, help='choose from: at, basic, cat, maxbm, nat (only batch mixing for max_k)')
parser.add_argument('--maxk', default=0, type=int, help='max k')
parser.add_argument('--k', default=-1, type=int, help='k')
parser.add_argument('--validk', default='k', type=str, help='choice of k for validation: byk, k')
parser.add_argument('--sgd', action='store_true', help='use sgd optimizer')
args = parser.parse_args()
# adjust
if args.data=='mnist':
    args.net=='cnn'
#     args.k=40
# else:
#     args.k=7
max_k = 40 if args.data=='mnist' else 7
if args.maxk>0: max_k = args.maxk
if args.savename==None: args.savename=args.method+'.'+args.data+'.'+args.net
savefolder = args.savename
if args.method=='cat': 
    savefolder += '.maxk'+str(max_k)
if args.validk!=None: savefolder+='.valid'+args.validk
elif args.method=='cat' or args.method=='basic': savefolder += '.k+%.2f'%args.expect_acc
if args.sgd: savefolder += '.sgd'
expect_acc = float(args.expect_acc)

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
])

transform_valid = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(), 
])

trainset, validset = None, None
valid_size = 0.02
if args.data=='mnist':
    trainset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    validset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
elif args.data=='cifar10':
    trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_valid)
elif args.data=='svhn': 
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    validset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_valid)

##### get random valid indices
valid_indices, train_indices = [], []
randidxfn_valid = './data/'+args.data+'.validk.randidx'
randidxfn_train = './data/'+args.data+'.traink.randidx'
if os.path.exists(randidxfn_valid): # load to **_randidx
    valid_indices = np.genfromtxt(randidxfn_valid).astype('int')
    train_indices = np.genfromtxt(randidxfn_train).astype('int')
else: # generate randidx, load to **_randidx and save to file
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False)
    train_labels = {}
    gid = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        for t in Variable(targets).data:
            if t in train_labels: train_labels[t].append(gid)
            else: train_labels[t]=[gid]
            gid += 1
    for tk, ti in train_labels.items():
        #np.random.seed(tk) # fix for reproducibility         
        np.random.shuffle(ti)                            
        split_sz = int(valid_size*len(ti))
        valid_indices += (ti[:split_sz])
        train_indices += (ti[split_sz:])
    valid_indices, train_indices = np.array(valid_indices), np.array(train_indices)
    np.savetxt(randidxfn_valid, valid_indices, fmt='%.f')
    np.savetxt(randidxfn_train, train_indices, fmt='%.f')

np.random.shuffle(train_indices)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=eval_batch_size, sampler=valid_sampler, num_workers=2)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

from collections import OrderedDict

# Model
def resumeFrom(modelFN, begin=True):
    global k, prevTime, optimizer, net
        # Load checkpoint.
    print('==> Resuming from checkpoint..', modelFN)
    assert os.path.exists(modelFN), 'Error: no checkpoint file found!'
    if begin: prevTime = int(os.path.basename(modelFN).split('.')[5][:-3])
    print('prevTime', prevTime)
    checkpoint = torch.load(modelFN)
    try:
        net.load_state_dict(checkpoint['net'])
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    this_epoch = checkpoint['epoch']
    k = checkpoint['k']
    if os.path.exists(modelFN+'.optim'):
        optimizer = optim.Adam(net.parameters(), lr=1e-4) 
        optimizer.load_state_dict(torch.load(modelFN+'.optim')['optim'])
    else:
        pass

    return this_epoch

k, prevTime = 0, 0
net=None
print('==> Building model..')
if args.net=='cnn':
    net = nn.Sequential(
        nn.Conv2d(1, 32, 5, stride=1,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5, stride=1,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
elif args.net=='resnet50': net = ResNet50()
elif args.net=='densenet161': net = DenseNet161()
optimizer = None #optim.Adam(net.parameters(), lr=1e-4)
if args.resume:
    if args.resumefrom == None:
        model_dir = 'checkpoint/'+savefolder
        all_models = [name for name in os.listdir(model_dir) if name.split('.')[-1] != 'optim']
        assert len(all_models)>=1, 'Error: no checkpoint file found in '+model_dir
        args.resumefrom = os.path.join(model_dir, sorted(all_models, key=lambda x : int(x.split('.')[3]))[-1]) # the biggest model
        print('args.resumefrom', args.resumefrom)
    start_epoch = resumeFrom(args.resumefrom)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
if optimizer is None: optimizer = optim.Adam(net.parameters(), lr=1e-4)
if args.sgd: 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
epsilon, step_size = 0, 0
if args.data=='mnist': epsilon, step_size = 0.3, 0.6
elif args.data=='cifar10': epsilon, step_size = 0.032, 0.064
elif args.data=='svhn': epsilon, step_size = 0.047, 0.094
attack=LinfPGDAttack(net,epsilon, 1, step_size, True, 'xent')

def advtrain_NAT(epoch):
    print('\nEpoch: %d' % epoch, 'k: %d' % k)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs_var,targets_var = Variable(inputs),Variable(targets)
        net.train()
        outputs = net(inputs_var)
        loss = criterion(outputs, targets_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        current_num = targets_var.size(0)
        current_correct = predicted.eq(targets_var.data).cpu().sum()
        
        total += current_num
        correct += current_correct

        train_loss += loss.data[0]
    
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return validnSaveModel(acc, k)

def advtrain_BASIC(epoch):
    global k, max_k, net
    print('\nEpoch: %d' % epoch, 'k: %d' % k)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs_var,targets_var = Variable(inputs),Variable(targets)
        net.eval()
        inputs_adv = attack.perturb_true(inputs_var, targets_var,k) if k!=0 else Variable(inputs)
        net.train()
        outputs = net(inputs_adv)
        loss = criterion(outputs, targets_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        current_num = targets_var.size(0)
        current_correct = predicted.eq(targets_var.data).cpu().sum()
        
        total += current_num
        correct += current_correct

        train_loss += loss.data[0]
    
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return validnSaveModel(acc, k)

def getKset(curk):
    if args.method=='at' or args.method=='maxbm': return [1,20,100]
    if args.method=='nat': return [0]
    if args.validk=='byk':
        if curk<=10: return [i for i in range(curk+1)]
        elif curk>10 and curk<20: return [i for i in range(11)]
        elif curk>=20: return [i for i in range(11)]+[20]
    elif args.validk=='k':
        return [curk]

def regularLoss(inputs, targets):  # in: inputs, targets; out: # correct prediction, loss value
    global k, net
    inputs_var, targets_var = Variable(inputs), Variable(targets)
    each_batch = inputs.size(0) / (k+1)
    if each_batch==0: return -1, -1
    for cur_k in range(k+1):
        if cur_k!=k:
            inputs_now, targets_now = Variable(inputs[cur_k*each_batch:(cur_k+1)*each_batch]), Variable(targets[cur_k*each_batch:(cur_k+1)*each_batch])
        else:
            inputs_now, targets_now = Variable(inputs[k*each_batch:]), Variable(targets[k*each_batch:])  # this gets max share
        inputs_adv = attack.perturb_true(inputs_now, targets_now,cur_k) if cur_k!=0 else inputs_now
        
        if cur_k!=k:
            inputs_var[cur_k*each_batch:(cur_k+1)*each_batch], targets_var[cur_k*each_batch:(cur_k+1)*each_batch]=inputs_adv, targets_now
        else:
            inputs_var[k*each_batch:], targets_var[k*each_batch:]= inputs_adv, targets_now
    # img = torchvision.utils.make_grid(inputs_adv.data[-3:,:,:,:])
    outputs = net(inputs_var)
    loss = criterion(outputs, targets_var)
    #print('reg loss', loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    current_correct = predicted.eq(targets_var.data).cpu().sum()

    return current_correct, loss.data[0]

prevBestEpoch = {'fn':None, 'eid':0, 'acc':0, 'loss':0}
def advtrain_CAT(epoch):  # basic + batch mixing
    global k, max_k, prevBestEpoch, net
    print('\nEpoch: %d' % epoch, 'k: %d' % k)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #########
        current_correct, current_loss = regularLoss(inputs, targets)        
        if current_correct==-1 and current_loss==-1: continue
        ###############
        correct += current_correct
        train_loss += current_loss
        current_num = Variable(targets).size(0)
        total += current_num
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 1.*correct/total
    return validnSaveModel(acc, k)

def validnSaveModel(trainAcc, inK, new=True): # , k, acc tbd: saveModel contains: state->time->filename(including acc&k)->...
    global savefolder, starttime, net, epoch, optimizer, k
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/'+savefolder):
        os.mkdir('checkpoint/'+savefolder)

    state = {
    'net': net.state_dict(), #net.module if use_cuda else net,
    'k': inK, # could be k or static_k
    'epoch': epoch,
    }
    state_optim = {
    'optim': optimizer.state_dict()
    }
    sofartime = timeit.default_timer()
    savename = './checkpoint/'+savefolder+'/'+args.savename+'.'+str(epoch)+'.step%d'%(inK)+'.'+ "%.f" % (sofartime-starttime)+'sec'
    if args.validk != None: # use validation
        validAccNQ, validLossNQ, validAccQ, validLossQ = valid_wca(getKset(inK))
        savename += '.validNQ'+'%.1f'%validAccNQ+'.lossNQ'+'%.1f'%validLossNQ+'.validQ'+'%.1f'%validAccQ+'.lossQ'+'%.1f'%validLossQ+'.train'+'%.1f'%(trainAcc)
        torch.save(state, savename)
        saveOptim(state_optim, savename+'.optim', epoch) #torch.save(state_optim, savename_optim)
        #if (validAcc>prevBestEpoch['acc'] or validLoss<prevBestEpoch['loss']) and trainAcc>0.9:
        print('new model:', savename, 'validAccNQ:', validAccNQ, 'validAccQ:', validAccQ, 'trainAcc:', trainAcc)
        validAcc = validAccQ if args.method=='cat' else validAccNQ
        validLoss = validLossQ if args.method=='cat' else validLossNQ
        if (validAcc>prevBestEpoch['acc'] ) or (not new):
            prevBestEpoch['fn'] = savename
            prevBestEpoch['eid'] = epoch
            prevBestEpoch['acc'] = validAcc
            prevBestEpoch['loss'] = validLoss
            print('new prev best', savename, epoch, validAcc)
        elif epoch-prevBestEpoch['eid'] >= 10 and trainAcc>0.9: # resume from prev Epoch
            prevBestEpoch['acc']=0 # just make sure to save the first epoch in new k
            if args.method=='cat' or args.method=='basic':
                removeFNs(prevBestEpoch['fn'], prevBestEpoch['eid']+1, epoch)
                resumeFrom(prevBestEpoch['fn'], False)
                k += 1
                epoch += 1
                print('use prev best model for new k', k)
                validnSaveModel(0, k, False) # recursive call validnSave to validate new k set with previous best model
                print('cur epoch:', epoch, 'next k', k, "prevBestEpoch['eid']", prevBestEpoch['eid'])
            else:
                print('cur epoch:', epoch, "should have resumed at", prevBestEpoch['eid'])
    else:
        savename += '.train'+'%.1f'%(trainAcc)
        torch.save(state, savename)
        saveOptim(state_optim, savename+'.optim', epoch) #torch.save(state_optim, savename_optim)
        if (args.method=='cat' or args.method=='basic') and trainAcc>expect_acc and k<max_k: # use max acc
            k+=1
    return epoch

def saveOptim(state_optim, savename_optim, epoch):
    torch.save(state_optim, savename_optim)
    tmp = savename_optim.split('/')
    fn = '/'.join(tmp[:-1])+'/'+'.'.join(tmp[-1].split('.')[:3])+'.'+str(epoch-11)+'.*'+'.optim'
    print('rm '+ fn)
    try:
        os.system('rm '+ fn)
    except:
        pass

def removeFNs(fileTemp, startE, endE):
    print('startE, endE', startE, endE)
    for ie in range(startE, endE+1):
        tmp = fileTemp.split('/')
        fn = '/'.join(tmp[:-1])+'/'+'.'.join(tmp[-1].split('.')[:3])+'.'+str(ie)+'.*'
        print('to delete', fn+'.optim')
        try:
            #os.system('rm '+fn)
            os.system('rm '+fn+'.optim')
        except:
            pass

def valid_wca(kset):
    print('kset', kset)
    net.eval()
    test_lossNQ, test_lossQ = 0, 0
    totalcorrectNQ, totalcorrectQ = 0, 0
    total  =  0
    predNQ, predQ = None, None
    for batch_idx, (inputs, targets) in enumerate(validloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs,targets = Variable(inputs), Variable(targets)
        correctNQ, correctQ = None, None
        for k in kset:
            inputs_advNQ, inputs_advQ = inputs, inputs
            if k>0: 
                inputs_advNQ, inputs_advQ=attack.perturb_quad(inputs, targets, k, 16)
            outputsNQ, outputsQ = net(inputs_advNQ), net(inputs_advQ)
            lossNQ = criterion(outputsNQ, targets)
            lossQ = criterion(outputsQ, targets)
            test_lossNQ += lossNQ.data[0]
            test_lossQ += lossQ.data[0]
            _, predictedNQ = torch.max(outputsNQ.data, 1)
            _, predictedQ = torch.max(outputsQ.data, 1)
            if correctNQ is None:
                correctNQ = predictedNQ.eq(targets.data).cpu()
            else:
                correctNQ *= predictedNQ.eq(targets.data).cpu()
            if correctQ is None:
                correctQ = predictedQ.eq(targets.data).cpu()
            else:
                correctQ *= predictedQ.eq(targets.data).cpu()
        total += targets.size(0)
        totalcorrectNQ+=correctNQ.sum()
        totalcorrectQ+=correctQ.sum()
        
        progress_bar(batch_idx, len(validloader), 'valid_Loss: %.3f | valid_Acc: %.3f%% (%d/%d)'
                     % (test_lossQ / (batch_idx + 1), 100. * totalcorrectQ / total, totalcorrectQ, total))
    accNQ = 100. * totalcorrectNQ / total
    print('valid acc NQ', accNQ, 'valid loss NQ', test_lossNQ)
    accQ = 100. * totalcorrectQ / total
    print('valid acc Q', accQ, 'valid loss Q', test_lossQ)
    return accNQ, test_lossNQ, accQ, test_lossQ


def advtrain_MAXBM(epoch):  # only batch mixing for max_k
    global net
    static_k = 40 if args.data=='mnist' else 7
    if args.data=='svhn': static_k=10
    print('\nEpoch: %d' % epoch, 'static_k: %d' % static_k)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs_var, targets_var = Variable(inputs), Variable(targets)
        each_batch = inputs.size(0) / (static_k+1)
        for cur_k in range(static_k+1):
            if cur_k!=static_k:
                inputs_now, targets_now = Variable(inputs[cur_k*each_batch:(cur_k+1)*each_batch]), Variable(targets[cur_k*each_batch:(cur_k+1)*each_batch])
            else:
                inputs_now, targets_now = Variable(inputs[static_k*each_batch:]), Variable(targets[static_k*each_batch:])
            inputs_adv = attack.perturb_true(inputs_now, targets_now,cur_k) if cur_k!=0 else inputs_now
            
            if cur_k!=static_k:
                inputs_var[cur_k*each_batch:(cur_k+1)*each_batch], targets_var[cur_k*each_batch:(cur_k+1)*each_batch]=inputs_adv, targets_now
            else:
                inputs_var[static_k*each_batch:], targets_var[static_k*each_batch:]= inputs_adv, targets_now
        outputs = net(inputs_var)
        loss = criterion(outputs, targets_var)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        current_num = targets_var.size(0)
        current_num = targets_var.size(0)
        current_correct = predicted.eq(targets_var.data).cpu().sum()
        
        total += current_num
        correct += current_correct

        train_loss += loss.data[0]
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    validnSaveModel(acc, static_k)

def advtrain_AT(epoch):
    global net
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    static_k = 40 if args.data=='mnist' else 7
    if args.k>0: static_k=args.k
    if args.data=='svhn': static_k=10
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
            #targets = targets.long().cuda() if args.data=='svhn' else targets.cuda()
            targets = targets.cuda()
        inputs_var,targets_var = Variable(inputs),Variable(targets)
        net.eval()
        inputs_adv = attack.perturb_true(inputs_var, targets_var, static_k)
        net.train()
        outputs = net(inputs_adv)
        loss = criterion(outputs, targets_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        current_num = targets_var.size(0)
        current_correct = predicted.eq(targets_var.data).cpu().sum()
        
        total += current_num
        correct += current_correct

        train_loss += loss.data[0]
	
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    validnSaveModel(acc, static_k)

starttime = timeit.default_timer()-prevTime
#for epoch in range(start_epoch, start_epoch+300):
epoch=start_epoch+1
while epoch<start_epoch+300:
    #scheduler.step()
    if args.method=='at':
        advtrain_AT(epoch)
    elif args.method=='basic':
        epoch = advtrain_BASIC(epoch)
    elif args.method=='nat':
        if args.sgd:
            if epoch < 150: lr=0.1
            elif epoch < 250: lr=0.01
            elif epoch < 350: lr=0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        epoch = advtrain_NAT(epoch)
    elif args.method=='cat':
        epoch=advtrain_CAT(epoch)
    elif args.method=='maxbm':
        advtrain_MAXBM(epoch)
    epoch += 1
