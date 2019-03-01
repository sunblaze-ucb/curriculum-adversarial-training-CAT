'''params to provide: --model_dir, --data, --ttest, --interval, --net, --start, --end'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import json, datetime

import torchvision
import torchvision.transforms as transforms
from PGDAttack import LinfPGDAttack

import os
import argparse

from models import *
from utils import progress_bar
from utils import quantization
from torch.autograd import Variable
import torch.utils.data.sampler as sampler


with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
np.random.seed(config['np_random_seed'])

# Setting up training parameters
train_batch_size = config['training_batch_size']
eval_batch_size = max(32,config['eval_batch_size'])
criterion = nn.CrossEntropyLoss()
criterion_perSample = nn.CrossEntropyLoss(reduce=False)

parser = argparse.ArgumentParser(description='CAT Testing')
parser.add_argument('--model_dir', default=None, type=str, help='model file or folder')
parser.add_argument('--data', default='mnist', type=str.lower, help='choose from: mnist, cifar10, svhn')
parser.add_argument('--net', default='cnn', type=str.lower, help='choose from: cnn, resnet50, densenet161')
parser.add_argument('--ttest', default='test', type=str.lower, help='choose from: train, test, valid')
parser.add_argument('--quant', default='true', type=str.lower, help='whether to apply quantization or not')
parser.add_argument('--interval', default=1, type=int, help='time interval between every two saved ckpts, in seconds')
parser.add_argument('--start', default=1, type=int, help='epoch id to start')
parser.add_argument('--end', default=10000, type=int, help='epoch id to end')
parser.add_argument('--method', default='cat', type=str.lower, help='temp arg to specify method to eval, for inconsistent model names')
parser.add_argument('--cwloss', action='store_true', help='use cwloss if specified')
parser.add_argument('--name', action='store_true', help='not extract info from model names if specified')
parser.add_argument('--nat', action='store_true', help='only test natural example')
args = parser.parse_args()
if args.model_dir[-1]=='/': args.model_dir = args.model_dir[:-1]
acc_out_file = '.'.join([args.model_dir, args.ttest, '%d' % args.interval])
if args.cwloss: acc_out_file += '.cwloss'
print('acc_out_file', acc_out_file)

assert os.path.exists(args.model_dir), 'Error: '+args.model_dir+' does not exist!'

bQuant= (args.quant=='true')
if os.path.basename(args.model_dir)[:2]=='at':
    bQuant = False

def getSec(modelName):
#  return int(modelName.split('.')[-1][:-3])
    timeId = 5   # tbd: check to be consistent with trained model names, now new trained at models have the same naming convention
    return int(os.path.basename(modelName).split('.')[timeId][:-3])

all_models = []
if os.path.isdir(args.model_dir):
    if args.name: 
        for name in os.listdir(args.model_dir):
            if name[-7:]=='results': continue
            eid = int(name[6:])
            if eid>=args.start and eid<=args.end: all_models.append(name)
#        all_models = [name for name in os.listdir(args.model_dir) if name[-7:]!='results' ]
        all_models = sorted(all_models, key=lambda x : int(x[6:]))
    else:
        for name in os.listdir(args.model_dir):
            if name.find('sec')<0 or name.split('.')[-1]=='results' or name.split('.')[-1]=='optim': continue  # avoid results folder and non-models
            eid = int(name.split('.')[3])
            if eid>=args.start and eid<=args.end:
                all_models.append(name)
        # file name example: at.mnist.cnn.138.9483sec (new: at.mnist.cnn.138.step40.9483sec.acc90), sort with epoch #
        all_models = sorted(all_models, key=lambda x : int(x.split('.')[3]))
        print('all_models', all_models)
        assert (len(all_models)>0), 'no model within provided start and end range'
        
else: # isfile
    all_models = [args.model_dir]

all_models = all_models[::-1] # reverse timing
print('all_models', all_models)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform = transforms.Compose([ # only for cifar and svhn
    transforms.CenterCrop(32),
    transforms.ToTensor(), 
])


bTest = (args.ttest=='test')
bTrain = not bTest
splitKey = 'train' if bTrain else 'test'
dataset, data_randidx = None, None
if args.data=='mnist':
    dataset=torchvision.datasets.MNIST(root='./data', train=bTrain, download=True, transform=transforms.Compose([transforms.ToTensor()]))
elif args.data=='cifar10':
    dataset=torchvision.datasets.CIFAR10(root='./data', train=bTrain, download=True, transform=transform)
elif args.data=='svhn': 
    dataset = torchvision.datasets.SVHN(root='./data', split=splitKey, download=True, transform=transform)

randidx_fn = './data/'+args.data+'.'+args.ttest+'.randidx'
if not bTest: randidx_fn = './data/'+args.data+'.'+args.ttest+'k.randidx' # use indices for train/valid

if os.path.exists(randidx_fn): # load to **_randidx
    data_randidx = np.genfromtxt(randidx_fn).astype('int')
else: # generate randidx, load to **_randidx and save to file
    data_randidx = np.random.permutation(np.array([i for i in range(len(dataset))])) # done np.random.seed at beginning, should be fixed random
    np.savetxt(randidx_fn, data_randidx, fmt='%.f')

print('data_randidx[:10]', data_randidx[:10], 'size', len(data_randidx))


class FixedRandSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, indices, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start
        self.indices = indices
        print('self.num_samples, self.start', self.num_samples, self.start)

    def __iter__(self):
        return (self.indices[i] for i in range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples




#attack = LinfPGDAttack(net, 0.032, 1, 0.064, True, 'cw')

net = None

def testadv_worstcase(loader, saveresultsfn, curSz):
    if os.path.exists(saveresultsfn+'.quant0') or os.path.exists(saveresultsfn+'.quant1'):
        print('this model has been tested, thus skip')
        arr0 = np.genfromtxt(saveresultsfn+'.quant0', delimiter='\t').astype('float')[1:3]
        arr1 = np.genfromtxt(saveresultsfn+'.quant1', delimiter='\t').astype('float')[1:3]
        return [arr0[0, -1], arr1[0, -1]], [arr0[1, -1], arr1[1, -1]]
    bTest = not args.ttest=='train'
    print('results file', saveresultsfn)

    ks = [0,1,2,3,4,5,6,7,8,9,10,20,50,100]
    if args.cwloss: ks = [100]
    if args.nat: ks=[0]
    net.eval()
    adv_cnt = 2 if bTest else 1
    test_losses = [0 for _ in range(adv_cnt)]
    loss_perKs = [dict((k, 0) for k in ks) for _ in range(adv_cnt)]
    totalcorrects = [0 for _ in range(adv_cnt)]
    preds = [None for _ in range(adv_cnt)]
    saveresults = [[] for _ in range(adv_cnt)] # tbd, consider to init as an np array np.zeros(adv_cnt, curSz)
    savelosses = [[] for _ in range(adv_cnt)]
    for i in range(curSz):
        for s in range(len(saveresults)):
            saveresults[s].append([])
            savelosses[s].append([])
    total = 0
    tidx = 0  # index in test example dataset
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs,targets = Variable(inputs), Variable(targets)
        #ks = [1,2,3]
        corrects = [0] * (2 if bTest else 1)
        for k in ks:
            if k==0: inadv, inadv_qt = inputs, inputs
            else: inadv, inadv_qt = attack.perturb_quad(inputs, targets, k,16)
            input_advs = [inadv]
            if bTest: input_advs.append(inadv_qt)
            elif bQuant: input_advs=[inadv_qt]
            for ii in range(adv_cnt):
                output = net(input_advs[ii])
                loss = criterion(output, targets) 

                loss_perSample = criterion_perSample(output, targets)
                np_loss_perSample = loss_perSample.data.cpu().numpy() if use_cuda else loss_perSample.data.numpy()


                test_losses[ii] += loss.data[0] # tbd: this should be able to be computed using average(sum(np_loss_perSample))
                loss_perKs[ii][k] += loss.data[0]
                _, pred = torch.max(output.data, 1)
                curRst = pred.eq(targets.data).cpu()
                for ic, cc in enumerate(curRst):
                    saveresults[ii][tidx+ic].append(int(cc)) # tbd: if np array, use concatenate
                    savelosses[ii][tidx+ic].append(np_loss_perSample[ic])
                if k==0:
                    corrects[ii] = curRst
                else:
                    corrects[ii] *= curRst
        for ii in range(len(saveresults)):
            for ic, cc in enumerate(corrects[ii]): 
                saveresults[ii][tidx+ic].append(int(cc))
                savelosses[ii][tidx+ic].append(max(savelosses[ii][tidx+ic]))
            totalcorrects[ii] += corrects[ii].sum()
        total += targets.size(0)
        tidx += targets.size(0)
        
        progress_bar(batch_idx, len(data_randidx), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_losses[-1] / (batch_idx + 1), 100. * totalcorrects[-1] / total, totalcorrects[-1], total))
    accs = []
    for ii in range(adv_cnt):
        acc = 100. * totalcorrects[ii] / total
        ## acc for each attack
        attack_accs = [0] * len(saveresults[ii][0])  # tbd: convert to np array and use np.sum/np.length to compute attack_accs
        for ie, er in enumerate(saveresults[ii]):
            for ik, kr in enumerate(er):
                if kr>0.5: attack_accs[ik] += 1
        attack_accs = [100.*cnt/total for cnt in attack_accs]
        losses = [loss_perKs[ii][k] for k in ks] + [test_losses[ii]]
        saveresults[ii].insert(0, attack_accs)
        saveresults[ii].insert(1, losses)
        # save results
        fn_rst = saveresultsfn+'.quant%.f'%(1 if (bQuant and not bTest) else ii)
        fn_loss = fn_rst + '.loss'
        tosave = [saveresults, savelosses]
        for idx_f, fn in enumerate([fn_rst, fn_loss]):
            with open(fn, 'w') as fp:
                for k in ks: fp.write("%s\t" % k)
                fp.write('total\n')
                ffmt = '%.f' if idx_f==0 else '%.2f'
                np.savetxt(fp, np.array(tosave[idx_f][ii]), fmt=ffmt, delimiter='\t')

        accs.append(acc)
    return accs, test_losses

epsilon, step_size = 0, 0
if args.data=='mnist': epsilon, step_size = 0.3, 0.6
elif args.data=='cifar10': epsilon, step_size = 0.032, 0.064
elif args.data=='svhn': epsilon, step_size = 0.047, 0.094

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
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

from collections import OrderedDict
if __name__ == '__main__':
    #acc = testnat()
    #print('natural: %f'%(acc))
    subPercent = 0.04 if args.ttest=='train' else 0.1
    if args.ttest=='valid': subPercent=0.1
    subsetSz = subPercent * len(data_randidx) 
    nSubsets = int(math.ceil(len(data_randidx)/subsetSz))
    macc_sofar_dict = dict((m, [0,0]) for m in all_models)
    mloss_sofar_dict = dict((m, [0,0]) for m in all_models) 
    subsetIdx = 0
    while subsetIdx < nSubsets:
        curSz = int(min(len(data_randidx), (subsetIdx+1)*subsetSz)-subsetIdx*subsetSz)
        loader = torch.utils.data.DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, num_workers=1, \
                                               sampler=FixedRandSampler(data_randidx, curSz, start=int(subsetIdx*subsetSz)))
        subsetIdx += 1
        prevEpoch, prevTime = 10000, 10000000000 # to make sure the first will be tested  
        for m in all_models:  # m: at.mnist.cnn.138[.step6].9483sec
            try:
                thisEpoch, thisTime = int(m.split('.')[3]), getSec(m)
            except:
                thisEpoch, thisTime = 1, 0  # maybe model name is not in that format
            #if thisTime - prevTime >= args.interval:  # test per interval, or you could use epoch#
            if prevTime - thisTime >= args.interval or thisEpoch==1:  # test per interval, or you could use epoch#
                print('prevTime, ', prevTime, 'thisTime', thisTime, 'thisEpoch', thisEpoch)
                print('cur model: ', m, '\n')
                pm = os.path.join(args.model_dir, m) if len(all_models)>1 else m
                checkpoint = torch.load(pm)
                try:
                    net.load_state_dict(checkpoint['net'])
                except:
                    try:
                        net = checkpoint['net']
                    except:
                        try:
                            new_state_dict = OrderedDict()
                            for k, v in checkpoint['net'].items():
                                name = k[7:] # remove `module.`
                                new_state_dict[name] = v
                            net.load_state_dict(new_state_dict)
                        except:
                            new_state_dict = OrderedDict()
                            for k, v in checkpoint['net'].items():
                                name = 'module.'+k # remove `module.`
                                new_state_dict[name] = v
                            net.load_state_dict(new_state_dict)
                attack=None
                resultsfolder = pm + '.results'
                if args.cwloss: 
                    resultsfolder = pm + '.cw.results'
                    attack = LinfPGDAttack(net,epsilon, 1, step_size, True, 'cw')
                else: attack = LinfPGDAttack(net,epsilon, 1, step_size, True, 'xent')
    
                if not os.path.isdir(resultsfolder):
                    os.mkdir(resultsfolder)

                saveresultsfn = os.path.join(resultsfolder, args.ttest+'.%.f_%.f'%(subsetIdx, nSubsets))
                worstAccs, losses = testadv_worstcase(loader, saveresultsfn, curSz)
                if worstAccs==-1: 
                    prevEpoch, prevTime = thisEpoch, thisTime
                    continue
                for i in range(len(losses)):
                    macc_sofar_dict[m][i] = (macc_sofar_dict[m][i]*(subsetIdx-1)*subsetSz + worstAccs[i]*curSz) / ((subsetIdx-1)*subsetSz+curSz)
                    mloss_sofar_dict[m][i] += losses[i]
                out_str = '\n dataset %.f/%.f, sz_so_far: %.f, model_name: '%(subsetIdx, nSubsets, (subsetIdx-1)*subsetSz+curSz)+ m + ', worst_case_acc: %.2f , total_loss: %.2f'%(macc_sofar_dict[m][0], mloss_sofar_dict[m][0])
                if len(worstAccs)>1:
                    out_str += ', worst_case_acc_quant: %.2f , total_loss_quant: %.2f'%(macc_sofar_dict[m][1], mloss_sofar_dict[m][1])
                print(out_str)
                append_write = 'a' if os.path.exists(acc_out_file) else 'w'
                with open(acc_out_file, append_write) as fp:
                    fp.write(out_str)
    		
                prevEpoch, prevTime = thisEpoch, thisTime
