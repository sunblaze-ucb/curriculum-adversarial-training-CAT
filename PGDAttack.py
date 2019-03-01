from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from utils import quantization

def cwloss(output, target, confidence=50,num_classes=10):
        # compute the probability of the label class versus the maximum other
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func,GPU=True):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.GPU = GPU

        if loss_func == 'xent':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == 'cw':
            self.criterion = cwloss
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.criterion = nn.CrossEntropyLoss()  
    
    def perturb_true(self, x_nat, y,k):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        step_size = (self.step_size *self.num_steps / k)*np.random.uniform(1,10.0/k)
        #step_size =  self.step_size * self.num_steps / k if np.random.random()>0.5 else 2*self.epsilon
        if self.GPU:
            x_nat = x_nat.data.cpu().numpy()
            #self.model.cuda()
        else:
            x_nat=x_nat.data.numpy()
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(k):
            if self.GPU:
                x=torch.Tensor(x)
                x=x.cuda()
            else:
                x=torch.Tensor(x)
            x=Variable(x,requires_grad=True)
            self.model.zero_grad()
            output=self.model(x)
            loss=self.criterion(output,y)
            loss.backward()
            if self.GPU:
                grad=x.grad.data.cpu().numpy()
                x = x.data.cpu().numpy()
            else:
                grad = x.grad.data.numpy()
                x=x.data.numpy()

            x = np.add(x, step_size * (np.sign(grad)), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1)  # ensure valid pixel range
        return Variable(torch.Tensor(x).cuda(),volatile=False)

    def perturb_quad(self, x_nat, y,k,level):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        step_size = self.step_size *self.num_steps / k
        if self.GPU:
            x_nat = x_nat.data.cpu().numpy()
#            self.model.cuda()
        else:
            x_nat=x_nat.data.numpy()
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(k):
            if self.GPU:
                x=torch.Tensor(x)
                x=x.cuda()
            else:
                x=torch.Tensor(x)
            x=Variable(x,requires_grad=True)
            self.model.zero_grad()
            output=self.model(x)
            loss=self.criterion(output,y)
            loss.backward()
            if self.GPU:
                grad=x.grad.data.cpu().numpy()
                x = x.data.cpu().numpy()
            else:
                grad = x.grad.data.numpy()
                x=x.data.numpy()

            x = np.add(x, step_size * (np.sign(grad)), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1)  # ensure valid pixel range
        x_qt=quantization(x,level)
        return Variable(torch.Tensor(x).cuda(),volatile=False), Variable(torch.Tensor(x_qt).cuda(),volatile=False)
