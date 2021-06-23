import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.nn import PyroSample, PyroModule
import pyro.distributions as dists 


class Lenet5Deterministic(nn.Module):
    def __init__(self):
        super(Lenet5Deterministic, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        ## Kernel size changed to 4 to fit data (original Lenet was designed for 32x32x1 imgs)
        self.conv3 = nn.Conv2d(16, 120, 4, stride=1)

        self.fc1 = nn.Linear(120, 84)  
        self.fc2 = nn.Linear(84, 1)
        
        self.activation = nn.Tanh()
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        #print(x.shape)
        x = F.avg_pool2d(self.activation(self.conv1(x)), 2, stride=2)
        #print(x.shape)
        x = F.avg_pool2d(self.activation(self.conv2(x)), 2, stride=2)
        #print(x.shape)
        x = self.activation(self.conv3(x))
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #print(x.shape)
        x = self.activation(self.fc1(x))
        #print(x.shape)
                
        x = self.output(self.fc2(x))
        #x = torch.tanh(self.fc2(x))
        
        x = x.squeeze(1)

        return x
    
class Lenet5Bayesian(PyroModule):
    def __init__(self, prior_scale=2., isFEBayesian=True, prior_scaleFE = 2.):
        super(Lenet5Bayesian, self).__init__()
        prior = dists.Normal(0, prior_scale)
        priorFE = dists.Normal(0, prior_scaleFE)
        
        if isFEBayesian:
            self.conv1 = PyroModule[nn.Conv2d](1, 6, 5, stride=1)
            self.conv1.weight = PyroSample(priorFE.expand([6, 1, 5, 5]).to_event(4))
            self.conv1.bias = PyroSample(priorFE.expand([6]).to_event(1))

            self.conv2 = PyroModule[nn.Conv2d](6, 16, 5, stride=1)   
            self.conv2.weight = PyroSample(priorFE.expand([16, 6, 5, 5]).to_event(4))
            self.conv2.bias = PyroSample(priorFE.expand([16]).to_event(1))

            self.conv3 = PyroModule[nn.Conv2d](16, 120, 4, stride=1)
            self.conv3.weight = PyroSample(priorFE.expand([120, 16, 4, 4]).to_event(4))
            self.conv3.bias = PyroSample(priorFE.expand([120]).to_event(1))
        else: 
            self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
            self.conv2 = nn.Conv2d(6, 16, 5, stride=1)   
            ## Kernel size changed to 4 to fit data (original Lenet was designed for 32x32x1 imgs)
            self.conv3 = nn.Conv2d(16, 120, 4, stride=1)
        
        self.fc1 = PyroModule[nn.Linear](120, 84)  
        self.fc1.weight = PyroSample(prior.expand([84, 120]).to_event(2))
        self.fc1.bias = PyroSample(prior.expand([84]).to_event(1))

        
        self.fc2 = PyroModule[nn.Linear](84, 1)
        self.fc2.weight = PyroSample(prior.expand([1, 84]).to_event(2))
        self.fc2.bias = PyroSample(prior.expand([1]).to_event(1))
        
        self.output = nn.Sigmoid()
        self.activation = nn.Tanh()

        
    def forward(self, x, y=None):
        x = F.avg_pool2d(self.activation(self.conv1(x)), 2, stride=2)
        x = F.avg_pool2d(self.activation(self.conv2(x)), 2, stride=2)
        x = self.activation(self.conv3(x))
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        x = self.activation(self.fc1(x))
        x = self.output(self.fc2(x))
        #x = self.fc2(x)
        f = x.squeeze(1)

        #print(f.shape)
        with pyro.plate("data", f.shape[0]):
            loc = pyro.deterministic("k", f, event_dim=0)   
            obs = pyro.sample("obs", dists.Bernoulli(probs=loc), obs=y) #likelihood
        
        return f