import random
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax
from queue import Queue
import numpy as np

def invert_sigmoid(x):
    return x.log() - (1-x).log()

class RandomCycleIter:
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
class ClassAwareSampler(Sampler):
    def __init__(self, labels, num_samples_cls=1, is_infinite=False):
        num_classes = len(np.unique(labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

        self.is_infinite = is_infinite
        
    def __iter__ (self):
        i = 0
        j = 0
        while i < self.num_samples or self.is_infinite:
            if j >= self.num_samples_cls:
                j = 0
        
            if j == 0:
                cls_data = self.data_iter_list[next(self.class_iter)]
                temp_tuple = next(zip(*[cls_data]*self.num_samples_cls))
                yield temp_tuple[j]
            else:
                yield temp_tuple[j]
            
            i += 1
            j += 1
    
    def __len__ (self):
        return self.num_samples
    
class SampleLearner(nn.Module):
    """
    Sample Learner
    """

    def __init__(self, num_classes, init_pow=0.):
        super(SampleLearner, self).__init__()
        self.num_classes = num_classes
        self.init_pow = init_pow

        self.fc = nn.Sequential(
            nn.Linear(num_classes, 1, bias=False),
            nn.Sigmoid()
        )

        # register intermediate variable between sampler and BP process
        self.sample_memory = Queue()

    def init_learner(self, img_num_per_cls):
        self.sample_per_class = (img_num_per_cls / img_num_per_cls.sum())
        self.fc.apply(self.init_weights_sampler)

    def init_weights_sampler(self, m):
        sample_per_class = 1. / self.sample_per_class.pow(self.init_pow)
        sample_per_class = sample_per_class / (sample_per_class.min() + sample_per_class.max())
        sample_per_class = invert_sigmoid(sample_per_class)
        if type(m) == nn.Linear:
            nn.init._no_grad_zero_(m.weight)
            with torch.no_grad():
                m.weight.add_(sample_per_class)

    def forward(self, onehot_targets, batch_size):
        """
        To be called in the sampler
        """
        weighted_onehot = self.fc(onehot_targets).squeeze(-1).unsqueeze(0)
        weighted_onehot = weighted_onehot.expand(batch_size, -1)
        weighted_onehot = gumbel_softmax(weighted_onehot.log(), hard=True, dim=-1) # B x N
        self.sample_memory.put(weighted_onehot.clone())
        return weighted_onehot.detach().nonzero()[:, 1] # B x 2 (batch_idx, sample_idx) => B x 1

    def forward_loss(self, x):
        """
        To be called when computing the meta loss
        """
        assert not self.sample_memory.empty()
        curr_sample = self.sample_memory.get()
        x = x.unsqueeze(-1) 
        try:
            x = x * curr_sample.detach() * curr_sample
        except:
            print(x.shape, curr_sample.shape)
        x = x.sum(-1).mean()
        return x

class MetaSampler(Sampler):
    def __init__(self, labels, batch_size, learner, device=None):
        self.device = device
        num_classes = len(np.unique(labels))
        self.num_samples = len(labels)

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.meta_learner = learner
        self.indices = list(range(len(labels)))

        targets = torch.tensor(labels).to(device)
        self.targets_onehot = nn.functional.one_hot(targets, num_classes).float().to(device)
        class_counts = torch.from_numpy(np.bincount(labels)).to(device)
        self.meta_learner.init_learner(class_counts.float())

    def __iter__(self):
        for _ in range(self.num_samples // self.batch_size):
            g = self.meta_learner(self.targets_onehot, self.batch_size)
            batch = [self.indices[i] for i in g]
            yield from iter(batch)

    def __len__(self):
        return self.num_samples // self.batch_size
