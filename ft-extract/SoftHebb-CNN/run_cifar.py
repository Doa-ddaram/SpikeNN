import math
import warnings
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from spikenn.dataset import SpikingDataset

"""
Single-file script to train a three-layer CNN on CIFAR10/100 using SoftHebb and save the output features encoded as spikes with first-spike coding.

The script is adapted from demo.py in the SoftHebb repo: https://github.com/NeuromorphicComputing/SoftHebb.
"""


class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 12,
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        # perform conv, obtain weighted input u \in [B, OC, OH, OW]
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        if self.training:
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            # Compute the winner neuron for each batch element and pixel
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            yx = F.conv2d(
                x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
                softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

            # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]
            self.weight.grad = delta_weight  # store in grad to be used with common optimizers

        return weighted_input


class DeepSoftHebb(nn.Module):
    def __init__(self, n_classes=10):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1,)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = SoftHebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65,)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = SoftHebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25,)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # block 4
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(24576, n_classes)
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(n_classes, 24576)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_features=False):
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))
        # block 4
        if return_features:
            return self.flatten(out)
        else:
            return self.classifier(self.dropout(self.flatten(out)))


class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    """
    Custom Learning Rate Scheduler for unsupervised training of SoftHebb Convolutional blocks.
    Difference between current neuron norm and theoretical converged norm (=1) scales the initial lr.
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]  # store initial lrs
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # difference between current neuron norm and theoretical converged norm (=1) scales the initial lr
                # initial_lr * |neuron_norm - 1| ** 0.5
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr


class TensorLRSGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, using a non-scalar (tensor) learning rate.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(-group['lr'] * d_p)
        return loss


class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

    
class ArrayCIFAR10(Dataset):

    def __init__(self, data_array, targets_array, device="cpu"):
        self.data = torch.tensor(data_array, dtype=torch.float, device=device).div_(255)
        self.data = self.data.permute(0, 3, 2, 1)
        self.targets = torch.tensor(targets_array, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        return img, target


def load_cifar10_numpy(path):
    width, height, channels = 32, 32, 3
    image_size = width * height * channels

    with open(path + '/X_train.bin', 'rb') as file:
        X_train = np.frombuffer(file.read(), dtype=np.uint8)
        num_samples = len(X_train) // image_size
        X_train = X_train.reshape((num_samples, width, height, channels))
    with open(path + '/y_train.bin', 'rb') as file:
        y_train = np.frombuffer(file.read(), dtype=np.uint8)
    
    X_val = None
    y_val = None
    if os.path.exists(path + '/X_val.bin'):
        with open(path + '/X_val.bin', 'rb') as file:
            X_val = np.frombuffer(file.read(), dtype=np.uint8)
            num_samples = len(X_val) // image_size
            X_val = X_val.reshape((num_samples, width, height, channels))
        with open(path + '/y_val.bin', 'rb') as file:
            y_val = np.frombuffer(file.read(), dtype=np.uint8)
    
    X_test = None
    y_test = None
    if os.path.exists(path + '/X_test.bin'):
        with open(path + '/X_test.bin', 'rb') as file:
            X_test = np.frombuffer(file.read(), dtype=np.uint8)
            num_samples = len(X_test) // image_size
            X_test = X_test.reshape((num_samples, width, height, channels))
        with open(path + '/y_test.bin', 'rb') as file:
            y_test = np.frombuffer(file.read(), dtype=np.uint8)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Main training loop CIFAR10
if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory")
    parser.add_argument("output_dir", type=str, help="Output data directory")
    parser.add_argument("--n_epochs", nargs="?", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10_numpy(args.input_dir)
    n_classes = len(np.unique(y_train))
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    device = torch.device('cuda:0')
    model = DeepSoftHebb(n_classes=n_classes)
    model.to(device)

    unsup_optimizer = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08, },  # SGD does descent, so set lr to negative
        {"params": model.conv2.parameters(), "lr": -0.005, },
        {"params": model.conv3.parameters(), "lr": -0.01, },
    ], lr=0)
    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)

    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    trainset = ArrayCIFAR10(X_train, y_train)
    unsup_trainloader = DataLoader(trainset, batch_size=10, shuffle=True)
    sup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, )

    if X_val is not None:
        valset = ArrayCIFAR10(X_val, y_val)
        valloader = DataLoader(valset, batch_size=1000, shuffle=False)
        
    if X_test is not None:
        testset = ArrayCIFAR10(X_test, y_test)
        testloader = DataLoader(testset, batch_size=1000, shuffle=False)


    # Unsupervised training with SoftHebb
    running_loss = 0.0
    for i, data in enumerate(unsup_trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        unsup_optimizer.zero_grad()

        # forward + update computation
        with torch.no_grad():
            outputs = model(inputs)

        # optimize
        unsup_optimizer.step()
        unsup_lr_scheduler.step()

    # Supervised training of classifier
    # set requires grad false and eval mode for all modules but classifier
    unsup_optimizer.zero_grad()
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv3.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.conv3.eval()
    model.bn1.eval()
    model.bn2.eval()
    model.bn3.eval()
    for epoch in range(n_epochs):
        model.classifier.train()
        model.dropout.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(sup_trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            sup_optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            sup_optimizer.step()

            # compute training statistics
            running_loss += loss.item()
            if epoch == n_epochs - 1:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        sup_lr_scheduler.step()
        # Evaluation on test set
        if epoch == n_epochs - 1:
            print(f'Train accuracy: {100 * correct // total} %')

            if X_test is not None:
                # on the test set
                model.eval()
                running_loss = 0.
                correct = 0
                total = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        # calculate outputs by running images through the network
                        outputs = model(images)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

                test_acc = 100 * correct / total
                with open(args.output_dir + "/ann_acc.txt", 'w') as file:
                    file.write(f"{test_acc}")
                print(f'Test accuracy: {test_acc} %')

    
    ### SAVE EXTRACTED FEATURES ###
    
    model.eval()
    train_out_data = []
    train_out_labels = []
    with torch.no_grad():
        for data in sup_trainloader:
            images, labels = data
            images = images.to(device)
            outputs = model(images, return_features=True)
            for o,l in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                o = 1 - o/o.max() # pixel to spike encoding
                train_out_data.append(o)
                train_out_labels.append(l)
    SpikingDataset.from_numpy(np.array(train_out_data), np.array(train_out_labels), max_time=1).save(args.output_dir + "/trainset.npy")
    
    if X_val is not None:
        val_out_data = []
        val_out_labels = []
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.to(device)
                outputs = model(images, return_features=True)
                for o,l in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                    o = 1 - o/o.max() # pixel to spike encoding
                    val_out_data.append(o)
                    val_out_labels.append(l)
        SpikingDataset.from_numpy(np.array(val_out_data), np.array(val_out_labels), max_time=1).save(args.output_dir + "/valset.npy")
    
    if X_test is not None:
        test_out_data = []
        test_out_labels = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                outputs = model(images, return_features=True)
                for o,l in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                    o = 1 - o/o.max() # pixel to spike encoding
                    test_out_data.append(o)
                    test_out_labels.append(l)
        SpikingDataset.from_numpy(np.array(test_out_data), np.array(test_out_labels), max_time=1).save(args.output_dir + "/testset.npy")