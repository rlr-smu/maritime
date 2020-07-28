import argparse
import os
import numpy as np
import math
import sys

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=1, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False




def gradient_penalty(fake_data, real_data, discriminator):
    alpha = torch.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates.requires_grad = True
    disc_interpolates, _ = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class timeDeltaDataset(Dataset):
    def __init__(self):
        with open('frodotransittimes.txt', 'r') as f:
            self.raw_samples = eval(f.read())
            self.min_sample = min(self.raw_samples)
            self.max_sample = 10800 #max(self.raw_samples) # TODO; how to normalise
            self.zero = (self.min_sample + self.max_sample) / 2
            print("self.zero", self.zero)
            print("self.min_sample", self.min_sample)
            print("self.max_sample", self.max_sample)
            self.samples = [(x-self.zero)/(self.max_sample-self.zero) for x in self.raw_samples]
            print(self.samples[:50])
            print(self.raw_samples[:50])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1, normalize=False),
            # *block(128, 256),
            # *block(256, 512),
            # *block(512, 1024),
            # nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 5),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(5, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataset = timeDeltaDataset()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)
print(len(dataset))
# print(dataset[100])

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

fig = plt.figure(figsize=(16,9))
plt_id =1
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (datapoint) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(datapoint.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (datapoint.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        # loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        # loss_D.backward()
        # optimizer_D.step()

        gp = gradient_penalty(fake_imgs.data, real_imgs.data, discriminator)
        lambda1 = 10
        errD = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + lambda1 * gp 
        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            # )

        if batches_done % opt.sample_interval == 0:

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )
            # print(gen_imgs.view(-1).detach().numpy())
            if(batches_done % (opt.sample_interval*12) ==0):
                latent = Variable(Tensor(np.random.normal(0, 1, (1024, opt.latent_dim))))
                generated_ts = generator(latent)
                gen_pts = generated_ts.view(-1).detach().numpy()
                # gen_pts = [int(x) + 2 for x in gen_pts]

                # UnNormalise gen_pts back to original 'time to cross' space
                gen_pts = [y*(dataset.max_sample-dataset.zero) + dataset.zero for y in gen_pts]
                print(len(gen_pts))
                
                x,y = 3,3
                splt = fig.add_subplot(x,y,plt_id)
                plt_id +=1
                splt.hist(gen_pts, bins=[x for  x in range(-120,int(max(gen_pts))+15,15)])
                plt.xlabel('seconds')
                if(plt_id> x*y):
                    plt_id = 1
                
            
                #plot discriminator function.
                # gens = [x*10 for x in range(-200,600)]
                gens_np = np.linspace(-2, 2, 401)
                gens = gens_np*(dataset.max_sample-dataset.zero) + dataset.zero
                gens_tens = Tensor(gens_np)
                
                outs = discriminator(gens_tens)
                
                outs_np = outs.detach().numpy().reshape(-1)
                
                plt.plot(gens, outs_np,label= 'discriminator'+ str(batches_done))
                # savefig('wgan_test.png' )

        batches_done += 1
plt.show()
