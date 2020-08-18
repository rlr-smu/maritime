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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import savefig
from datetime import datetime, timedelta
from torch.autograd import grad as torch_grad


os.makedirs("media", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--zone", type=int, default=11, help="zoneid of dataset")
parser.add_argument("--seed", type=int, default=7, help="seed to reproduce the results")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen generator samples")
parser.add_argument("--plot", type=bool, default=False, help="plot the graphs")
parser.add_argument("--savemp4", type=bool, default=False, help="savemp4?[plot has to be true for this to be true]")
parser.add_argument("--improved_wgan", type=bool, default=False, help="use improved wgan(gradient penalty)?")
parser.add_argument("--datafile", type=str, default='transittimes_cond_speed_z11.txt', help="real dataset file")

opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

savemp4 = opt.savemp4
if(opt.plot and savemp4):
	matplotlib.use("Agg")

point_shape = (1,) # to model a 1D distribution

cuda = True if torch.cuda.is_available() else False

def gradient_penalty(fake_data, real_data, discriminator):
    real_data = real_data.view(64,1,1,1)
    print("fake_data.shape ",fake_data.shape)
    print("real_data.shape ",real_data.shape)
    alpha = torch.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
    print("alpha ", alpha.shape)
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates.requires_grad = True
    print("interpolates ", interpolates.shape)
    disc_interpolates, _ = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def _gradient_penalty(generated_data, real_data, discriminator):
    '''
    Gradient penalty term for improved Wgan
    '''
    # print("generated_data.shape ",generated_data.shape)
    real_data = real_data.view(-1, 1)
    # print("real_data.shape ",real_data.shape)

    batch_size = real_data.size()[0]
    
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    # print("alpha.shape ",alpha.shape)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    # print("interpolated.shape", interpolated.shape)
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs= torch.ones(prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, data_point_size),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()

class timeDeltaDataset(Dataset):
    def __init__(self):
        with open(opt.datafile, 'r') as f:
            self.raw_samples = eval(f.read())
            self.min_sample = min(self.raw_samples)[0]
            self.max_sample = 10800 #max(self.raw_samples) , ignoring the other samples
            self.zero = (self.min_sample + self.max_sample) / 2

            print("DATASET")
            print("self.zero", self.zero)
            print("len(samples)", len(self.raw_samples))
            print("self.min_sample", self.min_sample)
            print("self.max_sample", self.max_sample)

            #normalise the samples to range -1 to 1
            self.samples = [((x[0]-self.zero)/(self.max_sample-self.zero), x[1]) for x in self.raw_samples]

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
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + 1, 16, normalize=False),
            *block(16, 32),
            # *block(256, 512),
            # *block(512, 1024),
            nn.Linear(32, int(np.prod(point_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen_input = torch.cat((z,labels), 1)
        output = self.model(gen_input)
        output = output.view(output.shape[0], *point_shape)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(point_shape)) + 1, 64), # +1 for the label
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(32, 32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, output, labels):
        
        output_flat = output.view(output.shape[0], -1)
        dis_input = torch.cat((output_flat,labels), 1)
        validity = self.model(dis_input)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataset = timeDeltaDataset()
real_pts = [dataset.raw_samples[x] for x in np.random.randint(0, len(dataset.raw_samples), size=1024)]
real_pts = [x for x in real_pts if x[0] <10800]
real_pt_labels = [real_pt[1] for real_pt in real_pts]
print("max(real_pts): ", max(real_pts))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)
# print(len(dataset))
# print(dataset[100])

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

fig, axes = plt.subplots(1, 2, figsize=(16,9))
ax2 = axes[1].twinx()
# ----------
#  Training
# ----------

batches_done = 0
dis_losses = []
gen_losses = []
kls = []
kls1 = []
kls2 = []
kls3 = []
minksdist = 1.0

def normalise(arr, minn, maxx):
    arrmin = min(arr)
    arrmax = max(arr)
    scale = (maxx-minn)/(arrmax-arrmin)
    return [(x-arrmin)*scale for x in arr]

speed_cuts = [0, 80, 120, 300]
speed_buk_ids = []
for cut in range(len(speed_cuts) - 1):
    speed_buk_ids.append([i for i,s in enumerate(real_pt_labels) if s >= speed_cuts[cut] and s < speed_cuts[cut+1] ])
    
def kl_distance(dist1, dist2):
    # print("len(dist1)", len(dist1))
    # print("len(dist2)", len(dist2))
    i1 = 0
    i2 = 0
    max_cum_dist = 0
    dist1 = sorted(dist1)
    dist2 = sorted(dist2)
    while i1 < len(dist1) and i2 < len(dist2):
        if dist1[i1] < dist2[i2]:
            i1 +=1
        else:
            i2 += 1
        max_cum_dist = max(abs(i1-i2), max_cum_dist)

    return max_cum_dist * 1.0 / 1024 

def cwgan_ks_distance(dist1, dist2, labels):
    # print("len(dist1)", len(dist1))
    # print("len(dist2)", len(dist2))
    ksdists = []
    for cut in speed_buk_ids:
        cdist1 = [dist1[id] for id in cut]
        cdist2 = [dist2[id] for id in cut]
        ksdists.append(kl_distance(cdist1, cdist2))
    return ksdists

# single Epoch Iteration, this is called from FuncAnimation for each epoch.
def update(frameid):
    global batches_done, dis_losses, gen_losses, minksdist
    axes[0].clear()
    axes[1].clear()
    ax2.clear()
    epoch = frameid
    if(epoch+1 == opt.n_epochs):
        print("minksdist:", minksdist)
    for i, (datapoint, speedlabels) in enumerate(dataloader):

        # Configure input
        real_pnts = Variable(datapoint.type(Tensor))
        real_labels = Variable(speedlabels.type(Tensor)).view(-1,1)
        # print("real_pnts.shape", real_pnts.shape)
        # print("real_labels.shape", real_labels.shape)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (datapoint.shape[0], opt.latent_dim))))

        # Generate a batch of points
        fake_pnts = generator(z, real_labels).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_pnts, real_labels)) + torch.mean(discriminator(fake_pnts, real_labels))
        if(opt.improved_wgan):
            penalty_weight = 5.0
            loss_D = loss_D + penalty_weight * _gradient_penalty(fake_pnts, real_pnts, discriminator)
        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        if(not opt.improved_wgan):
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of points
            gen_pnts = generator(z, real_labels)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_pnts, real_labels))

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
            # print(gen_pnts.view(-1).detach().numpy())
        batches_done += 1
    dis_losses.append(loss_D.item())
    gen_losses.append(loss_G.item())
    
    # get fake points
    latent = Variable(Tensor(np.random.normal(0, 1, (1023, opt.latent_dim))))
    test_labels = Variable(Tensor(np.array(real_pt_labels).reshape(-1,1))) #uniform speed distribution for testing
    generated_ts = generator(latent, test_labels)
    gen_pts = generated_ts.view(-1).detach().numpy()

    # UnNormalise gen_pts back to original 'time to cross' space
    gen_pts = [y*(dataset.max_sample-dataset.zero) + dataset.zero for y in gen_pts]
    # print(len(gen_pts))
    
    ksdists = cwgan_ks_distance(gen_pts, [real_pt[0] for real_pt in real_pts], [real_pt[1] for real_pt in real_pts])
    kls1.append(ksdists[0])
    kls2.append(ksdists[1])
    kls3.append(ksdists[2])
    avgksdist = sum(ksdists)/len(ksdists)
    kls.append(avgksdist)
    if(avgksdist < minksdist):
        minksdist = avgksdist
        # save weights
    
    # plot the generated points
    if(opt.plot):
        #=======right graph========#
        axes[1].set_title(opt.datafile+' batch_size:'+str(opt.batch_size)+' lr:'+str(opt.lr)+ ' '+('gradient penalty' if opt.improved_wgan else 'vanilla'))
        axes[1].set_xlabel('epochs')
        axes[1].set_ylabel('loss')
        diff = gen_losses[0] - dis_losses[0]
        axes[1].plot(dis_losses, color='red', label = 'Critic loss')
        axes[1].plot([gl-diff for gl in gen_losses], color='blue', label = 'Generator loss')
        # plot normalised KL distance
        
        
        ax2.plot(kls, color='orange', label = 'avgKS distance')
        ax2.plot(kls1, color='orange', label = 'KS distance low speed', alpha=0.3, linestyle='dashed')
        ax2.plot(kls2, color='orange', label = 'KS distance medium', alpha = 0.5, linestyle='dashed')
        ax2.plot(kls3, color='orange', label = 'KS distance high', alpha = 0.7, linestyle='dashed')
        axes[1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        #=======left graph========#
        hist_size = 120
        # axes[0].hist(gen_pts, bins=[x for  x in range(-2*hist_size,int(max(gen_pts))+hist_size,hist_size)], label='fake', alpha=0.5)
        # axes[0].hist(real_pts, bins=[x for  x in range(-2*hist_size,int(max(real_pts))+hist_size,hist_size)], label='real', alpha=0.5)
        # print("gen_pts.shape ", gen_pts)
        # print("real_pts.shape ", real_pts)
        axes[0].scatter(gen_pts, test_labels, label='fake')
        axes[0].scatter([real_pt[0] for real_pt in real_pts], [real_pt[1] for real_pt in real_pts], label='real')
        axes[0].set_xlabel('data_point(x)')
        axes[0].set_ylabel('number of datapoints(for histograms). critic(x) for line')

        #plot discriminator function (normalised to range 0 to 100).
        # gens_np = np.linspace(-1, 1, 201)
        # gens = gens_np*(dataset.max_sample-dataset.zero) + dataset.zero
        # gens_tens = Tensor(gens_np)
        # outs = discriminator(gens_tens, real_labels)
        # outs_np = outs.detach().numpy().reshape(-1)
        # outs_np = normalise(outs_np,0,100)

        # axes[0].plot(gens, outs_np,label= 'normalised discriminator '+ str(epoch))
        axes[0].set_title('Real Data vs. Generated Data')
        axes[0].legend(loc='upper left')
        fig.tight_layout() 
    return []

animation_fps = 240 
mp4_fps = 24
n_frames = opt.n_epochs
ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False, interval = 1000/animation_fps)	
if savemp4:
    ani.save('media/'+'cwgan_progress_'+str(opt.zone)+'_'+datetime.now().strftime('%Y%m%d_%H%M')+'.mp4', fps=mp4_fps, dpi=100)
else:
    plt.show()
