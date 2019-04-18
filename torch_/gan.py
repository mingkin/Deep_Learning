# -*- coding: utf-8 -*-

"""
Author: kingming

File: gan.py

Time: 2019/4/18 下午8:13

License: (C) Copyright 2018, xxx Corporation Limited.

"""

import torch
from  torch import nn, optim,autograd
import numpy as np
import visdom
import random
from matplotlib import pyplot as plt


h_dim =400
batch_size =512
viz = visdom.Visdom()
'visdom 可视化要先运行   python -m visdom.server'

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            #z:[b, 2] =>[b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return  output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # z:[b, 2] =>[b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return  output.view(-1)



def data_generator():
    '8-gaussian mixture models'
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1./np.sqrt(2), 1./np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale*x, scale*y) for x,y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2)*0.02
            center = random.choice(centers)
            point[0] +=center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /=1.414
        yield dataset


def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda() # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()


    # draw samples
    with torch.no_grad():
        z = torch.randn(batch_size, 2).cuda() # [b, 2]
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))




def main():
    torch.manual_seed(23)
    np.random.seed(23)
    data_iter = data_generator()


    G = Generator()
    D = Discriminator()
    print(G,D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    for epoch in range(5000):
        for _ in range(5):
            xr = next(data_iter)
            xr = torch.from_numpy(xr)
            predr = D(xr)
            loss_r = -predr.mean()

            z = torch.randn(batch_size, 2)
            xf = G(z).detach()
            predf = D(xf)
            loss_f = predf.mean()

            loss_D = loss_r + loss_f

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        z = torch.randn(batch_size, 2)
        xf = G(z)
        predf = D(xf)
        loss_G = -predf.mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch %100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print(loss_D.item(),loss_G.item())

if __name__ == "__main__":
    main()



