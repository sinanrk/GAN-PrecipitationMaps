
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

L.seed_everything(1729)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu_name}")
else:
    print("CUDA is not available. You may want to run on CPU.\n")


class Generator(nn.Module):
    def __init__(self, latent_dim=128, channels=1, image_size=(72, 144)):
        super(Generator, self).__init__()

        self.image_size = image_size

        self.linear = nn.Linear(latent_dim, 128 * (image_size[0] // 18) * (image_size[1] // 18))

        self.main = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(64, channels, kernel_size=18, stride=2, padding=(4,0), bias=False),
                                  nn.Sigmoid()
                                  )
    def forward(self, input):
        x = self.linear(input)
        x = x.view(x.size(0), 128, self.image_size[0] // 18, self.image_size[1] // 18) # (128,4,8)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(img_channels, 32, kernel_size=5, stride=2, padding=2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(0.4),
                                   nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(11520, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, img):
        validity = self.model(img)
        return validity

class TeleGAN(L.LightningModule):
    def __init__(self, latent_dim, img_shape, img_channels = 1):
        super(TeleGAN, self).__init__()
        self.automatic_optimization = False

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_channels = img_channels
        self.generator = Generator(latent_dim, channels = img_channels, image_size=img_shape)
        self.discriminator = Discriminator(img_channels=img_channels, img_size=img_shape)

        self.criterion = nn.BCELoss()

        self.g_loss_list = []
        self.d_loss_list = []

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def training_step(self, batch, batch_idx):

        opt_g, opt_d = self.optimizers()

        real_imgs = batch

        real_imgs = real_imgs.view(-1,1,*self.img_shape)


        valid = torch.ones(real_imgs.size(0), 1, device=self.device)#*0.9
        valid = valid.type_as(real_imgs)

        z = torch.randn(real_imgs.size(0), self.latent_dim, device=self.device)
        z = z.type_as(real_imgs)

        # Train Generator
        self.toggle_optimizer(opt_g)
        generated_imgs = self(z)
        g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.g_loss_list.append(g_loss)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        valid = torch.ones(real_imgs.size(0), 1, device=self.device)#*0.9
        valid = valid.type_as(real_imgs)

        fake = torch.zeros(real_imgs.size(0), 1, device=self.device)
        fake = fake.type_as(real_imgs)

        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.d_loss_list.append(d_loss)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        adv_loss = d_loss + g_loss
        self.log("adv_loss", adv_loss, prog_bar=True)

    def configure_optimizers(self):
        lr_g = 0.00002
        lr_d = 1*lr_g
        betas = (0.5, 0.999)

        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas)

        return [opt_g, opt_d], []

    def plot_generated_images(self):
        z = torch.randn(4, self.latent_dim, device = self.device)

        # log sampled images
        sample_imgs = self(z).cpu()
        fig, ax = plt.subplots(2,2, figsize=(20,10))
        ax = ax.flatten()
        for i in range(sample_imgs.shape[0]):
          ax[i].set(xticks=[], yticks=[], xlabel='', ylabel='')
          ax[i].imshow(sample_imgs[i].detach().view(*self.img_shape))
          ax[i].set_title(f'Sample {i + 1}')
        fig.suptitle(f'Generated Samples - Epoch: {self.current_epoch}', fontsize=16)
        fig.savefig(f'./checkpoints/Gen_epoch_{self.current_epoch}.png')
        plt.close(fig)

    def on_train_epoch_end(self):
        self.plot_generated_images()

class ERA5data():
    def __init__(self):
        super(ERA5data, self).__init__()

        self.dtindex = pd.date_range('1979-01', '2021-12', freq='M').strftime('%Y%m')

    def __len__(self):
        return len(self.dtindex)

    def __getitem__(self, idx):
        data = xr.open_dataset(f'./data/pcp_mon/gpcp_v02r03_monthly_d{self.dtindex[idx]}.nc')
        data = data.precip.values
        data = data/47.327435 # all values are in [0, 47.327435]
        return data

print('## STARTING')
era5pcp = ERA5data()
dataloader = DataLoader(era5pcp, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
print('## DATA LOADED')

sampleimg = next(iter(dataloader))[0][0]
plt.imshow(sampleimg)
print(f'Max = {sampleimg.max()}, Min = {sampleimg.min()}')

#Initialize GAN model
latent_dim = 128
img_shape = (72, 144)
gan = TeleGAN(latent_dim, img_shape)

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                      filename='Model_epoch_{epoch:02d}.pt',
                                      monitor='adv_loss',
                                      save_top_k=1,
                                      mode='min',
                                      )

trainer = L.Trainer(max_epochs=100,
                    accelerator='gpu',
                    callbacks=[checkpoint_callback])

# Train the GAN
trainer.fit(gan, dataloader)
