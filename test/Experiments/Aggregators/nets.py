import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self,input_dim=784, latent_dim=20,hidden_dim=500):
        super(VAE,self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)
        
        self.input_dim = input_dim
    def encoder(self,x_in):
        x = F.relu(self.fc_e1(x_in.view(-1,self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus( self.fc_logvar(x) )
        return mean, logvar
    
    def decoder(self,z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = F.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)
    
    def sample_normal(self,mean,logvar):
        sd = torch.exp(logvar*0.5)
        e = Variable(torch.randn(sd.size())) # Sample from standard normal
        z = e.mul(sd).add_(mean)
        return z
    
    def forward(self,x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def test(self, input_data):
        running_loss = []
        for single_x in input_data:
            single_x = torch.tensor(single_x).float()

            x_in = Variable(single_x)
            x_out, z_mu, z_logvar = self.forward(x_in)
            # loss = self.criterion(x_out, x_in, z_mu, z_logvar)
            x_out = x_out.view(-1)
            x_in = x_in.view(-1)
            bce_loss = F.mse_loss(x_out, x_in, size_average=False)
            kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
            loss = (bce_loss + kld_loss)

            running_loss.append(loss.item())
        return running_loss