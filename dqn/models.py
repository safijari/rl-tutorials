import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.001):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)


class ConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.0001):
        assert len(obs_shape) == 3  # channel, height, and width
        super(ConvModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        # canonical input shape = 84x84
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]

        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(fc_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        conv_latent = self.conv_net(x/255.0)  # shape: (N, )
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))


if __name__ == "__main__":
    m = ConvModel((4, 84, 84), 4)
    tensor = torch.zeros((1, 4, 84, 84))
    print(m.forward(tensor))
