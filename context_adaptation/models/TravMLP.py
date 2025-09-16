import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPTravRegress(nn.Module):
    def __init__(self, feats, labels, config, params_dir=None):
        super().__init__()

        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.update_thresh = config['update_thresh']
        self.train_kernel = config['train_kernel']
        self.norm_alpha = config.get('norm_decay', 0.5) # smoothing factor for running stats

        self.mlp_params = None
        self.params_dir = params_dir

        if self.params_dir is not None:
            self.load_params(self.params_dir)

        self.in_mean = torch.mean(feats, dim=0)
        self.in_std = torch.std(feats, dim=0) + 1e-6
        self.out_mean = torch.mean(labels)
        self.out_std = torch.std(labels) + 1e-6

        self.feats = feats
        self.labels = labels
        self.init_size = len(self.feats)

        self.reinit_model()

    def build_mlp(self, in_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))   # regression output
        return nn.Sequential(*layers)

    def reinit_model(self):
        input_dim = self.feats.shape[1]
        self.mlp = self.build_mlp(input_dim, self.hidden_dim, self.num_layers).cuda()

        if self.mlp_params is not None:
            self.mlp.load_state_dict(self.mlp_params)

    def load_params(self, params_dir):
        self.mlp_params = torch.load(params_dir, weights_only=True)

    def update_running_stats(self, new_feats, new_labels):
        #TODO should i be doing this
        batch_mean = torch.mean(new_feats, dim=0)
        batch_std = torch.std(new_feats, dim=0) + 1e-6
        self.in_mean = self.norm_alpha * self.in_mean + (1 - self.norm_alpha) * batch_mean
        self.in_std = self.norm_alpha * self.in_std + (1 - self.norm_alpha) * batch_std

        out_mean = torch.mean(new_labels)
        out_std = torch.std(new_labels) + 1e-6
        self.out_mean = self.norm_alpha * self.out_mean + (1 - self.norm_alpha) * out_mean
        self.out_std = self.norm_alpha * self.out_std + (1 - self.norm_alpha) * out_std


    def train(self, feats, labels, num_steps = 1, lr = .00001):
        #TODO limit batch size if needed

        self.update_running_stats(feats, labels)

        feats = (feats - self.in_mean) / self.in_std
        labels = (labels - self.out_mean) / self.out_std

        self.feats = feats
        self.labels = labels

        if self.train_kernel:
            optimizer = torch.optim.SGD(self.mlp.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            optimizer.zero_grad()

            self.mlp.train()
            for i in range(num_steps):
                optimizer.zero_grad()
                pred = self.mlp(feats).squeeze(-1)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            self.mlp_params = self.mlp.state_dict()

    def forward(self, feats, cvar= None):
        self.mlp.eval()
        feats = (feats - self.in_mean) / self.in_std
        pred = self.mlp(feats).squeeze(-1)
        # de-normalize back
        pred = (pred * self.out_std) + self.out_mean

        return pred, None
