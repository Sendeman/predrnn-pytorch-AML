import os
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr, filename='model.ckpt-'):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        if filename == 'model.ckpt-':
            filename += str(itr)
        
        checkpoint_path = os.path.join(self.configs.save_dir, filename)
        torch.save(stats, checkpoint_path)
            

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=self.configs.device)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        __, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def validate(self, frames, mask):
        self.network.eval()
        with torch.no_grad():
            frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
            mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
            __, loss = self.network(frames_tensor, mask_tensor)
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()