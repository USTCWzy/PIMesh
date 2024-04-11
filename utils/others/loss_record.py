import numpy as np
import os
import time
import matplotlib.pyplot as plt
class updateLoss():
    def __init__(self, curr_model_path):
        super().__init__()
        self.loss_dict = {
            'train': {
                'total loss': []
            },
            'eval': {
                'total loss': []
            },
            'test': {
                'total loss': []
            }
        }
        self.start_time = 0
        self.end_time = 0
        self.curr_model_path = curr_model_path

    def start(self):
        self.start_time = time.time()

    def end(self):
        elapsed = time.time() - self.start_time
        time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                                 time.gmtime(elapsed))
        print('Processing the data took: {}'.format(time_msg))
        self.save()

    def update(self, loss, loss_dict, type='train'):

        if type not in self.loss_dict.keys():
            print('record type error!')
        else:
            for key in loss_dict:
                if key not in self.loss_dict[type]:
                    self.loss_dict[type][key] = []
                self.loss_dict[type][key].append(loss_dict[key])
            if loss is not None:
                self.loss_dict[type]['total loss'].append(loss)

    def save(self):
        np.savez(os.path.join(self.curr_model_path, 'loss.npz'), **self.loss_dict)

    def plot(self, epoch):
        os.makedirs(os.path.join(self.curr_model_path, 'loss_images'), exist_ok=True)

        for key in self.loss_dict:
            for loss_name in self.loss_dict[key]:
                if len((self.loss_dict[key][loss_name])) > 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.loss_dict[key][loss_name])
                    plt.savefig(os.path.join(self.curr_model_path, 'loss_images', f'{key}_{loss_name}.jpg'))
                    plt.close()
        self.save()

def print_loss(loss, loss_dict):
    lossstr = f'Losses: {np.round(loss, 1)} |'
    for key, val in loss_dict.items():
        lossstr += '| {}: {} '.format(key, np.round(val, 2))
    return lossstr