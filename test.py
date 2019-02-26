import time
import torch
import numpy as np
from net import ActionLSTM
from data import ActionDataset
import torch.utils.data

if __name__ == '__main__':
    net = ActionLSTM()
    net.load_state_dict(torch.load('model.pth', map_location='cpu'), strict=False)

    dataset = ActionDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    sample = iter(data_loader).__next__()

    time_period = 100
    for i in range(0, sample['values'][0].__len__() - time_period, 10):
        start_time = time.time()
        data_in_the_second = sample['values'][:, i:i + time_period, :]

        out = net(data_in_the_second)

        raw_pred = np.argmax(out.detach().numpy(), -1)[0]

        action_names = ['sitting', 'standing', 'walking']
        pred_action = action_names[raw_pred]
        gt_action = action_names[sample['raw_label'][0]]

        end_time = time.time()
        print('I guess you are {}, and actually you are {}. Time comsumed: {}'.format(pred_action, gt_action,
                                                                                      end_time - start_time))
