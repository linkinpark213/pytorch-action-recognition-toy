import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from net import ActionLSTM
from data import ActionDataset
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    batch_size = 8
    dataset = ActionDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = ActionLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    writer = SummaryWriter('./log')

    sample = iter(data_loader).__next__()

    global_step = 0
    for i in range(40):
        for j, sample in enumerate(data_loader):
            global_step += 1
            net.zero_grad()
            optimizer.zero_grad()

            out = net(sample['values'])
            loss_value = criterion(out, sample['label'])

            pred = np.argmax(out.detach().numpy(), -1)
            tags = sample['raw_label'].detach().numpy()
            accuracy = float(np.where(pred == tags, 1, 0).sum() / batch_size)

            print(
                'Epoch {}, Itertaion {}, Loss = {}, Accuracy = {:.2f} %'.format(i + 1, j + 1, loss_value, accuracy * 100))

            writer.add_scalar('loss', loss_value, global_step=global_step)
            writer.add_scalar('accuracy', accuracy, global_step=global_step)

            loss_value.backward()
            optimizer.step()

    writer.close()
    state_dict = net.state_dict()
    torch.save(state_dict, 'model.pth')
