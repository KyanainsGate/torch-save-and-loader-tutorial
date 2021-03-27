import random
import os
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim

import myutils
from dataset_interface import DataInterface
from model import CNN, Net

# Set Seed
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch training')
    parser.add_argument('--datapath', '-dp', type=str, default="./data",
                        help='Data downloaded directory')
    parser.add_argument('--task', '-t', type=str, default="MNIST",
                        # parser.add_argument('--task', '-t', type=str, default="CIFAR10",
                        help='Classification dataset')
    parser.add_argument('--threading', '-thr', type=int, default=5,
                        help='CPU thread number for data loading')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch', '-b', type=float, default=512,
                        help='batch size (default: 0.01)')
    parser.add_argument('--logdir', '-log', type=str, default="./log",
                        help='log directory (default: ./log)')
    parser.add_argument('--span', '-s', type=int, default=1,
                        help='log directory (default: 0.01)')
    args = parser.parse_args()
    return args


def train_classification(net, dataloaders_dict: dict, criterion, optimizer, num_epochs, logpath, start_epoch=0):
    writer = SummaryWriter(logpath)  # Create Tensorboard  summary writer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Use device : ", device)

    # ネットワークをGPUへ
    net.to(device)
    if start_epoch != 0:
        # If saved model,
        print('Restart Option')
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        pass

    # ネットワークがある程度固定であれば、高速化させる
    # cudnn.benchmark = True
    cudnn.benchmark = False

    best_acc = 0

    acc_dict = {
        "train": 0,
        "val": 0,
    }

    # epochのループ
    for epoch in range(num_epochs + 1):
        if epoch < start_epoch:
            print('Skip until ' + str(start_epoch) + " (now:" + str(epoch) + ")")
            continue
            pass
        epoch_train_corrects = 0  # epochの正解数
        epoch_train_loss = 0.0
        epoch_val_corrects = 0  # epochの正解数
        epoch_val_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            correct = 0
            total = 0

            if ((epoch - start_epoch) == 0) and (phase == 'train'):
                print("As the first step, Optimization will NOT be done")
                pass
            elif ((epoch - start_epoch) == 1) and (phase == 'train'):
                print("Start optimization ...")
                pass

            # データローダーからminibatchずつ取り出すループ
            for inputs, labels in dataloaders_dict[phase]:
                if inputs.size()[0] == 1:
                    print('batch == 1 induce batch-norm error, so will be skipped')
                    continue

                if phase == 'train':
                    net.train()  # モデルを訓練モードに
                    optimizer.zero_grad()

                else:
                    net.eval()  # モデルを検証モードに

                # GPUが使えるならGPUにデータを送る
                imges = inputs.to(device)
                labels = labels.to(device)

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imges)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)  # ラベルを予測
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        if ((epoch - start_epoch) == 0):
                            pass
                        else:
                            loss.backward()  # 勾配の計算
                            optimizer.step()  # Update of Adam optimaizer
                        epoch_train_loss += loss.item() * inputs.size(0)  # Scalar値の抽出k関数==item()
                        epoch_train_corrects += torch.sum(preds == labels.data)
                    else:
                        epoch_val_loss += loss.item() * inputs.size(0)
                        epoch_val_corrects += torch.sum(preds == labels.data)
                        pass
                    pass
                pass  # Minibatch end
            acc_dict[phase] = float(correct / total) * 100.
            pass  # Phase end
        epoch_train_percent = acc_dict["train"]
        epoch_val_percent = acc_dict["val"]

        # epochごとのlossと正解率を表示
        print("Train : Loss: {:.2f} Acc: {:.2f}".format(epoch_train_loss, epoch_train_percent))
        print("Valid : Loss: {:.2f} Acc: {:.2f}".format(epoch_val_loss, epoch_val_percent))
        now_best = False
        if best_acc > epoch_val_percent:
            now_best = True
            best_acc = epoch_val_percent
            pass  # End save

        # Summary Writerへ書き込み
        writer.add_scalar("train/acc", epoch_train_percent, epoch)
        writer.add_scalar("val/acc", epoch_val_percent, epoch)

        # 最後のネットワークを保存する
        if epoch % 1 == 0:
            state = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            myutils.save_checkpoint(state=state, is_best=now_best, save_path=logpath,
                                    filename="checkpoint_" + str(epoch).zfill(4) + "_.pth")
            pass  # End
        pass  # End epoch
    pass


class TrainingIter(object):
    def __init__(self):
        args = parser()
        self.dataloaders_dict, self.log_path, self.model, \
        self.criterion, self.optimizer, self.max_epoch = self._parser2config(args)
        self.log = args.logdir
        self.weight_path = ""
        pass

    def _parser2config(self, args_: argparse.Namespace):
        # max epoch
        max_epoch = args_.epochs

        # Call training data
        data_path = args_.datapath
        task = args_.task
        batch_size = args_.batch
        num_works = args_.threading
        dls = DataInterface(root_path=data_path, dataset=task, batch_size=batch_size, num_works=num_works)
        dataloaders_dict = dls.dataloader_dict()

        # Log directory creation
        log = args_.logdir
        log_path = myutils.timestamped_path(log)

        # Def model
        net = Net()

        # Def optimizer
        criterion = torch.nn.CrossEntropyLoss()

        # Set optimizer
        lr = args_.lr
        optimizer = optim.Adam(net.parameters(), lr=lr)

        return dataloaders_dict, log_path, net, criterion, optimizer, max_epoch

    def run(self):
        train_classification(net=self.model, dataloaders_dict=self.dataloaders_dict, criterion=self.criterion,
                             optimizer=self.optimizer, logpath=self.log_path, num_epochs=self.max_epoch)
        pass

    def restart(self, load_logpath):
        model_, optimizer_, epoch_ = myutils.load_checkpoint(model=self.model, optimizer=self.optimizer,
                                                             filename=load_logpath, )
        train_classification(net=model_, dataloaders_dict=self.dataloaders_dict, criterion=self.criterion,
                             num_epochs=self.max_epoch,
                             optimizer=optimizer_, logpath=self.log_path, start_epoch=epoch_)
        pass

    def get_load_weight(self, yd: str, epoch: int, hms="", zero_fill=4):
        date_ = myutils.get_timestamped_weight_path(yd=yd, hms=hms, epoch=epoch, zero_fill=zero_fill)
        self.weight_path = os.path.join(self.log, date_)
        return self.weight_path
