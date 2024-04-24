import Models
import dataloader
import torch
import torch.nn as nn
from sklearn import metrics
import os
import pandas as pd
from dataloader_nan import ExCustomDataset_fMRI_MAHGCN, data_split, count_parameters_in_MB
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import sys
from tqdm import tqdm
import time
import datetime
import io
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def main():
    cv = 1
    subinfo_csv = '/public_bme/share/sMRI/Yuanbo//subinfo_ABIDE.csv'
    save_dir = '/home_data/home/wangyb12023/result/fMRI/'
    temp_csv = '/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(subinfo_csv)
    temp = pd.read_csv(temp_csv)
    log_format = "%(asctime)s %(message)s"
    current_datetime = datetime.datetime.now()
    log_filename = "/home_data/home/wangyb12023/result/fMRI/MAHGCN_output_{}.log".format(
        current_datetime.strftime("%Y%m%d_%H%M%S"))
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dataset = ExCustomDataset_fMRI_MAHGCN(temp=temp, df=df, transforms=False)
    seed = 123
    train_sampler, val_sampler, in_test_sampler, _, n_trains, n_vals, n_in_tests, _ = data_split(dataset, seed=seed)
    num_workers = 8
    batch_size = 32
    train_queue = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    dataset_val = ExCustomDataset_fMRI_MAHGCN(temp=temp, df=df, transforms=False)

    val_queue = DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler,
                           shuffle=False, num_workers=num_workers, pin_memory=True)

    in_test_queue = DataLoader(dataset, batch_size=batch_size, sampler=in_test_sampler,
                               shuffle=False, num_workers=num_workers, pin_memory=True)
    # train_loader, val_loader, ratio, ratio1 = loader[0], loader[1], loader[2], loader[3]
    ROInum = 500
    layer = 5
    model = Models.MAHGCNNET(ROInum, layer)
    model = model.cuda()
    logging.info("Network structure: {}".format(model))
    logging.info("param size = %.3f MB", count_parameters_in_MB(model))
    # if ratio < 1:
    #     weight = torch.cuda.FloatTensor([1, 1 / ratio])
    # else:
    #     weight = torch.cuda.FloatTensor([ratio, 1])
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=8e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    auc_baseline = 0.60
    qualified = []
    log_dir = os.path.join(save_dir, "Tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    dash_writer = SummaryWriter(log_dir)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    with open('/home_data/home/wangyb12023/result/fMRI/results_2.txt', 'w') as f:
        f.write("Epoch\tTest Acc\tTest Sen\tTest Spe\tTest AUC\n")
        for epoch in tqdm(range(150), file=f):
            epoch_start_time = time.time()
            if epoch > 0:
                scheduler.step()
            # print(ratio1)
            model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, batch in enumerate(tqdm(train_queue)):
                    fmri, labels =batch[0], batch[1]
                    g1, g2, g3, g4, g5 = fmri[0].cuda(), fmri[1].cuda(), fmri[2].cuda(), fmri[3].cuda(), fmri[4].cuda()
                    optimizer.zero_grad()
                    # print(fmri.size())
                    out_feature, outputs = model(g1,g2,g3,g4,g5)
                    labels = labels.long().cuda()
                    loss = loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    predicted = torch.max(outputs.data, 1)[1]
                    sum_loss += loss.item()
                    # print("predicted", predicted)
                    total += labels.size(0)
                    # print("total", total)
                    correct += predicted.eq(labels.data).sum()
                    # print("correct", correct)
            train_loss = sum_loss / (i + 1)
            train_accuracy = 100. * correct / total
            train_losses.append(sum_loss/(i + 1))
            train_accuracies.append((100. * correct / total))
            dash_writer.add_scalar("Train/Loss", train_loss, epoch)
            dash_writer.add_scalar("Train/Accuracy", train_accuracy, epoch)

            logging.info('[epoch:%d] Train Loss: %.03f | Train Accuracy: %.3f%%',
                         epoch + 1, train_loss, train_accuracy)

            model.eval()

            with torch.no_grad():
                test_total = 0.0
                test_correct = 0.0
                predicted_all = []
                test_y_all = []
                for batch_idx, batch in enumerate(val_queue):
                    fmri, labels = batch[0], batch[1]
                    g1, g2, g3, g4, g5 = fmri[0].cuda(), fmri[1].cuda(), fmri[2].cuda(), fmri[3].cuda(), fmri[4].cuda()
                    out_feature, outputs = model(g1, g2, g3, g4, g5)
                    labels = labels.to(outputs.device)
                    predicted = torch.max(outputs.data, 1)[1]
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.data).sum()
                    # print("predicted: ", predicted)
                    logging.info("predicted %s", predicted)
                    labels = labels
                    predicted_all = predicted_all + predicted.tolist()
                    test_y_all = test_y_all + labels.tolist()

                val_accuracy = 100. * test_correct / test_total
                val_accuracies.append(val_accuracy)

                # 将验证损失和准确率写入TensorBoard
                dash_writer.add_scalar("Val/Accuracy", val_accuracy, epoch)
                # print('Val\'s accuracy is: %.3f%%' % (100. * test_correct / test_total))
                accuracy = 100. * test_correct / test_total
                sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)

                spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)

                auc = metrics.roc_auc_score(test_y_all, predicted_all)
                # print('|test accuracy:', accuracy,
                #       '|test sen:', sens,
                #       '|test spe:', spec,
                #       '|test auc:', auc,
                #       )
                # epoch_start_time = 0
                logging.info(
                    '|test accuracy: %.3f |test sen: %.3f |test spe: %.3f |test auc: %.3f |Time Taken: %d sec *******',
                    accuracy, sens, spec, auc, time.time() - epoch_start_time)

                # if auc >= auc_baseline and sens > 0.5 and spec > 0.5:
                #     auc_baseline = auc
                #     best = auc
                #     qualified.append([accuracy, sens, spec, auc])
                #     a = {'model_state_dict': model.state_dict(), 'accuracy': accuracy, 'sen': sens, 'spec': spec,
                #          'auc': auc}
                #     torch.save(a, '/home_data/home/wangyb12023/result/fMRI/1.pth')
                #     print('got one model with |test accuracy:', accuracy,
                #           '|test sen:', sens,
                #           '|test spe:', spec,
                #           )
                #
                #     qualified.append([accuracy, sens, spec, auc])
                #     print('qualified', qualified)
                if accuracy > 60 and sens > 0.6 and spec > 0.6 and auc >= auc_baseline:
                    qualified.append([accuracy, sens, spec, auc])
                    f.write(f"epoch: {epoch + 1} | Qualified: {accuracy:.3f} | {sens:.3f} | {spec:.3f} | {auc:.3f}\n")

                    # 保存模型
                    state = {
                        'model_state_dict': model.state_dict(),
                        'accuracy': accuracy,
                        'sen': sens,
                        'spec': spec,
                        'auc': auc
                    }
                    torch.save(state, f'/home_data/home/wangyb12023/result/fMRI/qualified_model_{len(qualified)}_MAHGCN_6_0.0007.pth')

                    # print('got one qualified model with | Test Acc:', accuracy,
                    #       '| Test Sen:', sens,
                    #       '| Test Spe:', spec)
                    # print('qualified', qualified)
                    logging.info('got one qualified model with | Test Acc: %.3f | Test Sen: %.3f | Test Spe: %.3f',
                                 accuracy, sens, spec)
                    logging.info('qualified %s', qualified)
    #                 train_loss = torch.tensor(train_loss)
    #                 train_accuracy = torch.tensor(train_accuracy)
    #                 val_accuracy = torch.tensor(val_accuracy)
    # train_losses_cpu = [loss.cpu().item() for loss in train_losses]
    # train_accuracies_cpu = [accuracy.cpu().item() for accuracy in train_accuracies]
    # val_accuracies_cpu = [accuracy.cpu().item() for accuracy in val_accuracies]

# 在绘图之前，添加一个正确的epoch范围
    epochs = range(150)
    # 在绘图之前，将 CUDA 张量移动到 CPU 并转换为 NumPy 数组
    train_losses_tensor = torch.tensor(train_losses)
    train_accuracies_tensor = torch.tensor(train_accuracies)
    val_accuracies_tensor = torch.tensor(val_accuracies)
    print(train_losses_tensor, train_accuracies_tensor)
    print(val_accuracies_tensor)
    logging.info('train_loss', train_losses)
    logging.info('val_accuracy', val_accuracies)
    logging.info('train_accuracy', train_accuracies)
    # 绘制结果
    # epochs = range(1, len(train_losses) + 1)

    # 绘制结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_tensor, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies_tensor, label='Train Accuracy')
    plt.plot(epochs, val_accuracies_tensor, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
