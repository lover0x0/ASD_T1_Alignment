from ASD_T1 import Models
import dataloader
import torch
import torch.nn as nn
from sklearn import metrics
import os
import numpy as np
from ASD_T1.ResNet3D import resnet18
import pandas as pd
from dataloader_nan import ExCustomDataset, data_split, count_parameters_in_MB
from torch.utils.data import DataLoader
import logging
import sys
from tqdm import tqdm
import time
import datetime
import io
from torch.utils.tensorboard import SummaryWriter

def main():
    cv = 1
    subinfo_csv = '/public_bme/share/sMRI/Yuanbo//subinfo_ABIDE.csv'
    save_dir = '/home_data/home/wangyb12023/result/T1/'
    temp_csv = '/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(subinfo_csv)
    temp = pd.read_csv(temp_csv)
    log_format = "%(asctime)s %(message)s"
    current_datetime = datetime.datetime.now()
    log_filename = "/home_data/home/wangyb12023/result/T1/output_{}.log".format(
        current_datetime.strftime("%Y%m%d_%H%M%S"))
    logging.basicConfig( stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dataset = ExCustomDataset(temp=temp, df=df, transforms=True)
    seed = 123
    train_sampler, val_sampler, in_test_sampler, _, n_trains, n_vals, n_in_tests, _ = data_split(dataset,seed=seed)
    # loader = dataloader.get_T1_dataloader()
    # # train_loader, val_loader, ratio, ratio1 = loader[0], loader[1], loader[2], loader[3]
    # train_loader, val_loader = loader[0], loader[1]
    num_workers = 8
    batch_size = 8
    train_queue = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    dataset_val = ExCustomDataset(temp=temp, df=df, transforms=False)

    val_queue = DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler,
                           shuffle=False, num_workers=num_workers, pin_memory=True)

    in_test_queue = DataLoader(dataset, batch_size=batch_size, sampler=in_test_sampler,
                               shuffle=False, num_workers=num_workers, pin_memory=True)
    model = resnet18()
    # model=None
    # model = torch.load("/home_data/home/wangyb12023/result/T1/qualified_model_2_BASE_1.pth")
    model = model.cuda()
    logging.info("Network structure: {}".format(model))
    logging.info("param size = %.3f MB", count_parameters_in_MB(model))
    # if ratio < 1:
    #     weight = torch.cuda.FloatTensor([1, 1 / ratio])
    # else:
    #     weight = torch.cuda.FloatTensor([ratio, 1])
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    auc_baseline = 0.60
    qualified = []
    with open('/home_data/home/wangyb12023/result/T1/results_BASE.txt', 'w') as f:
        f.write("Epoch\tTest Acc\tTest Sen\tTest Spe\tTest AUC\n")
        for epoch in tqdm(range(200), file=f):
            epoch_start_time = time.time()
            if epoch > 0:
                scheduler.step()
            # print(ratio1)
            model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, batch in enumerate(tqdm(train_queue)):
                    T1, labels = batch[0].cuda(), batch[1]
                    # T1 = T1.unsqueeze(1)
                    optimizer.zero_grad()
                    # print(fmri.size())
                    outputs = model(T1)
                    labels = labels.long().cuda()
                    # outputs = outputs.view(outputs.shape[0], -1).contiguous()
                    loss = loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # predicted = torch.max(outputs.data, 1)[1]
                    predicted = torch.argmax(outputs, dim=1)
                    sum_loss += loss.item()
                    # print("predicted", predicted)
                    total += labels.size(0)
                    # print("total", total)
                    correct += predicted.eq(labels.data).sum()
                    # print("correct", correct)
            # print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
            #               % (epoch + 1, sum_loss / (i + 1), 100. * correct / total))  # epoch平均损失和正确率
            logging.info('[epoch:%d] Loss: %.03f | Acc: %.3f%%',
                         epoch + 1, sum_loss / (i + 1), 100. * correct / total)

            model.eval()

            with torch.no_grad():
                test_total = 0.0
                test_correct = 0.0
                predicted_all = []
                test_y_all = []
                for batch_idx, batch in enumerate(val_queue):
                    T1, labels = batch[0].cuda(), batch[1]
                    #T1 = T1.unsqueeze(1)
                    outputs = model(T1.contiguous())
                    # outputs = outputs.view(outputs.shape[0], -1)
                    labels = labels.to(outputs.device)
                    predicted = torch.max(outputs.data, 1)[1]
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.data).sum()
                    # print("predicted: ", predicted)
                    logging.info("predicted %s", predicted)
                    labels = labels
                    predicted = predicted
                    predicted_all = predicted_all + predicted.tolist()
                    test_y_all = test_y_all + labels.tolist()
                # print('Val\'s accuracy is: %.3f%%' % (100. * test_correct / test_total))
                # print("Type of test_y_all:", type(test_y_all))
                # print("Data type of test_y_all:", test_y_all.dtype)
                # print("Shape of test_y_all:", test_y_all.shape)
                #
                # # 检查 predicted_all 的类型和格式
                # print("Type of predicted_all:", type(predicted_all))
                # print("Data type of predicted_all:", predicted_all.dtype)
                # print("Shape of predicted_all:", predicted_all.shape)
                # predicted_flat = np.array(predicted).flatten()  # 将 predicted 转换为一维数组
                # test_y_all_flat = np.array(test_y_all).flatten()  # 将 test_y_all 转换为一维数组
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
                    torch.save(state, f'/home_data/home/wangyb12023/result/T1/qualified_model_{len(qualified)}_BASE_1_4_14.pth')

                    # print('got one qualified model with | Test Acc:', accuracy,
                    #       '| Test Sen:', sens,
                    #       '| Test Spe:', spec)
                    # print('qualified', qualified)
                    logging.info('got one qualified model with | Test Acc: %.3f | Test Sen: %.3f | Test Spe: %.3f',
                                 accuracy, sens, spec)
                    logging.info('qualified %s', qualified)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()









