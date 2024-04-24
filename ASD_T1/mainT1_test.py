import torch
import torch.nn as nn
from ASD_T1.ResNet3D import resnet18
import logging
import sys
import datetime
import time
from tqdm import tqdm
from sklearn import metrics
from dataloader_nan import ExCustomDataset, data_split, count_parameters_in_MB
import pandas as pd
from torch.utils.data import DataLoader

def main():
    cv = 1
    subinfo_csv = '/public_bme/share/sMRI/Yuanbo//subinfo_ABIDE.csv'
    save_dir = '/home_data/home/wangyb12023/result/T1/'
    temp_csv = '/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv'
    # os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(subinfo_csv)
    temp = pd.read_csv(temp_csv)
    log_format = "%(asctime)s %(message)s"
    current_datetime = datetime.datetime.now()
    log_filename = "/home_data/home/wangyb12023/result/T1/test_output_model_0.0001_2{}.log".format(
        current_datetime.strftime("%Y%m%d_%H%M%S"))
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dataset = ExCustomDataset(temp=temp, df=df, transforms=True)
    seed = 123
    train_sampler, val_sampler, in_test_sampler, _, n_trains, n_vals, n_in_tests, _ = data_split(dataset,seed=seed)

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
    checkpoint = torch.load('/home_data/home/wangyb12023/result/T1/qualified_model_2_BASE_1_4_14_wd0.001.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()  # 将模型移到GPU上
    logging.info("Network structure: {}".format(model))
    logging.info("param size = %.3f MB", count_parameters_in_MB(model))

    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    auc_baseline = 0.60
    qualified = []

    model.eval()

    with torch.no_grad():
        test_total = 0.0
        test_correct = 0.0
        predicted_all = []
        test_y_all = []
        for batch_idx, batch in enumerate(tqdm(in_test_queue)):
            T1, labels = batch[0].cuda(), batch[1]
            out_feature, outputs = model(T1.contiguous())
            labels = labels.to(outputs.device)
            predicted = torch.max(outputs.data, 1)[1]
            test_total += labels.size(0)
            test_correct += predicted.eq(labels.data).sum()
            logging.info("predicted %s", predicted)
            labels = labels
            predicted = predicted
            predicted_all = predicted_all + predicted.tolist()
            test_y_all = test_y_all + labels.tolist()

        accuracy = 100. * test_correct / test_total
        sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
        spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
        auc = metrics.roc_auc_score(test_y_all, predicted_all)
        logging.info(
            '|test accuracy: %.3f |test sen: %.3f |test spe: %.3f |test auc: %.3f',
            accuracy, sens, spec, auc)

if __name__ == "__main__":
    main()
