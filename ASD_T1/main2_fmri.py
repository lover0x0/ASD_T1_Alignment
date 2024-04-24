import Models
import dataloader
import torch
import torch.nn as nn
from sklearn import metrics
import os




def main():
    save_dir = '/home_data/home/wangyb12023/result/fMRI/'
    os.makedirs(save_dir, exist_ok=True)
    loader = dataloader.get_dataloader('fMRI')
    train_loader, val_loader, ratio, ratio1 = loader[0], loader[1], loader[2], loader[3]
    model = Models.GCNNET()
    model = model.cuda()
    if ratio < 1:
        weight = torch.cuda.FloatTensor([1, 1 / ratio])
    else:
        weight = torch.cuda.FloatTensor([ratio, 1])
    loss_func = nn.CrossEntropyLoss(weight).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-1)

    auc_baseline = 0.60
    qualified = []
    with open('/home_data/home/wangyb12023/result/fMRI/results.txt', 'w') as f:
        f.write("Epoch\tTest Acc\tTest Sen\tTest Spe\tTest AUC\n")
    for epoch in range(200):
        print(ratio1)
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, batch in enumerate(train_loader):
                fmri, labels = batch["fMRI"].cuda(), batch["label"].cuda()
                optimizer.zero_grad()
                # print(fmri.size())
                outputs = model(fmri)
                labels = labels.long()
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                predicted = torch.max(outputs.data, 1)[1]

                sum_loss += loss.item()
                # print("predicted", predicted)
                total += labels.size(0)
                # print("total", total)
                correct += predicted.eq(labels.data).cpu().sum()
                # print("correct", correct)
        print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, sum_loss / (i + 1), 100. * correct / total))  # epoch平均损失和正确率

        model.eval()

        with torch.no_grad():
            test_total = 0.0
            test_correct = 0.0
            predicted_all = []
            test_y_all = []
            for batch_idx, batch in enumerate(val_loader):
                fmri, labels = batch["fMRI"].cuda(), batch["label"].cuda()
                outputs = model(fmri)
                predicted = torch.max(outputs.data, 1)[1]
                test_total += labels.size(0)
                test_correct += predicted.eq(labels.data).cpu().sum()
                print("predicted: ", predicted)
                labels = labels.cpu()
                predicted = predicted.cpu()
                predicted_all = predicted_all + predicted.tolist()
                test_y_all = test_y_all + labels.tolist()
            # print('Val\'s accuracy is: %.3f%%' % (100. * test_correct / test_total))
            accuracy = 100. * test_correct / test_total
            sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)

            spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)

            auc = metrics.roc_auc_score(test_y_all, predicted_all)
            print('|test accuracy:', accuracy,
                  '|test sen:', sens,
                  '|test spe:', spec,
                  '|test auc:', auc,
                  )

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
                torch.save(state, f'/home_data/home/wangyb12023/result/fMRI/qualified_model_{len(qualified)}.pth')

                print('got one qualified model with | Test Acc:', accuracy,
                      '| Test Sen:', sens,
                      '| Test Spe:', spec)
                print('qualified', qualified)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()









