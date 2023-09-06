import torch
from torch import nn
from sklearn.model_selection import KFold
from dataset import *
from torch.utils.data import DataLoader
from metrics import compute_cls_metrics, compute_reg_metrics

from model import PermuteDDS
from dgllife.utils import EarlyStopping, Meter, RandomSplitter
from prettytable import PrettyTable
from utils import *
from train import run_a_train_epoch, run_an_eval_epoch
from optimizer import Adan


os.environ["TOKENIZERS_PARALLELISM"] = 'false'

if __name__ == '__main__':
    dataset_name = 'ONEIL'  # ONEIL or ALMANAC
    task_name = 'regression'  # regression or classification

    # random, leave_cline, leave_comb
    cv_mode_ls = [1, 2, 3]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512
    num_epochs = 500

    seed = 42
    lr = 5e-3
    setup_seed(seed)

    synergy, drug2smile, drug2hastt, drug2map4, drug2maccs, cline2gene, cline2mutation, gene_dim, mutation_dim = get_data(
        dataset_name)

    for cv_mode in cv_mode_ls:
        if cv_mode == 1:
            result_path = os.getcwd() + '/result/' + dataset_name + '-' + task_name + '/random_split/'
            mkdir(result_path)
        elif cv_mode == 2:
            result_path = os.getcwd() + '/result/' + dataset_name + '-' + task_name + '/leave_cline/'
            mkdir(result_path)
        else:
            result_path = os.getcwd() + '/result/' + dataset_name + '-' + task_name + '/leave_comb/'
            mkdir(result_path)

        if task_name == 'classification':
            # k-fold val
            val_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
            # k-fold test
            t_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
            # 独立测试结果
            ind_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
        else:
            val_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])
            t_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])
            ind_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])

        ind_tables.float_format = '.3'
        val_tables.float_format = '.3'
        t_tables.float_format = '.3'

        config_drug_feature = init_featurizer()
        # synergy_data used for train-val-test
        synergy_data, independent_test = data_split(synergy, test_size=0.1, rd_seed=seed)
        # [drug1, drug2, gene, mutation, label]
        independent_test = process_data(independent_test, drug2smile, cline2gene, cline2mutation, task_name=task_name)

        independent_ds = FPDataset(independent_test[:, 0], independent_test[:, 1], independent_test[:, 2],
                                   independent_test[:, 3], independent_test[:, 4], drug2hastt, drug2map4, drug2maccs)

        independent_loader = DataLoader(independent_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        independent_path = result_path + 'independent_test/'
        mkdir(independent_path)

        if cv_mode == 1:  # random split
            cv_data = synergy_data
        elif cv_mode == 2:  # leave_cline
            cv_data = np.unique(synergy_data[:, 2])
        else:  # leave_comb
            cv_data = np.unique(np.vstack([synergy_data[:, 0], synergy_data[:, 1]]), axis=1).T

        # 记录最终的五次平均
        test_mean = np.array([0., 0., 0., 0.])
        ind_mean = np.array([0., 0., 0., 0.])
        # leave_out操作在测试集上进行
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for test_fold, (cv_index, test_index) in enumerate(kf.split(cv_data)):
            if cv_mode == 1:
                synergy_cv, synergy_test = cv_data[cv_index], cv_data[test_index]
            elif cv_mode == 2:
                cline_cv, cline_test = cv_data[cv_index], cv_data[test_index]
                synergy_cv = np.array([i for i in synergy_data if i[2] in cline_cv])

                synergy_test = np.array([i for i in synergy_data if i[2] in cline_test])
            else:
                pair_cv, pair_test = cv_data[cv_index], cv_data[test_index]
                print(pair_cv)
                print(pair_test)
                synergy_cv = np.array(
                    [j for i in pair_cv for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_test = np.array(
                    [j for i in pair_test for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])

            synergy_cv = process_data(synergy_cv, drug2smile, cline2gene, cline2mutation, task_name=task_name)
            synergy_test = process_data(synergy_test, drug2smile, cline2gene, cline2mutation, task_name=task_name)

            synergy_train, synergy_validation = data_split(synergy_cv, test_size=0.1, rd_seed=seed)

            trn_ds = FPDataset(synergy_train[:, 0], synergy_train[:, 1], synergy_train[:, 2], synergy_train[:, 3],
                               synergy_train[:, 4], drug2hastt, drug2map4, drug2maccs)
            val_ds = FPDataset(synergy_validation[:, 0], synergy_validation[:, 1], synergy_validation[:, 2],
                               synergy_validation[:, 3],
                               synergy_validation[:, 4], drug2hastt, drug2map4, drug2maccs)
            test_ds = FPDataset(synergy_test[:, 0], synergy_test[:, 1], synergy_test[:, 2], synergy_test[:, 3],
                                synergy_test[:, 4], drug2hastt, drug2map4, drug2maccs)

            train_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
            valid_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

            model = PermuteDDS(gene_dim, mutation_dim, d_model=300).to(device)
            optimizer = Adan(
                model.parameters(),
                lr=lr,  # learning rate (can be much higher than Adam, up to 5-10x)
                betas=(0.02, 0.08, 0.01),
                # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
                weight_decay=0.02  # weight decay 0.02 is optimal per author
            )
            # optimizer = torch.optim.AdamW(model.parameters(), lr )

            # train_size = len(train_loader)
            # total_steps = (train_size // BATCH_SIZE) * num_epochs if train_size % BATCH_SIZE == 0 else (train_size // BATCH_SIZE + 1) * num_epochs
            # # # cosine+warmup
            # scheduler = get_cosine_schedule_with_warmup(optimizer,
            #                                             num_warmup_steps=total_steps * 0.1,
            #                                             num_training_steps=total_steps)
            # scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(train_loader))

            model_path = result_path + 'fold_' + str(test_fold) + '_model.pth'
            stopper = EarlyStopping(mode='lower', patience=25, filename=model_path)
            # stopper = EarlyStopping(mode='lower', patience=25, filename='models/cross-transV2' + task_name + dataset_name)

            if task_name == 'classification':
                loss_criterion = nn.BCEWithLogitsLoss()
            else:
                loss_criterion = nn.MSELoss()
                # loss_criterion = CCCLoss()
                # loss_criterion = ValidationRMSELoss()

            for epoch in range(num_epochs):
                # Train
                run_a_train_epoch(device, epoch, model, train_loader, loss_criterion, optimizer, scheduler=None)
                # Validation and early stop
                val_pred, val_true, val_loss = run_an_eval_epoch(device, model, valid_loader, task_name, loss_criterion)

                if task_name == 'classification':
                    e_tables = PrettyTable(['Epoch', 'AUC', 'AUPR', 'F1', 'ACC'])
                    auc, aupr, f1_score, acc = compute_cls_metrics(val_true, val_pred)
                    row = [epoch, auc, aupr, f1_score, acc]
                else:
                    e_tables = PrettyTable(['Epoch', 'RMSE', 'R2', 'Pearson r', 'MAE'])
                    rmse, r2, r, mae = compute_reg_metrics(val_true, val_pred)
                    row = [epoch, rmse, r2, r, mae]

                early_stop = stopper.step(val_loss, model)
                e_tables.float_format = '.3'

                e_tables.add_row(row)
                print(e_tables)
                if early_stop:
                    break
            stopper.load_checkpoint(model)

            # 最佳验证集结果
            print('Val Best----------------')
            val_pred, val_true, val_loss = run_an_eval_epoch(device, model, valid_loader, task_name, loss_criterion)

            if task_name == 'classification':
                auc, aupr, f1_score, acc = compute_cls_metrics(val_true, val_pred)
                row = ['val', auc, aupr, f1_score, acc]
            else:
                rmse, r2, r, mae = compute_reg_metrics(val_true, val_pred)
                row = ['val', rmse, r2, r, mae]

            val_tables.add_row(row)
            print(val_tables)

            print(
                '---------------------------------------------------Test---------------------------------------------------')
            test_pred, test_y, test_loss = run_an_eval_epoch(device, model, test_loader, task_name, loss_criterion)
            np.savetxt(result_path + 'fold_' + str(test_fold) + '_test_y_true.txt', test_y)
            np.savetxt(result_path + 'fold_' + str(test_fold) + '_pred.txt', test_pred)

            independent_pred, independent_y, _ = run_an_eval_epoch(device, model, independent_loader, task_name,
                                                                   loss_criterion)
            np.savetxt(independent_path + 'fold_' + str(test_fold) + '_y_true.txt', independent_y)
            np.savetxt(independent_path + 'fold_' + str(test_fold) + '_pred.txt', independent_pred)

            if task_name == 'classification':
                auc, aupr, f1_score, acc = compute_cls_metrics(test_y, test_pred)
                test_mean += np.array([auc, aupr, f1_score, acc])
                row_test = ['test', auc, aupr, f1_score, acc]

                ind_auc, ind_aupr, ind_f1_score, ind_acc = compute_cls_metrics(independent_y, independent_pred)
                ind_mean += np.array([ind_auc, ind_aupr, ind_f1_score, ind_acc])
                row_ind = ['independent', ind_auc, ind_aupr, ind_f1_score, ind_acc]

            else:
                rmse, r2, r, mae = compute_reg_metrics(test_y, test_pred)
                test_mean += np.array([rmse, r2, r, mae])
                row_test = ['test', rmse, r2, r, mae]

                ind_rmse, ind_r2, ind_r, ind_mae = compute_reg_metrics(independent_y, independent_pred)
                ind_mean += np.array([ind_rmse, ind_r2, ind_r, ind_mae])
                row_ind = ['independent', ind_rmse, ind_r2, ind_r, ind_mae]

            t_tables.add_row(row_test)
            print(t_tables)

            ind_tables.add_row(row_ind)
            print(ind_tables)
            print(
                '---------------------------------------------------Test---------------------------------------------------')

        print('--------------------------------Final Results-----------------------------------')
        test_mean /= 5
        test_mean_row = ['mean', test_mean[0], test_mean[1], test_mean[2], test_mean[3]]
        t_tables.add_row(test_mean_row)
        print(t_tables)

        ind_mean /= 5
        ind_mean_row = ['mean', ind_mean[0], ind_mean[1], ind_mean[2], ind_mean[3]]
        ind_tables.add_row(ind_mean_row)
        print(ind_tables)

        val_filename = result_path + 'val.csv'
        test_filename = result_path + 'test.csv'
        independent_filename = independent_path + 'independent_metric' + '.csv'

        ptable_to_csv(val_tables, val_filename)
        ptable_to_csv(t_tables, test_filename)
        ptable_to_csv(ind_tables, independent_filename)
