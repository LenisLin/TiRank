import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import optuna
from concurrent.futures import ThreadPoolExecutor

import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

from .Loss import *
from .Model import TiRankModel
from .Visualization import plot_loss
from .Dataloader import transform_test_exp

# Training


def Train_one_epoch(model, dataloader_A, dataloader_B, mode='Cox', infer_mode="SC", adj_A=None, adj_B=None, pre_patho_labels=None, optimizer=None, alphas=[1, 1, 1, 1], device="cpu"):

    model.train()

    # Initialize variables for loss components
    total_loss_all = 0.0
    regularization_loss_all = 0.0
    bulk_loss_all = 0.0
    cosine_loss_all = 0.0
    patho_loss_all = 0.0  # Added for patho loss
    loss_dict = {}

    ## DataLoader Bulk
    if mode == 'Cox':
        (X_a, t, e) = next(iter(dataloader_A))
        X_a = X_a.to(device)
        t = t.to(device)
        e = e.to(device)

    if mode in ['Classification', 'Regression']:
        (X_a, label) = next(iter(dataloader_A))
        X_a = X_a.to(device)
        label = label.to(device)

    for batch_B in dataloader_B:
        ## DataLoader SC or ST
        (X_b, idx) = batch_B
        X_b = X_b.to(device)

        if adj_A is not None:
            A = adj_A[idx, :][:, idx]
            A = A.to(device)

        if adj_B is not None:
            B = adj_B[idx, :][:, idx]
            B = B.to(device)

        if pre_patho_labels is not None:
            # Convert the tensor idx to numpy for indexing pandas series
            idx_np = idx.cpu().numpy()
            pre_patho = pre_patho_labels.iloc[idx_np].values
            # Specify dtype if necessary
            pre_patho = torch.tensor(pre_patho, dtype=torch.uint8)
            pre_patho = pre_patho.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        _, risk_scores_a, _ = model(X_a)
        embeddings_b, _, pred_patho = model(X_b)

        regularization_loss_ = regularization_loss(model.feature_weights)

        # Calculate loss
        if mode == 'Cox':
            bulk_loss_ = cox_loss(risk_scores_a, t, e)

        elif mode == 'Classification':
            bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

        elif mode == 'Regression':
            bulk_loss_ = MSE_loss(risk_scores_a, label)

        if infer_mode == 'SC':
            cosine_loss_exp_ = cosine_loss(embeddings_b, A)

            # total loss
            total_loss = regularization_loss_ * alphas[0] + \
                bulk_loss_ * alphas[1] + \
                cosine_loss_exp_ * alphas[2]
            
            pathoLloss = torch.tensor(0)

        elif infer_mode == 'ST' and adj_B is not None:
            cosine_loss_exp_ = cosine_loss(embeddings_b, A)
            cosine_loss_spatial_ = cosine_loss(embeddings_b, B)


            # total loss
            total_loss = regularization_loss_ * alphas[0] + \
                bulk_loss_ * alphas[1] + \
                cosine_loss_exp_ * alphas[2] + \
                cosine_loss_spatial_ * alphas[3]

        elif infer_mode == 'ST' and pre_patho is not None:
            cosine_loss_exp_ = cosine_loss(embeddings_b, A)
            pathoLloss = CrossEntropy_loss(pred_patho, pre_patho)

            # total loss

            total_loss = regularization_loss_ * alphas[0] + \
                bulk_loss_ * alphas[1] + \
                cosine_loss_exp_ * alphas[2] + \
                pathoLloss * alphas[3]
        else:
            raise ValueError(
                f"Unsupported mode: {infer_mode}. There are two mode: Cell and Spot.")

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        total_loss_all += total_loss.item()
        regularization_loss_all += regularization_loss_.item()
        bulk_loss_all += bulk_loss_.item()
        cosine_loss_all += cosine_loss_exp_.item()
        patho_loss_all += pathoLloss.item()

    loss_dict["all_loss_"] = total_loss_all / len(dataloader_B)
    loss_dict["regularization_loss_"] = regularization_loss_all / len(dataloader_B)
    loss_dict["bulk_loss_"] = bulk_loss_all / len(dataloader_B)
    loss_dict["cosine_loss_"] = cosine_loss_all / len(dataloader_B)
    loss_dict["patho_loss_all"] = patho_loss_all / len(dataloader_B)

    return loss_dict

# Validate


def Validate_model(model, dataloader_A, dataloader_B, mode='Cox', infer_mode="SC", adj_A=None, adj_B=None, pre_patho_labels=None, alphas=[1, 1, 1, 1], device="cpu"):

    model.eval()  # Set the model to evaluation mode

    validation_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        # RNA-seq data whole batch validation
        iter_A = iter(dataloader_A)

        if mode == 'Cox':
            (X_a, t, e) = next(iter_A)
            X_a = X_a.to(device)
            t = t.to(device)
            e = e.to(device)

        if mode in ['Classification', 'Regression']:
            (X_a, label) = next(iter_A)
            X_a = X_a.to(device)
            label = label.to(device)

        for batch_B in dataloader_B:
            # Similar setup as in the training function
            (X_b, idx) = batch_B
            X_b = X_b.to(device)

            if adj_A is not None:
                A = adj_A[idx, :][:, idx]
                A = A.to(device)

            if adj_B is not None:
                B = adj_B[idx, :][:, idx]
                B = B.to(device)

            if pre_patho_labels is not None:
                idx_np = idx.cpu().numpy()
                pre_patho = pre_patho_labels.iloc[idx_np].values
                pre_patho = torch.tensor(pre_patho, dtype=torch.uint8)
                pre_patho = pre_patho.to(device)

            # Forward pass only
            _, risk_scores_a, _ = model(X_a)
            embeddings_b, _, pred_patho = model(X_b)

            # Compute loss as in training function but without backward and optimizer steps
            # The computation here assumes that the loss functions and infer_mode conditions are the same as in training
            # Please adapt if your validation conditions differ from the training

            # Calculate loss
            if mode == 'Cox':
                bulk_loss_ = cox_loss(risk_scores_a, t, e)

            elif mode == 'Classification':
                bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

            elif mode == 'Regression':
                bulk_loss_ = MSE_loss(risk_scores_a, label)

            if infer_mode == 'SC':
                cosine_loss_exp_ = cosine_loss(embeddings_b, A)

                # total loss
                total_loss = bulk_loss_ * alphas[0] + cosine_loss_exp_ * alphas[1]

            elif infer_mode == 'ST' and adj_B is not None:
                cosine_loss_exp_ = cosine_loss(embeddings_b, A)
                cosine_loss_spatial_ = cosine_loss(embeddings_b, B)

                # total loss

                total_loss = bulk_loss_ * alphas[0] + \
                    cosine_loss_exp_ * alphas[1] + \
                    cosine_loss_spatial_ * alphas[2]

            elif infer_mode == 'ST' and pre_patho is not None:
                cosine_loss_exp_ = cosine_loss(embeddings_b, A)
                pathoLloss = CrossEntropy_loss(pred_patho, pre_patho)

                # total loss

                total_loss = bulk_loss_ * alphas[0] + \
                    cosine_loss_exp_ * alphas[1] + \
                    pathoLloss * alphas[2]
            else:
                raise ValueError(
                    f"Unsupported mode: {infer_mode}. There are two mode: Cell and Spot.")

            # Add up the total loss for the validation set
            validation_loss += total_loss.item()

    # Calculate the average loss over all batches in the dataloader
    return validation_loss / len(dataloader_B)

# Reject


def Reject_With_GMM_Bio(pred_bulk, pred_sc, tolerance, min_components, max_components):
    print(
        f"Perform Rejection with GMM mode with tolerance={tolerance}, components=[{min_components},{max_components}]!")

    gmm_bulk = GaussianMixture(n_components=2, random_state=619).fit(pred_bulk)

    gmm_bulk_mean_1 = np.max(gmm_bulk.means_)
    gmm_bulk_mean_0 = np.min(gmm_bulk.means_)

    if (gmm_bulk_mean_1 - gmm_bulk_mean_0) <= 0.5:
        print("Underfitting!")

    # Iterate over the number of components
    for n_components in range(min_components, max_components + 1):
        gmm_sc = GaussianMixture(
            n_components=n_components, random_state=619).fit(pred_sc)

        means = gmm_sc.means_

        # Check if any of the means are close to 0 or 1
        zero_close = any(abs(mean - gmm_bulk_mean_0) <=
                         tolerance for mean in means)
        one_close = any(abs(gmm_bulk_mean_1 - mean) <=
                        tolerance for mean in means)

        if zero_close and one_close:
            print(
                f"Found distributions with means close to 0 and 1 with {n_components} components.")

            # # Print the means and covariances
            # print("Means of the gaussians in gmm_sc: ", gmm_sc.means_)
            # print("Covariances of the gaussians in gmm_sc: ", gmm_sc.covariances_)

            # Find the component whose mean is nearest to 0 and 1
            # 1
            diffs_1 = abs(gmm_bulk_mean_1 - gmm_sc.means_)
            nearest_component_1 = np.where(diffs_1 <= tolerance)[0]

            # 0
            diffs_0 = abs(gmm_sc.means_ - gmm_bulk_mean_0)
            nearest_component_0 = np.where(diffs_0 <= tolerance)[0]

            # concat
            remain_component = np.concatenate(
                (nearest_component_1, nearest_component_0))

            # The mask of those rejection cell
            labels_sc = gmm_sc.predict(pred_sc)

            mask = np.ones(shape=(len(labels_sc), 1))

            mask[np.isin(labels_sc, remain_component)] = 0

            print(
                f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

            return mask

    print(f"Two distribution rejection faild.")

    print(f"Perform single distribution rejection.")
    for n_components in range(2, max_components + 1):
        gmm_sc = GaussianMixture(
            n_components=n_components, random_state=619).fit(pred_sc)

        means = gmm_sc.means_

        # Check if any of the means are close to 0 or 1
        zero_close = any(abs(mean - gmm_bulk_mean_0) <=
                         tolerance for mean in means)
        one_close = any(abs(gmm_bulk_mean_1 - mean) <=
                        tolerance for mean in means)

        if zero_close or one_close:
            if zero_close:
                print(
                    f"Found distributions with means close to 0 with {n_components} components.")
            if one_close:
                print(
                    f"Found distributions with means close to 1 with {n_components} components.")

            # # Print the means and covariances
            # print("Means of the gaussians in gmm_sc: ", gmm_sc.means_)
            # print("Covariances of the gaussians in gmm_sc: ", gmm_sc.covariances_)

            # Find the component whose mean is nearest to 0 and 1
            # 1
            diffs_1 = gmm_bulk_mean_1 - gmm_sc.means_
            nearest_component_1 = np.where(diffs_1 <= tolerance)[0]

            # 0
            diffs_0 = gmm_sc.means_ - gmm_bulk_mean_0
            nearest_component_0 = np.where(diffs_0 <= tolerance)[0]

            # concat
            remain_component = np.concatenate(
                (nearest_component_1, nearest_component_0))

            # The mask of those rejection cell
            labels_sc = gmm_sc.predict(pred_sc)

            mask = np.ones(shape=(len(labels_sc), 1))

            mask[np.isin(labels_sc, remain_component)] = 0

            print(
                f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

            return mask

    print(f"Single distribution rejection faild.")
    mask = np.zeros(shape=(len(pred_sc), 1))

    return mask


def Reject_With_GMM_Reg(pred_bulk, pred_sc, tolerance):
    print(f"Perform Rejection with GMM mode with tolerance={tolerance}!")

    gmm_bulk = GaussianMixture(n_components=1, random_state=619).fit(pred_bulk)
    gmm_sc = GaussianMixture(n_components=1, random_state=619).fit(pred_sc)

    # Mean
    gmm_bulk_mean = gmm_bulk.means_[0][0]
    gmm_sc_mean = gmm_sc.means_[0][0]

    # Std
    gmm_bulk_variance = gmm_bulk.covariances_[0][0]
    gmm_bulk_std = np.sqrt(gmm_bulk_variance)

    if (abs(gmm_sc_mean - gmm_bulk_mean)) >= 0.5:
        print("Integration was failed !")
        mask = np.ones(shape=(len(pred_sc), 1))

    else:
        if tolerance > gmm_bulk_std:
            tolerance = gmm_bulk_std
        lower_bound = gmm_bulk_mean - tolerance
        upper_bound = gmm_bulk_mean + tolerance

        mask = np.ones(shape=(len(pred_sc), 1))
        mask[(pred_sc >= lower_bound) & (pred_sc <= upper_bound)] = 0

    print(
        f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

    return mask


def Reject_With_StrictNumber(pred_bulk, pred_sc, tolerance):
    print(
        f"Perform Rejection with Strict Number mode with tolerance={tolerance}!")

    gmm_bulk = GaussianMixture(n_components=2, random_state=619).fit(pred_bulk)

    # Get mean
    gmm_bulk_means = gmm_bulk.means_.flatten()
    gmm_bulk_mean_1 = np.max(gmm_bulk_means)
    gmm_bulk_mean_0 = np.min(gmm_bulk_means)

    # Get std
    gmm_bulk_stds = np.sqrt(gmm_bulk.covariances_)
    gmm_bulk_std_1 = gmm_bulk_stds[0][0]
    gmm_bulk_std_0 = gmm_bulk_stds[1][0]

    if (gmm_bulk_mean_1 - gmm_bulk_mean_0) <= 0.8:
        print("Underfitting!")

    # Calculate the percentile range for the tolerance
    lower_percentile = 0.5 - tolerance / 2
    upper_percentile = 0.5 + tolerance / 2

    # Calculate the range for the first Gaussian distribution
    range_low_1 = stats.norm.ppf(
        lower_percentile, gmm_bulk_mean_1, gmm_bulk_std_1)
    range_high_1 = stats.norm.ppf(
        upper_percentile, gmm_bulk_mean_1, gmm_bulk_std_1)

    # Calculate the range for the second Gaussian distribution
    range_low_0 = stats.norm.ppf(
        lower_percentile, gmm_bulk_mean_0, gmm_bulk_std_0)
    range_high_0 = stats.norm.ppf(
        upper_percentile, gmm_bulk_mean_0, gmm_bulk_std_0)

    print(
        f"For the first Gaussian distribution with mean {gmm_bulk_mean_1} and std {gmm_bulk_std_1}:")
    print(
        f"The range around the mean that contains {tolerance*100}% of the samples is approximately from {range_low_1} to {range_high_1}")

    print(
        f"For the second Gaussian distribution with mean {gmm_bulk_mean_0} and std {gmm_bulk_std_0}:")
    print(
        f"The range around the mean that contains {tolerance*100}% of the samples is approximately from {range_low_0} to {range_high_0}")

    mask = np.ones(shape=(len(pred_sc), 1))

    # Set mask to zero where the condition for the first Gaussian distribution is met
    mask[(pred_sc >= range_low_1) & (pred_sc <= range_high_1)] = 0

    # Set mask to zero where the condition for the second Gaussian distribution is met
    mask[(pred_sc >= range_low_0) & (pred_sc <= range_high_0)] = 0

    print(
        f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

    return mask

# Hyper-paremeters tuning


def objective(trial,
              n_features, nhead, nhid1,
              nhid2, n_output, nlayers, n_pred, dropout, n_patho, mode, encoder_type,
              train_loader_Bulk, val_loader_Bulk, train_loader_SC, adj_A, adj_B, pre_patho_labels, device, infer_mode, model_save_path):

    print(f"Initiate model in trail {trial.number} with mode: {mode}, infer mode: {infer_mode} encoder type: {encoder_type} on device: {device}.")
    model = TiRankModel(n_features=n_features, nhead=nhead, nhid1=nhid1, nhid2=nhid2, n_output=n_output,
                   nlayers=nlayers, n_pred=n_pred, dropout=dropout, n_patho = n_patho, mode=mode, encoder_type=encoder_type)
    model = model.to(device)

    # Define hyperparameters with trial object
    lr_choices = [1e-2, 6e-3, 3e-3, 1e-3, 6e-4, 3e-4, 1e-4,1e-5,1e-6]
    lr = trial.suggest_categorical("lr", lr_choices)

    n_epochs_choices = [300, 350, 400, 450, 500]
    n_epochs = trial.suggest_categorical("n_epochs", n_epochs_choices)

    # Define alpha values as specific choices
    alpha_0_choices = [0.1, 0.06, 0.03, 0.01, 0.005]
    alpha_1_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alpha_2_choices = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    alpha_3_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Suggest categorical choices for each alpha
    alphas = [
        trial.suggest_categorical("alpha_0", alpha_0_choices),
        trial.suggest_categorical("alpha_1", alpha_1_choices),
        trial.suggest_categorical("alpha_2", alpha_2_choices),
        trial.suggest_categorical("alpha_3", alpha_3_choices),
    ]

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    train_loss_dcit = dict()

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss_dcit_epoch = Train_one_epoch(
            model=model,
            dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
            mode=mode, infer_mode=infer_mode,
            adj_A=adj_A,adj_B = adj_B, pre_patho_labels = pre_patho_labels,
            optimizer=optimizer, alphas=alphas, device=device)

        train_loss_dcit["Epoch_"+str(epoch+1)] = train_loss_dcit_epoch

        scheduler.step()

        # Val
        model.eval()
        val_loss = Validate_model(
            model=model,
            dataloader_A=val_loader_Bulk, dataloader_B=train_loader_SC,
            mode=mode, infer_mode=infer_mode,
            adj_A=adj_A,adj_B = adj_B, pre_patho_labels = pre_patho_labels,
            alphas=[1, 0.1, 1, 0], device=device)


        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Generate the filename including hyperparameters
    hyperparams_str = f"lr_{lr}_epochs_{n_epochs}_alpha0_{alphas[0]}_alpha1_{alphas[1]}_alpha2_{alphas[2]}_alpha3_{alphas[3]}"
    model_filename = f"model_{hyperparams_str}.pt"
    plot_filename = f"loss_curve_trial_{trial.number}_{hyperparams_str}_val_loss_{val_loss:.4f}.png"

    # Saving the model and plot with new filenames
    if trial.number == 0 or (trial.study.best_trials and val_loss < trial.study.best_value):
        print(f"Saving model and plot for Trial {trial.number} with Validation Loss = {val_loss:.4f}")
        plot_loss(train_loss_dcit, alphas, savePath=os.path.join(model_save_path, plot_filename))
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))

    return val_loss


def tune_hyperparameters(
    savePath,  # Dictionary containing all model parameters
    device='cpu',  # Computing device ('cpu' or 'cuda')
    n_trials=50,  # Number of optimization trials
):
    

    savePath_3 = os.path.join(savePath,"3_Analysis")
    savePath_data2train = os.path.join(savePath_3,"data2train")

    # Load dataloaders
    f = open(os.path.join(savePath_data2train, 'train_loader_Bulk.pkl'), 'rb')
    train_loader_Bulk = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, 'val_loader_Bulk.pkl'), 'rb')
    val_loader_Bulk = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, 'train_loader_SC.pkl'), 'rb')
    train_loader_SC = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_data2train, 'adj_A.pkl'), 'rb')
    adj_A = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, 'adj_B.pkl'), 'rb')
    adj_B = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, 'patholabels.pkl'), 'rb')
    pre_patho_labels = pickle.load(f)
    f.close()

    # Initial model parameters
    f = open(os.path.join(savePath_3, 'model_para.pkl'), 'rb')
    model_para = pickle.load(f)
    f.close()

    # Extract parameters from the model_para dictionary
    n_features = model_para.get('n_features', None)
    nhead = model_para.get('nhead', 8)
    nhid1 = model_para.get('nhid1', 256)
    nhid2 = model_para.get('nhid2', 128)
    n_output = model_para.get('n_output', 10)
    nlayers = model_para.get('nlayers', 2)
    n_pred = model_para.get('n_pred', 1)
    dropout = model_para.get('dropout', 0.5)
    n_patho = model_para.get('n_patho', 0)
    mode = model_para.get('mode', 'Cox')
    infer_mode = model_para.get('infer_mode', 'Cell')
    encoder_type = model_para.get('encoder_type', 'MLP')
    model_save_path = model_para.get('model_save_path', './checkpoints/')

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial,
                                           n_features, nhead, nhid1,
                                           nhid2, n_output, nlayers, n_pred, dropout, n_patho, mode, encoder_type,
                                           train_loader_Bulk, val_loader_Bulk, train_loader_SC, 
                                           adj_A ,adj_B, pre_patho_labels, device, infer_mode, model_save_path), n_trials=n_trials)
    
    # save the best hyperparameters
    best_params = study.best_trial.params
    with open(os.path.join(savePath_3, 'best_params.pkl'), 'wb') as f:
        print("Best hyperparameters:", best_params)
        pickle.dump(best_params, f) ## bet parameters set
    f.close()

    return None

# Get Best performance model
def get_best_model(savePath):
    print("Loading the Best Model.")
    savePath_3 = os.path.join(savePath,"3_Analysis") 
    model_save_path = os.path.join(savePath_3,"checkpoints")

    f = open(os.path.join(savePath_3, 'best_params.pkl'), 'rb')
    best_params = pickle.load(f)
    f.close()

    lr = best_params['lr']
    n_epochs = best_params['n_epochs']
    alpha_0 = best_params['alpha_0']
    alpha_1 = best_params['alpha_1']
    alpha_2 = best_params['alpha_2']
    alpha_3 = best_params['alpha_3']

    f = open(os.path.join(savePath_3, 'model_para.pkl'), 'rb')
    model_para = pickle.load(f)
    f.close()

    ## Load model
    filename = os.path.join(model_save_path,f"model_lr_{lr}_epochs_{n_epochs}_alpha0_{alpha_0}_alpha1_{alpha_1}_alpha2_{alpha_2}_alpha3_{alpha_3}.pt")

    n_features = model_para.get('n_features', None)
    nhead = model_para.get('nhead', 8)
    nhid1 = model_para.get('nhid1', 256)
    nhid2 = model_para.get('nhid2', 128)
    n_output = model_para.get('n_output', 10)
    nlayers = model_para.get('nlayers', 2)
    n_pred = model_para.get('n_pred', 1)
    dropout = model_para.get('dropout', 0.5)
    n_patho = model_para.get('n_patho', 0)
    mode = model_para.get('mode', 'Cox')
    encoder_type = model_para.get('encoder_type', 'MLP')
    model_save_path = model_para.get('model_save_path', './checkpoints/')


    model = TiRankModel(n_features=n_features, nhead=nhead, nhid1=nhid1,
                nhid2=nhid2, n_output=n_output, nlayers=nlayers, n_pred=n_pred, dropout=dropout, n_patho=n_patho, mode=mode, encoder_type=encoder_type)
    model.load_state_dict(torch.load(filename))
    model = model.to("cpu")

    return model

# Predict


def Predict(savePath, mode, do_reject=True, tolerance=0.05, reject_mode="GMM"):
    savePath_1 = os.path.join(savePath,"1_loaddata")
    savePath_2 = os.path.join(savePath,"2_preprocessing")
    savePath_3 = os.path.join(savePath,"3_Analysis")

    model = get_best_model(savePath)

    print("Starting Inference.")

    # Load data
    ### Training bulk set
    f = open(os.path.join(savePath_2, 'train_bulk_gene_pairs_mat.pkl'), 'rb')
    train_bulk_gene_pairs_mat = pickle.load(f)
    f.close()

    ### All bulk
    f = open(os.path.join(savePath_1, 'bulk_exp.pkl'), 'rb')
    bulkExp = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_1, 'bulk_clinical.pkl'), 'rb')
    bulkClinical = pickle.load(f)
    f.close()

    bulk_rownames=bulkClinical.index.tolist()

    ### Transfer all bulk into gene pairs
    bulk_GPmat = transform_test_exp(train_exp = train_bulk_gene_pairs_mat,test_exp = bulkExp)

    ### SC gene pair matrix
    f = open(os.path.join(savePath_2, 'sc_gene_pairs_mat.pkl'), 'rb')
    sc_GPmat = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_2, 'scAnndata.pkl'), 'rb')
    scAnndata = pickle.load(f)
    f.close()

    sc_rownames = scAnndata.obs.index.tolist()

    model.eval()
    
    # Predict on cell
    Exp_Tensor_sc = torch.from_numpy(np.array(sc_GPmat))
    Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)
    embeddings_sc, pred_sc, _ = model(Exp_Tensor_sc)

    # Predict on bulk
    Exp_Tensor_bulk = torch.from_numpy(np.array(bulk_GPmat))
    Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)
    embeddings_bulk, pred_bulk, _ = model(Exp_Tensor_bulk)

    if mode == "Cox":
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)

    elif mode == "Classification":
        pred_sc = pred_sc[:, 1].detach().numpy().reshape(-1, 1)
        pred_bulk = pred_bulk[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)

    embeddings_sc = embeddings_sc.detach().numpy()
    embeddings_bulk = embeddings_bulk.detach().numpy()

    if do_reject:
        if reject_mode == "GMM":
            if mode in ["Cox", "Classification"]:
                reject_mask = Reject_With_GMM_Bio(pred_bulk, pred_sc,
                                                  tolerance=tolerance, min_components=3, max_components=15)
            if mode == "Regression":
                reject_mask = Reject_With_GMM_Reg(
                    pred_bulk, pred_sc, tolerance=tolerance)

        elif reject_mode == "Strict":
            if mode in ["Cox", "Classification"]:
                reject_mask = Reject_With_StrictNumber(
                    pred_bulk, pred_sc, tolerance=tolerance)

            if mode == "Regression":
                print("Test")
        else:
            raise ValueError(f"Unsupported Rejcetion Mode: {reject_mode}")

    else:
        reject_mask = np.zeros_like()

    saveDF_sc = pd.DataFrame(data=np.concatenate(
        (reject_mask, pred_sc, embeddings_sc), axis=1), index=sc_GPmat.index)

    colnames = ["Reject", "Pred_score"]
    colnames.extend(["embedding_" + str(i + 1)
                    for i in range(embeddings_sc.shape[1])])

    saveDF_sc.columns = colnames
    saveDF_sc.index = sc_rownames

    reject_mask_ = np.zeros_like(pred_bulk)
    saveDF_bulk = pd.DataFrame(data=np.concatenate(
        (reject_mask_, pred_bulk, embeddings_bulk), axis=1), index=bulk_GPmat.index)

    saveDF_bulk.columns = colnames
    saveDF_bulk.index = bulk_rownames

    print("Inference Done.")

    with open(os.path.join(savePath_3, 'saveDF_bulk.pkl'), 'wb') as f:
        pickle.dump(saveDF_bulk, f)
    f.close()
    with open(os.path.join(savePath_3, 'saveDF_sc.pkl'), 'wb') as f:
        pickle.dump(saveDF_sc, f)
    f.close()

    ## Original categorize function
    if saveDF_sc.shape[0] != scAnndata.obs.shape[0]:
        raise ValueError(
            "The prediction matrix was not match with original scAnndata.")

    scAnndata.obsm["Rank_Embedding"] = saveDF_sc.iloc[:, 2:]
    scAnndata.obs["Reject"] = saveDF_sc.iloc[:, 0]
    scAnndata.obs["Rank_Score"] = saveDF_sc.iloc[:, 1]

    if mode in ["Cox", "Classification"]:
        temp = scAnndata.obs["Rank_Score"] * (1 - scAnndata.obs["Reject"])
        scAnndata.obs["Rank_Label"] = [
            "Background" if i == 0 else
            "Rank-" if 0 <= i < 0.5 else
            "Rank+"
            for i in temp
        ]

        print(f"We set Rank score < 0.5 as Rank- () while > 0.5 as Rank+ ")

    if mode == "Regression":
        scAnndata.obs["Rank_Label"] = scAnndata.obs["Rank_Score"] * \
            (1 - scAnndata.obs["Reject"])
            
    ## Save
    sc_pred_df = scAnndata.obs
    sc_pred_df.to_csv(os.path.join(savePath_3, "spot_predict_score.csv"))
    scAnndata.write_h5ad(filename=os.path.join(savePath_3, "final_anndata.h5ad"))

    return None

# permutation test to determine the phenotype associated cluster


def permute_once(Rank_Labels, Labels, unique_labels):
    shuffled_rank_labels = np.random.permutation(Rank_Labels)
    local_counts = {label: {"Background": 0, "Rank+": 0, "Rank-": 0}
                    for label in unique_labels}
    for label in unique_labels:
        indices = [i for i, x in enumerate(Labels) if x == label]
        subset = [shuffled_rank_labels[i] for i in indices]
        local_counts[label] = dict((x, subset.count(x))
                                   for x in {"Background", "Rank+", "Rank-"})
    return local_counts


def Pcluster(scAnndata, clusterColName, perm_n=1001):
    # Check if the clusterColName is in the observation matrix
    if clusterColName not in scAnndata.obs.keys():
        raise ValueError(
            f"{clusterColName} was not in anndata observation matrix.")

    # Extract data from the Anndata object
    Labels = scAnndata.obs[clusterColName].tolist()
    Rank_Labels = scAnndata.obs["Rank_Label"].tolist()

   # Count the actual occurrences
    unique_labels = set(Labels)
    actual_counts = {}
    for label in unique_labels:
        indices = [i for i, x in enumerate(Labels) if x == label]
        subset = [Rank_Labels[i] for i in indices]
        actual_counts[label] = dict((x, subset.count(x))
                                    for x in {"Background", "Rank+", "Rank-"})

    # Permutation procedure using multi-threading
    permuted_counts = {label: {"Background": [],
                               "Rank+": [], "Rank-": []} for label in unique_labels}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            permute_once, Rank_Labels, Labels, unique_labels) for _ in range(perm_n)]
        for future in futures:
            result = future.result()
            for label in unique_labels:
                for key in {"Background", "Rank+", "Rank-"}:
                    permuted_counts[label][key].append(result[label][key])

    # Calculate p-values based on the permuted distribution
    p_values = {label: {} for label in unique_labels}
    for label in unique_labels:
        for key in {"Rank+", "Rank-", "Background"}:
            observed = actual_counts[label][key]
            if sum(permuted_counts[label][key]) == 0:
                p_values[label][key] = np.nan

            else:
                extreme_count = sum(
                    1 for x in permuted_counts[label][key] if observed > x)
                p_values[label][key] = extreme_count / perm_n

    # transpose the DataFrame to get labels as rows
    df_p_values = pd.DataFrame(p_values).T

    return df_p_values