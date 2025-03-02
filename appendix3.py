
import numpy as np
import matplotlib.pyplot as plt
import signal_handler
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sklearn.linear_model as lr

from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
import scipy.signal as sn
from fc_LFP import norm_cross_cor, visual_fc_comparison


remote_PC = False
if not(remote_PC): PATH = 'D:/Files/peirama_dipl/' # my PC path
if remote_PC: PATH = '/home/skoutinos/' # remote PC path

device = 'cpu'
move_to_gpu_list = ['None', 'all', 'batches'] # 'None'-> training is done in the cpu, 'all'-> all data are being moved in gpu at once, 'batches'-> data are move to gpu one batch at a time
move_to_gpu = move_to_gpu_list[0]
if move_to_gpu != 'None': device = 'cuda' if torch.cuda.is_available() else 'cpu'
if move_to_gpu != 'None': print(torch.cuda.get_device_name())

def main():
    pass
    tag= 'All_WT_0Mg'
    downsample_scale = 10
    sliding_window_step = 50
    input_size = 500
    output_size = 100

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]

    extract_data = 0
    train_model = False

    batch_size = 1024
    num_epochs = 4
    learning_rate = 0.001

    hidden_size = 512
    num_layers = 1
    bidirectional = True

    ar_model, train_loader, val_loader, _ =   prepare_residuals(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, extract_data, batch_size)

    model = AR_LSTM_fc(input_size, output_size, hidden_size, num_layers, bidirectional)
    criterion = nn.MSELoss()
    #criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    if train_model: trained_model, _ = train_hybrid_fc_nn(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
    if not(train_model): 
            checkpoint = torch.load(PATH + 'project_files/models/hybrid_ar_lstm_fc.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            trained_model = model

    num_gen_points = 234# output_size #3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
    if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό
    starting_point = 322
    # test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale)
    # np.save(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy', test_series)
    test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy')
    # fc_signal = create_forecasted_signal(trained_model, ar_model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)

    number_of_starting_points = 10
    index_range = np.arange(input_size, test_series.shape[1] - num_gen_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    starting_points_list = np.random.choice(index_range, size = number_of_starting_points, replace=False)
    #MAE_list = create_metrics_ar(trained_model, ar_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    #print(MAE_list)
    produce_metric_means(trained_model, ar_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method)

    plot_starting_point = np.random.randint(input_size, test_series.shape[1])
    actual_signal = test_series[0, plot_starting_point : plot_starting_point + num_gen_points]
    model_generated_signal = create_forecasted_signal(trained_model, ar_model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)
    fs = 1/(test_series[1,3] - test_series[1,2])
    visual_fc_comparison(actual_signal, fs, model_generated_signal, ml_method1='ar_lstm_hybrid', domain = 'both', save=False)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 

def prepare_residuals(tag, downsample_scale, input_size, output_size, window_step, scaling_method, extract_data, batch_size):
    save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
    if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path)
    if not(extract_data): lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    window_size = input_size + output_size
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    windowed_data = np.lib.stride_tricks.sliding_window_view(lfp_data, window_size, axis=1, writeable=True)[:,::window_step,:]
    print(f'windowed data have shape {windowed_data.shape}')

    windowed_data = np.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1],window_size)) # αυτή η εντολή κόβει τα παράθυρα και τα μπλέκει. Είναι καλύ για ml αλγόριθμους
    if scaling_method!= 'None': windowed_data = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά
    x_data = windowed_data[:,0:input_size]
    y_data = windowed_data[:,input_size:window_size]

    model = lr.LinearRegression() # instead of Linear regression, other ML regressor methods could be used
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    residuals = y_data - y_pred

    x_data = torch.from_numpy(x_data).float()
    residuals = torch.from_numpy(residuals).float()
    x_data =  torch.unsqueeze(x_data, 2)
    residuals =  torch.unsqueeze(residuals, 2)
        
    if move_to_gpu == 'all': x_data=x_data.to(device); residuals=residuals.to(device)
    dataset = torch.utils.data.TensorDataset(x_data, residuals)
    # το γεγονός ότι γίνεται random split στα παράθυρα προκαλεί temporal leakage, και καθιστά τσ validation scores μη αξιόπιστα. Θα έπρεπε είτε τα validation παράθυρα να είναι μετά από όλα τα trainin παράθυρα, είτε να προέρχονται από μια νέα χρονοσειρά όπως γίνται στο testing
    train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.8, 0.2]); del dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=train_data.__len__()
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=val_data.__len__()
    return model, train_loader, val_loader, scaler


### Architecture (class) of the LSTM-based-neural-network
class AR_LSTM_fc(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depends from the output size and the target data """
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional):
        super(AR_LSTM_fc, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        if bidirectional: self.bidirectional = 2
        if not(bidirectional): self.bidirectional = 1

        self.lstm=nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional) # nn.LSTM has dynamic layers throught the num_layer parameter which creates stacked LSTM
        self.linear1 = nn.Linear(self.bidirectional*self.hidden_size, 1) # combines straight and inverse h_t's
        self.linear2 = nn.Linear(self.input_size, self.output_size) # combines the points from time steps into final output

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = torch.squeeze(x)
        x = self.linear2(x)
        return x
    

def train_hybrid_fc_nn(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
        epoch_val_loss = 10^6 # initialization for the loop
        model = model.to(device)

        for epoch in range(num_epochs):
            train_time = []
            model.train()
            t1 = time.perf_counter()
            batch_train_losses = []
            for x_train, y_train in train_loader:
                if move_to_gpu == 'batches': x_train = x_train.to(device); y_train = y_train.to(device)
                train_pred = model(x_train)
                y_train = torch.squeeze(y_train); train_pred = torch.squeeze(train_pred)
                loss = criterion (y_train, train_pred)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_train_losses.append(loss.item()) # list of train_losses for every batch
            t2 = time.perf_counter()
            epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses) # mean of train_losses of all batches in every epoch
            train_time.append(t2-t1)

            model.eval()
            epoch_val_loss_old = epoch_val_loss
            batch_val_losses = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    if move_to_gpu == 'batches': x_val = x_val.to(device); y_val = y_val.to(device)
                    test_pred = model(x_val)
                    y_val = torch.squeeze(y_val); test_pred = torch.squeeze(test_pred)
                    val_loss = criterion (y_val, test_pred)
                    batch_val_losses.append(val_loss.item()) # list of val_losses for every batch
                epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses) # mean of train_losses of all batches in every epoch
            epoch_str = f'Epoch:{epoch+1}/{num_epochs} -> train (batch mean) loss = {epoch_train_loss} - val (batch mean) loss = {epoch_val_loss}'
            time_str = f'train_time: {train_time[0]}'
            print(epoch_str + ' - ' + time_str)

            if  epoch_val_loss < epoch_val_loss_old: 
                torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.input_size, 'output_size':model.output_size, 'hidden_size':model.hidden_size, 
                                                                                 'bidirectional':model.bidirectional, 'num_layers':model.num_layers, }}, 
                                                                                                                                  PATH + 'project_files/hybrid_ar_lstm_fc.pt')
                best_model = model
                best_val_score = epoch_val_loss
        
        return best_model, best_val_score


def create_forecasted_signal(res_model, linear_model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method):
    
    if starting_point<input_size:
        print ("starting_point can be smaller than input_size. There aren't enough points to be used as input. Starting point wiil change")
        starting_point = input_size
        print(f'New starting point is {starting_point} (equal to input size)')

    lfp_segment = test_series[0, starting_point-input_size : starting_point]
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    if scaling_method!= 'None': lfp_segment  = scaler.fit_transform1d(lfp_segment)

    res_model = res_model.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
    num_predictions =  int(num_gen_points/output_size) + 1
    input_signal = lfp_segment [(len(lfp_segment)-input_size) :] # λήψη των τελευταιων σημείων του σήματος για forecasting
    
    gen_signal = []
    res_model.eval()
    
    with torch.no_grad():
        for i in np.arange(num_predictions):
            linear_part = linear_model.predict(input_signal.reshape(1,-1))
            linear_part = np.squeeze(linear_part)

            input_signal = torch.from_numpy(input_signal).float()
            input_signal = torch.unsqueeze(input_signal, 0)
            input_signal = torch.unsqueeze(input_signal, 2)
            residual_part = res_model(input_signal)
            residual_part = torch.squeeze(residual_part)
            output = linear_part + residual_part.detach().numpy()

            input_signal = torch.squeeze(input_signal)
            input_signal = input_signal.numpy()
            input_signal = np.hstack((input_signal[output_size:], output))  # θα ενώσω τα δύο κομμάτια μετακινόντας το κάποιες θέσεις
    
            gen_signal = gen_signal + list(output) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης

    gen_signal = np.array(gen_signal)
    gen_signal =  gen_signal[:num_gen_points]
    if scaling_method!= 'None': gen_signal  = scaler.inverse1d(gen_signal)
    return gen_signal


def create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric):
    starting_points_list = np.array(starting_points_list)
    metric_array_list = np.zeros(len(starting_points_list))
    fs = 1/(test_series[1,3] - test_series[1,2])
    #test_signal = test_series[0,:]
    for idx, starting_point in enumerate(starting_points_list):
        actual_signal = test_series[0, starting_point : starting_point + num_gen_points]
        gen_signal = create_forecasted_signal(res_model, linear_model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)

        if metric == 'MAE':
                model_metric = mean_absolute_error(actual_signal, gen_signal)
        elif metric == 'RMSE':
            model_metric = np.sqrt(mean_squared_error(actual_signal, gen_signal))
        elif metric == 'norm-cross-corr': # Pearson r is equal to deiscrete normalized cross-corelation at zero time-lag,
            model_metric, _ = stats.pearsonr(actual_signal, gen_signal)
        elif metric == 'max Cross-cor': # this computes the maximum crsss-corelation between the 2 signals
            model_metric = np.max(norm_cross_cor(actual_signal, gen_signal)) # εδώ η cross-correlation κανονικοποιείται στο [-1,1]
        elif metric == 'RMS-PSD': # this is the root-mean-square-error of the PSD's. It is a metric fo how similar frequencies the two signals have
            _, Pxx_1 = sn.periodogram(actual_signal, fs=fs, return_onesided=True, scaling='density')
            _, Pxx_2 = sn.periodogram(gen_signal, fs=fs, return_onesided=True, scaling='density')
            model_metric = np.sqrt(mean_squared_error(Pxx_1, Pxx_2))
        metric_array_list[idx] = model_metric
    return metric_array_list


def produce_metric_means(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method):
    MAE_list = create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    RMSE_list = create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMSE')
    norm_cross_corr_list = create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='norm-cross-corr')
    max_cross_cor_list = create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='max Cross-cor')
    RMS_PSD_list = create_metrics_ar(res_model, linear_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMS-PSD')

    MAE_str = f'Absolute mean error {MAE_list.mean()}'
    RMSE_str = f'Root mean square error is {RMSE_list.mean()}'
    norm_cross_corr_str = f'Pearson r (normalized cross-correlation of zero phase) is {norm_cross_corr_list.mean()}'
    max_cross_cor_str = f'Maximum cross-correlation is {max_cross_cor_list.mean()}'
    RMS_PSD_str = f'Root mean square error of PSD is {RMS_PSD_list.mean()}'

    print(MAE_str +'\n'+ RMSE_str +'\n'+ norm_cross_corr_str +'\n'+ max_cross_cor_str +'\n'+ RMS_PSD_str)


main()