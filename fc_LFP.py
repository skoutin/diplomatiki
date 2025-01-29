"""This file uses an LSTM-RNN for the forecasting of the LFP signal"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import signal_handler

import sklearn.linear_model as lrm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
import statsmodels.stats.descriptivestats as stats_ds
import scipy.signal as sn 
import colorednoise as cn

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES

remote_PC = False
if not(remote_PC): PATH = 'D:/Files/peirama_dipl/' # my PC path
if remote_PC: PATH = '/home/skoutinos/' # remote PC path

execute = 'main()' # options: main() , 'multiple_trainings'
save_terminal_output_to_file = False
save_plots = False

move_to_gpu_list = ['None', 'all', 'batches'] # 'None'-> training is done in the cpu, 'all'-> all data are being moved in gpu at once, 'batches'-> data are move to gpu one batch at a time
move_to_gpu = move_to_gpu_list[1]
if move_to_gpu != 'None': device = 'cuda' if torch.cuda.is_available() else 'cpu'
if move_to_gpu != 'None': print(torch.cuda.get_device_name())

optimizer_used = 'adam' # cetermines the optimizer used for lstm training. Two options are 'adam' & 'sgd'
loss_function_used_list = ['mse', 'mae', 'huber']
loss_function_used = loss_function_used_list[0]

tf_like_output = False # If true the tearget data will be same as the input data, bat one time step ahead e.g. input = x1,x2,...x10, ouput = x2,x3,...x11. This is used in teacher forcing taining
lstm_seq_type = 'seq2one' # Determines if LSTM will use only the last ouput, or all of the outputs as it unrolls. Choises are 'seq2seq' or 'seq2one'
bidirectional = True # determines if the LSTM will be bidirectional or not

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():

    ###### DATA PARAMETERS
    tag= 'All_WT_0Mg' # All_EA_WT_0Mg' #'All_WT_0Mg'   # determines which groups of files will be loaded, and used for training
    downsample_scale = 1000 # determines how many time will the signal be downsampled
    sliding_window_step = 100 # this is the number of the window sliding for the creation of the windows that will be used for training
    
    sliding_window_step_ml_methods = 50 # for the training of older methods that use a subset of whole training data by taking a bigger sliding window
    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]

    extract_data = 0 # if it is True the data are being extracted by .mat files and are being saved in a .npy file, if it is False data are being loaded from the .npy file
    if remote_PC: extract_data = False

    ##### TRAINING PARAMETERS
    input_size = 100 # this is the number of the input_data of the LSTM, i.e. the number of points used for forecasting
    output_size = 30 # this is the number of output_data of the LSTM, i.e. the future points forecasted by the LSTM
    if tf_like_output: output_size = input_size # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση
    # sliding_window_step = input_size # for not overlaping windows
    hidden_state_dim = 7 # the size of the hidden/cell state of LSTM
    num_layers = 1 # the number of consecutive LSTM cells the nn.LSTM will have (i.e. number of stacked LSTM's)
    lr = 0.1 # optimizer's learning rate
    momentum = 0.9 # optimizer's momentum -> for SGD, not for Adam (Adam has inherent momentum)
    epochs = 2
    batch_size = 1024 # how many rows each batch will have. 1 is the minimum and creates the max number of batches but they are the smallest in terms of size
    
    ##### TRAIN_SWITCH PARAMETERS
    train_LSTM = 1 # for True it trains the model, for False it loads a saved model # ΠΡΟΣΟΧΗ αν κάνεις load μοντέλο που το έχεις εκπαιδεύσει με άλλο output_type προφανώς θα προκύψει σφάλμα -> επιλύθηκε με την αποθήκευση και τη φόρτωση των παραμέτρων μαζί με το LSTM
    load_lstm = 0
    train_older = 0 # trains linear (autoregresson) and dummy regresson
    load_older = 0
    save_load_model_number = 0 # καθορίζει ποιο LSTM μοντέλο θα φορτωθεί (η αποθήκευση γίνεται στο φάκελο και τα μεταφέρεις manually στους φακέλους model)
    evaluate_models = 0


    # Extract and save data for training and validation
    if extract_data:
        save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
        lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path) 
        print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

   # train or load LSTM 
    if train_LSTM: lstm_model, _ = LSTM_train(tag, downsample_scale, sliding_window_step, hidden_state_dim, input_size, output_size, num_layers, batch_size, lr, momentum, epochs, scaling_method, save_load_model_number)
    if load_lstm:
        dict_train = load_params(save_load_model_number)
        input_size, hidden_state_dim, num_layers, output_size = dict_train['input_size'], dict_train['hidden_state_dim'], dict_train['num_layers'], dict_train['output_size'] # # φορτώνει τις παραμέτρους του loaded LSTM. Αυτές οι παράμετροι χρειάζονται για το loading του LSTM
        downsample_scale, scaling_method = dict_train['downsample_scale'], dict_train['scaling_method'] # φορτώνει τις παραμέτρους του loaded LSTM. Αυτές οι παράμετροι χρειάζονται για το generate/compare
        tag = dict_train['tag'] # αυτή η παράμετρος χρειάζεται για την εκπαίδευση των άλλων μεθόδων στα ίδια δεδομένα με αυτά της εκπαίδευσης του LSTM
        lstm_model = LSTM_load(save_load_model_number, input_size, hidden_state_dim, num_layers, output_size)

    visualize = False
    if visualize:
        input_tensor = np.linspace(0,1,input_size); input_tensor=torch.tensor(input_tensor, dtype=torch.float32)
        input_tensor=torch.unsqueeze(input_tensor,1)
        input_tensor=torch.unsqueeze(input_tensor,0); print(input_tensor.shape)
        forecasted = lstm_model(input_tensor) # input must be dims (batch_size, sequence_length, input_size=1)
        torch.onnx.export(lstm_model, (input_tensor,), PATH + 'project_files/' + 'lstm_model.onnx', input_names=["input"])

    # asseses underfiiting/overfitting by inspecting how LSTM performs on data, previously seen during training
    if train_LSTM or load_lstm:
        print('\nLoad previously seen data in order to check underfiiting/overfitting')
        if not(remote_PC): check_series = signal_handler.combine (signal_handler.lists_of_names('WT1'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
        if remote_PC: check_series = np.load(PATH + 'project_files/WT1_ds'+ str(downsample_scale)  + '.npy') # for remote pc
        print('length of check series is ', check_series.shape)
        num_gen_points = output_size #3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
        if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό
        number_of_starting_points = 40 # καθορίζει πόσα τυχαία σημεία έναρξης της πρόβλεψης θα παρθούν για την παραγωγή των λιστών της κάθε μετρικής
        evaluate_lstm(lstm_model, check_series, num_gen_points, number_of_starting_points, scaling_method, make_barplots=True)




    # train linear and dummy regressors with the same data
    dummy_save_name = 'dummy_fc_model' #'None'
    linear_save_name = 'linear_fc_model'
    if train_older:
        dummy = train_older_methods('dummy', tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step_ml_methods, model_save_name=dummy_save_name)
        linear = train_older_methods('linear', tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step_ml_methods, model_save_name=linear_save_name)
    if load_older:
        with open(PATH + 'project_files/' + dummy_save_name + '.pkl', 'rb') as file: dummy = pickle.load(file)
        with open(PATH + 'project_files/' + linear_save_name + '.pkl', 'rb') as file: linear = pickle.load(file)
    
        

    if evaluate_models:
        number_of_starting_points = 100 # 32000  καθορίζει πόσα τυχαία σημεία έναρξης της πρόβλεψης θα παρθούν για την παραγωγή των λιστών της κάθε μετρικής ή στατιστική σύγκριση των μεθόδων
        num_gen_points = output_size # 3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
        if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό

        # load testing data
        print('\nLoad test series in order to test and compare the trained algorithms')
        # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
        # μπορείς να προσθέσεις περισσότερες από μια testing χρονοσειρές, για ακόμα μεγαλύτερη στατιστική γενικευσιμότητα των αποτελεσμάτων
        if not(remote_PC): test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) 
        if remote_PC: test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
        print('length of test series is ', test_series.shape); print('\n\n')
    
        ## Evaluate LSTM forecasting effeciency
        # μη χρησιμοποιήσεις τα δεδομένα του testing για να επιλέξεις καλύτερο μοντέλο. Το μοντέλο πρέπει να είναι τυφλό στα testing data και για αυτό οι μετρικές δεν μπαίνουν στο training_log
        if train_LSTM or load_lstm: evaluate_lstm(lstm_model, test_series, num_gen_points, number_of_starting_points, scaling_method, make_barplots=True)


        # compare statisticaly different methods
        if ((train_LSTM or load_lstm) and (train_older or load_older)):
            for metric_used in ['MAE', 'norm-cross-corr', 'RMS-PSD']:
                # testing parameters
                # test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # φορτώνεται πιο πάνω
                
                # LSTM - Dummy comparison
                lstm_MAE_list, dummy_MAE_list = produce_metric_samples(test_series, lstm_model, dummy, number_of_starting_points, num_gen_points, scaling_method, metric=metric_used)
                statistical_comparison(lstm_MAE_list, dummy_MAE_list, normality_test ='SW', comparing_name = 'Dummy-regressor', metric_name=metric_used, plot_visuals=False)
                # LSTM - Linear comparison
                lstm_MAE_list, linear_MAE_list = produce_metric_samples(test_series, lstm_model, linear, number_of_starting_points, num_gen_points, scaling_method, metric=metric_used)
                statistical_comparison(lstm_MAE_list, linear_MAE_list, normality_test ='SW', comparing_name = 'Autoregressive', metric_name=metric_used, plot_visuals=False)
                # LSTM - pink noise comparison
                lstm_MAE_list, noise_MAE_list = produce_metric_samples(test_series, lstm_model, 'pink_noise', number_of_starting_points, num_gen_points, scaling_method, metric=metric_used)
                statistical_comparison(lstm_MAE_list, noise_MAE_list, normality_test ='SW', comparing_name = 'pink noise', metric_name=metric_used, plot_visuals=False)

            # visual presentations of forecasting
            starting_point = np.random.randint(lstm_model.seq_len, test_series.shape[1]) # εδώ θα χρειαστεί μόλις ένα σημείο για visualization
            num_gen_points = output_size #3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
            if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό
            test_signal = test_series[0,:]
            fs = 1/(test_series[1,3] - test_series[1,2])
            actual_signal = test_signal [starting_point : starting_point + num_gen_points]
            lstm_gen_signal = lstm_generate_lfp(lstm_model, test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=1)
            linear_gen_signal = ml_generate_lfp(linear, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
            dummy_gen_signal = ml_generate_lfp(dummy, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
            noise = produce_noise (beta = 1, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
            gen_noise_signal = noise * (actual_signal.std()/noise.std()) # bring the noise to the same scale as the signal
            visual_fc_comparison(actual_signal, fs, lstm_gen_signal, ml_method1='LSTM', domain='both', gen_signal2=linear_gen_signal, ml_method2='Autoregressive')
            visual_fc_comparison(actual_signal, fs, lstm_gen_signal, ml_method1='LSTM', domain='both', gen_signal2=dummy_gen_signal, ml_method2='Dummy-regressor')
            visual_fc_comparison(actual_signal, fs, lstm_gen_signal, ml_method1='LSTM', domain='both', gen_signal2=gen_noise_signal, ml_method2='pink noise')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def LSTM_train(tag, downsample_scale, sliding_window_step, hidden_state_dim, input_size, output_size, num_layers, batch_size, lr, momentum, epochs, scaling_method, save_load_model_number):
    print('\nTrain LSTM-RNN:')

    # creates a folder to save the new trained model
    created_folder = False
    if save_load_model_number == 0: created_folder = True # ο φάκελος '0' θα είναι ο προτυπος φάκελος για να μη φτιάχνονται ασκοπα καινούργιοι
    while created_folder == False:
        newpath = PATH + '/project_files/models/model' + str(save_load_model_number) + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            created_folder = True
        else:
            save_load_model_number = save_load_model_number + 1

    # Import & prepare data
    save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
    lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών
    train_loader, val_loader, _  = prepare_data(lfp_data, input_size, output_size, sliding_window_step, scaling_method, batch_size=batch_size)
    print('Data loaders have been prepared')

    ## NN instance creation
    lstm_model_init = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size)
    if loss_function_used=='mse': criterion = nn.MSELoss()
    if loss_function_used=='mae': criterion = nn.L1Loss()
    if loss_function_used=='huber': criterion = nn.HuberLoss()
    if optimizer_used=='sgd': optimizer = optim.SGD(lstm_model_init.parameters(), lr, momentum)
    if optimizer_used=='adam': optimizer = optim.Adam(lstm_model_init.parameters(), lr)

    # # try forward method
    # a=np.ones((14,input_size,1)); a=torch.tensor(a, dtype=torch.float32); print(a.shape); 
    # arr = lstm_model_init(a) # input must be dims (batch_size, sequence_length, input_size=1) 
    # print('arr output shape is', arr.shape); print(arr)

    # train lstm and save it
    if move_to_gpu != 'None': lstm_model_init = lstm_model_init.to(device)
    lstm_model, training_string, model_val_score = training_lstm_loop(lstm_model_init, criterion, optimizer, epochs, train_loader, val_loader, save_load_model_number) 
    create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, momentum, sliding_window_step, scaling_method, tag, input_size, output_size, training_string, save_load_model_number)
    save_params(tag, downsample_scale, scaling_method, input_size, output_size, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, training_string, save_load_model_number)
    return lstm_model, model_val_score

#--------------------------------------------------------------------------------------------

def prepare_data(lfp_data_matrix, input_size, output_size, window_step, scaling_method, prepare_data_for_lstm=True, batch_size=1):
    """This function prepares the data (i.e. normalizes, divide long signals, creates windowed data, wraps them into loaders) and returs the train_loader and val_loader 
    objects that will be used to feed the batces in the LSTM during the training loop"""

    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    window_size = input_size + output_size

    # Δημιουργία δεδομένων εκπαίδευσης με κόψιμο τους σε παράθυρα όπου κάθε παράθυρο περιλαμβάνει τα target_data και input_data μιας forecasting δοκιμής/εκπαίδευσης
    if not(prepare_data_for_lstm):
        windowed_data = np.lib.stride_tricks.sliding_window_view(lfp_data_matrix, window_size, axis=1, writeable=True)[:,::window_step,:]
        windowed_data = np.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1],window_size)) # πλέον με το reshape τα windows παύουν έτσι κι αλλιώς να είναι views
        print('numpy windowed data are', windowed_data.shape)
        windowed_data  = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά
        input_data = windowed_data[:,0:input_size]
        target_data = windowed_data[:,input_size:window_size]
        return input_data, target_data, scaler

    if prepare_data_for_lstm:  
        data = torch.from_numpy(lfp_data_matrix).float()
        windowed_data = data.unfold(dimension=1, size = window_size, step = window_step)
        # windowed_data = windowed_data.contiguous()
        # windowed_data = windowed_data.view((windowed_data.shape[0]*windowed_data.shape[1],window_size,1))
        windowed_data = torch.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1],window_size)) # πλέον με το reshape τα windows παύουν έτσι κι αλλιώς να είναι views
        windowed_data = windowed_data.numpy()
        windowed_data  = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά
        windowed_data = torch.from_numpy(windowed_data).float()
        windowed_data = torch.unsqueeze(windowed_data, 2)
        print('torch windowed data are', windowed_data.shape)
        input_data = windowed_data[:,0:input_size,:]
        target_data = windowed_data[:,input_size:window_size,:]

        if tf_like_output: target_data = windowed_data[:,1:output_size+1,:] # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση
        if move_to_gpu == 'all': input_data=input_data.to(device); target_data=target_data.to(device) # εδώ τα data σίγουρα παύουν να είναι views. Για αυτό πιο κάτω στο training μόνο σε αυτή την περίπτωση τα batches δεν αντιγράφονται
        dataset = torch.utils.data.TensorDataset(input_data, target_data)
        train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.8, 0.2])
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=train_data.__len__()
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=val_data.__len__()
        return train_loader, val_loader, scaler

#--------------------------------------------------------------------------------------------

### Architecture (class) of the LSTM-based-neural-network
class LSTM_fc(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depends from the output size and the target data """
    def __init__(self, seq_len, hidden_size, num_layers, output_size):
        super(LSTM_fc, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.output_size = output_size
        if bidirectional: self.bidirectional = 2
        if not(bidirectional): self.bidirectional = 1

        self.lstm=nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional) # nn.LSTM has dynamic layers throught the num_layer parameter which creates stacked LSTM
        self.linear1 = nn.Linear(self.bidirectional * self.num_layers, 1) # combines all h_t's (inverse and straight form all layers) into one 
        self.linear2 = nn.Linear(self.hidden_size, self.output_size) # from the h_t's it produces the final output
        self.linear3 = nn.Linear(self.bidirectional*self.hidden_size, 1) # combines straight and inverse h_t's
        self.linear4 = nn.Linear(self.seq_len, self.output_size) # combines the points from time steps into final output


    def forward(self, x):
        # out is all (h_t) from the last layer of the LSTM for each time step (num_layers doesnt change it). In bidirectional LSTM it cnotains conactenated straight h_n and inverse h_n for each time step (in the 3rd axis)
        # h_n contains the h_n from last time step. It contains h_n of last time steps for all layer. It contains strait and inverse h_t if LSTM is bidirectional. so its 1st axis is bidirectional*num_layers
        out, (h_n, _) = self.lstm(x) # out dims (batch_size, L, hidden_size) if batch_first=True
        if lstm_seq_type == 'seq2one': 
            x = h_n # because 'seq2one' I keep only the last h_n, throwing the other time_steps
            x = torch.transpose(x, 0, 1) # put the batches in first axis
            x = torch.transpose(x, 1, 2) # swap in order to be appropriate for linear layers
            x = self.linear1(x) # if bidirectional = False & num_layers = 1, this does nothing (it has 1 input to 1 output)
            x = torch.squeeze(x)
            x = self.linear2(x)
        if lstm_seq_type == 'seq2seq': 
            x = out # because seq to one I keep only the  h_n's from all time steps
            x = self.linear3(x)
            x = torch.squeeze(x)
            x = self.linear4(x)
        return x

#--------------------------------------------------------------------------------------------

### train & validate the LSTM model
def training_lstm_loop(model, criterion, optimizer, num_epochs, train_loader, val_loader, model_number):
    """This is the NN training function. It recieves the typical parameters fo model, loss function (criterion), optimizer and epochs. 
    It also recieves as input the torch dataloader objects for training ana validation data. """

    print('start training')
    epoch_val_loss = 10^6 # initialization for the loop
    train_losses_list=[]
    val_losses_list=[]
    training_string =''
    time_str='' # initialization if measure_time = False

    for epoch in range(num_epochs):
        # train_time = []
        model.train()
        t1 = time.perf_counter()
        batch_train_losses = []
        for x_train, y_train in train_loader:
            if move_to_gpu != 'None': x_train = x_train.to(device); y_train = y_train.to(device)
            train_pred = model(x_train)
            train_pred = torch.squeeze(train_pred); y_train = torch.squeeze(y_train)
            loss = criterion (y_train, train_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_train_losses.append(loss.item()) # list of train_losses for every batch
        t2 = time.perf_counter()
        epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses) # mean of train_losses of all batches in every epoch
        train_time = t2 - t1

        model.eval()
        epoch_val_loss_old = epoch_val_loss
        batch_val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                if move_to_gpu != 'None': x_val = x_val.to(device); y_val = y_val.to(device)
                test_pred = model(x_val)
                train_pred = torch.squeeze(train_pred); y_val = torch.squeeze(y_val)
                val_loss = criterion (y_val, test_pred)
                batch_val_losses.append(val_loss.item()) # list of val_losses for every batch
            epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses) # mean of train_losses of all batches in every epoch
        
        epoch_str = f'Epoch:{epoch+1}/{num_epochs} -> train (batch mean) loss = {epoch_train_loss} - val (batch mean) loss = {epoch_val_loss}'
        time_str = f'train_time: {train_time}'
        print(epoch_str + ' - ' + time_str)
        training_string = training_string + '\n' + epoch_str + ' - ' + time_str
        train_losses_list.append(epoch_train_loss)
        val_losses_list.append(epoch_val_loss)

        if  epoch_val_loss < epoch_val_loss_old: 
            torch.save(model.state_dict(), PATH + 'project_files/models/model' + str(model_number) + '/LSTM_forecasting_model.pt')
            #torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.input_size, 'hidden_size':model.hidden_size, 'num_layers':model.num_layers, 'output_size':model.output_size}}, PATH + 'project_files/MLP_fc_regressor.pt')
            best_model = model
            best_val_score = epoch_val_loss
        
    ### plot train and validation losses
    plt.plot(range(num_epochs), train_losses_list, label = 'Train loss')
    plt.plot(range(num_epochs), val_losses_list, label = 'Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('LSTM training - Loss to Epochs diagram')
    plt.legend()
    plt.savefig(PATH + 'project_files/models/model' + str(model_number) + '/loss_to_epoch.png')
    plt.show()
    plt.close()
        
    return best_model, training_string, best_val_score


#--------------------------------------------------------------------------------------------

### creates and saves report of the training of the LSTM model to a text file
def create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, momentum, window_step, scaling_method, tag, input_size, output_size, training_string, model_number):
    """This function creates a small text file with the parameters and the results of the training"""
    
    files_string = f'files used: {tag}'
    ds_string = f'Downsampling: {downsample_scale}'
    scaling_string = f'normalization method: {scaling_method}'
    window_string = f'sliding window step: {window_step}'

    if move_to_gpu == 'all': training_method_string = 'training_method: All data passed to GPU in the beggining'
    elif move_to_gpu == 'batches': training_method_string = 'training_method: Batches are passed to GPU seperately'
    else: training_method_string = f"training_method: 'cpu'"
    bidirectional_str = f'The LSTM is bidirectional: {bidirectional}'
    train_lstm_str = f'train with only last sequence (h_n) or all sequences: {lstm_seq_type}'
    fc_move_by_one_string = f'tf_like_output: {tf_like_output}'
    optimizer_str = f'Optimizer used for training: {optimizer_used}'
    loss_str = f'Loss function used for training: {loss_function_used}'

    input_string = f'input size: {input_size}'
    output_string = f'output size: {output_size}'
    hidden_size_string = f'Size of LSTM hidden state: {hidden_state_dim}'
    layers_string = f'Number of stacked LSTM layers: {num_layers}'
    batch_string = f'Size of batches: {batch_size}'
    lr_string = f'Learning rate: {lr}'
    mom_string = f'Momentum: {momentum}'
    
    data_string = (files_string +'\n'+ds_string +'\n'+scaling_string +'\n'+window_string)
    globals_string = (training_method_string +'\n'+bidirectional_str +'\n'+train_lstm_str +'\n'+fc_move_by_one_string +'\n'+optimizer_str +'\n'+loss_str)
    train_params_string = (input_string +'\n'+output_string +'\n'+hidden_size_string +'\n'+layers_string +'\n'+batch_string +'\n'+lr_string +'\n'+mom_string)

    whole_string = (data_string +'\n\n'+ globals_string +'\n\n'+ train_params_string +'\n\n\n'+ training_string)

    # whole_string = (ds_string + '\n'+hidden_size_string + '\n'+layers_string + '\n'+batch_string + '\n'+lr_string + '\n'+ mom_string + '\n'+window_string + '\n'+scaling_string + 
    #                 '\n\n'+files_string + '\n'+input_string + '\n'+output_string + '\n\n'+fc_move_by_one_string +'\n'+training_method_string + 
    #                 '\n' + train_lstm_str + '\n' + bidirectional_str + '\n\n\n'+ training_string)
    with open(PATH + '/project_files/models/model' + str(model_number) + '/training_log.txt', "w+") as file: file.write(whole_string)


# load the LSTM model if you have saved it, in order not to run training again if its time-consuming
def LSTM_load(model_number, input_size, hidden_state_dim, num_layers, output_size):
    model = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size) 
    model.load_state_dict(torch.load(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_forecasting_model.pt'))
    print('LSTM model has been loaded')
    return model


### saves the parameters of the LSTM model to a dictionary and then saves the dictionary to a picle file
def save_params(tag, downsample_scale, scaling_method, input_size, output_size, hidden_state_dim, num_layers, batch_size, lr, window_step, training_string, model_number):
    dict_param = {'tag':tag, 'downsample_scale':downsample_scale, 'scaling_method':scaling_method, 'input_size':input_size, 'output_size':output_size, 
    'hidden_state_dim':hidden_state_dim, 'num_layers':num_layers, 'batch_size':batch_size, 'lr':lr, 'window_step':window_step, 'training_string':training_string }
    with open(PATH + '/project_files/models/model' + str(model_number) + '/LSTM_params.pkl', "wb") as file: pickle.dump(dict_param, file)


### loads the saved parameters of the LSTM model from the picled dictionary file
def load_params(model_number):
    with open(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_params.pkl', 'rb') as file: dict_param = pickle.load(file)
    return dict_param

#--------------------------------------------------------------------------------------------

### Generate/forecast LFP signal with the LSTM model
def lstm_generate_lfp(model, signal, num_gen_points:int, scaling_method, only_gen_signal:bool):
    """""This function ganerates a number of points (num_gen_points), in the end of the given LFP signal (signal) by using a trained NN (model)
        1) model -> is the LSTM forecasting model
        2) signal -> must be an lfp signal in tensor form. It should be bigger in length than the input_size of the LSTM model
        3) num_gen_points -> is the number of the points that will be generated/forecasted in total
        4) scaling_method -> the method of scaling the data, all the availble methods exist on the signal_handler.scaler() object
        5) if only_gen_signal is False the functios put the generated signal in the end of the input signal. If it's True it returns only the generated signal ommiting the input signal
    """""

    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    model = model.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
    if torch.is_tensor(signal):
        signal = signal.to('cpu') # στέλνει το σήμα στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
        signal = signal.cpu().numpy()
    else: signal = np.float32(signal)
    signal = scaler.fit_transform1d(signal) # κάνονικοποιεί τo σήμα-input με τον ίδιο τρόπο που έχει μάθει να δέχεται κανονικοποιημένα inputs το LSTM
    if not(only_gen_signal): generated_signal= list(signal) # αν θέλουμε το παραγώμενο σήμα να περιέχει το input
    if only_gen_signal: generated_signal=[] # αν θέλουμε το παραγώμενο σήμα να περιέχει μονο το generated χωρίς το input
    fc_repeats = int(num_gen_points/model.output_size) + 1 # παράγει μερικά παραπάνω σημεία και κόβει τα τελευταία για να μπορεί να παράγει σημεία που δεν είναι πολλπλάσια του output_size
    if tf_like_output: fc_repeats = num_gen_points
    
    model.eval()
    signal =  torch.from_numpy(signal)
    starting_signal = signal[(len(signal)-model.seq_len):] # παiρνει τα τελευταία σημεία του σήματος (τόσα όσο είναι το input του model) για να παράξει τη συνέχεια του σήματος
    for i in range(fc_repeats): # παράγει το output, το κάνει λίστα αριθμών και επεικείνει με αυτό, τη generated_signal
        starting_signal_input=torch.unsqueeze(starting_signal, 1)
        starting_signal_input=torch.unsqueeze(starting_signal_input, 0)
        output = model(starting_signal_input)
        output = torch.squeeze(output)
        if tf_like_output: output = output[-1]
        if output.shape == torch.Size([]): output = torch.unsqueeze(output, 0) # αυτό χρειάζεται όταν το output εχει διάσταση 1
        generated_signal = generated_signal + list(output.detach().numpy()) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης
        if not(tf_like_output): starting_signal = torch.cat((starting_signal, output), dim=0)[model.output_size:] # κατασκευή νέου input για το model
        if tf_like_output: starting_signal = torch.cat((starting_signal, output), dim=0)[1:] # κατασκευή νέου input για το model
    generated_signal = np.array(generated_signal) # η λίστα generated_signal μετατρέπεται σε np.ndarray
    if not(only_gen_signal): generated_signal = generated_signal[: signal.shape[0] + num_gen_points] # κρατιοούνται μόνο τα σημεία που ζητήθηκαν να παραχθούν (είχαν παραχθεί λίγα περισσότερα, που είναι πολλαπλάσια του LSTM output)
    if only_gen_signal: generated_signal = generated_signal[:num_gen_points] # αν θέλουμε το παραγώμενο σήμα να περιέχει μονο το generated χωρίς το input χρειάζεται κι αυτή η εντολή
    generated_signal = scaler.inverse1d(generated_signal) # αποκανονικοποίηση του τελικού αποτελέσματος για να είναι στην κλίμακα του LFP
    return generated_signal

#--------------------------------------------------------------------------------------------

###  test trained LSTM metrics and visualizations
def evaluate_lstm(lstm_model, test_series, num_gen_points, number_of_starting_points, scaling_method, make_barplots, return_metrics = 'None'):
    """Uses a test series independent from the validation data, to test how effective a trained forecasting LFP LSTM model is"""
    print('\nTest trained LSTM: Compare actual and generated signal')

    input_size = lstm_model.seq_len
    output_size = lstm_model.output_size
    if number_of_starting_points == 1: make_barplots = False # για ένα σημείο δεν παράγεται κάποιο σχήμα που να έχει νόημα, οπότε η εντολή απενεργοποιείται

    index_range = np.arange(input_size, test_series.shape[1] - num_gen_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    starting_points_list = np.random.choice(index_range, size = number_of_starting_points, replace=False)
    
    if make_barplots == True: print('Barplots of the metrics on different starting signals are being plotted')
    MAE_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE', make_barplot=make_barplots)
    RMSE_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMSE', make_barplot=make_barplots)
    norm_cross_corr_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='norm-cross-corr', make_barplot=make_barplots)
    max_cross_cor_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='max Cross-cor', make_barplot=make_barplots)
    RMS_PSD_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMS-PSD', make_barplot=make_barplots)

    if number_of_starting_points == 1: title_str = f"The following metrics are produced from the forecasting of point {starting_points_list[0]} as starting point"
    else: title_str = f"The following metrics are means of the produced metrics from the forecasting of {number_of_starting_points} random starting points"
    MAE_str = f'Absolute mean error {MAE_list.mean()}'
    RMSE_str = f'Root mean square error is {RMSE_list.mean()}'
    norm_cross_corr_str = f'Pearson r (normalized cross-correlation of zero phase) is {norm_cross_corr_list.mean()}'
    max_cross_cor_str = f'Maximum cross-correlation is {max_cross_cor_list.mean()}'
    RMS_PSD_str = f'Root mean square error of PSD is {RMS_PSD_list.mean()}'

    testing_string = title_str + '\n' + MAE_str + '\n' + RMSE_str + '\n' + norm_cross_corr_str + '\n' + max_cross_cor_str + '\n' + RMS_PSD_str
    print(testing_string)

    # visual representation on a random point of the signal
    starting_point = starting_points_list[0]
    print(f'Create visual representation of the forecasting from the point {starting_point} of the test series')
    test_signal = test_series[0,:]
    tensor_test_signal = torch.from_numpy(test_signal).clone().float()
    fs = 1/(test_series[1,3] - test_series[1,2])
    actual_signal = test_signal [starting_point : starting_point + num_gen_points]
    lstm_gen_signal = lstm_generate_lfp(lstm_model, tensor_test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=1)
    visual_fc_comparison(actual_signal, fs, lstm_gen_signal, 'lstm', domain = 'both')
    visual_fc_comparison(actual_signal, fs, lstm_gen_signal, 'lstm', domain = 'cross-correlation')

    if return_metrics == 'lists': return MAE_list, RMSE_list, norm_cross_corr_list, max_cross_cor_list, RMS_PSD_list
    if return_metrics == 'means': return MAE_list.mean(), RMSE_list.mean(), norm_cross_corr_list.mean(), max_cross_cor_list.mean(), RMS_PSD_list.mean()
    if return_metrics == 'string': return testing_string # μη χρησιμοποιήσεις τα δεδομένα του testing για να επιλέξεις καλύτερο μοντέλο. Το μοντέλο πρέπει να είναι τυφλό στα testing data και για αυτό οι μετρικές δεν μπαίνουν στο training_log


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### εκπαιδεύει στο forecasting με άλλες μεθόδους πέραν του LSTM. Μπορείς να προσθέσεις και άλλες μεθόδους αν θέλεις.
def train_older_methods(ml_method, tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step, model_save_name:str):
    print('\nTrain ' + ml_method + ' regressor:')
    
    save_load_path= PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών
    x_data, y_data, _  = prepare_data(lfp_data, input_size, output_size, sliding_window_step, scaling_method, prepare_data_for_lstm=False)

    if ml_method == 'linear':
        model = lrm.LinearRegression()
        name_string = 'Linear regression'
    elif ml_method == 'dummy':
        model = DummyRegressor(strategy='mean')
        name_string = 'Dummy regressor'

    model.fit(x_data, y_data)
    # model.fit(x_train, y_train) # δε χρειάζεται αφού τελικά τα testing data είναι άλλα
    # print(name_string + ' R^2 score is ', model.score(x_test, y_test)) # δε χρειάζεται αφού τελικά τα testing data είναι άλλα
    # pred = model.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting, αλλά εδώ τεθηκε μόνο για έλεγχο

    if model_save_name != 'None': 
        with open(PATH + 'project_files/' + model_save_name + ".pkl", "wb") as file: pickle.dump(model, file)
        
    return model

#--------------------------------------------------------------------------------------------

def ml_generate_lfp(model, signal:np.ndarray, input_size:int, output_size:int, num_gen_points:int, scaling_method, only_generated:bool):
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    num_predictions =  int(num_gen_points/output_size) + 1
    signal = scaler.fit_transform1d(signal) # κανονικοποίηση
    input_signal = signal[(len(signal)-input_size) :] # λήψη των τελευταιων σημείων του σήματος για forecasting
    input_signal = input_signal.reshape(1,-1) # πρέπει να είναι σε αυτή τη μορφή για να εισαχθεί στο predict
    if only_generated: generated_signal = []
    if not(only_generated): generated_signal= list(signal)

    for i in np.arange(num_predictions):
        new_pred = model.predict(input_signal)
        if new_pred.shape == (1,): new_pred = new_pred.reshape(1,1) # για output = 1;;;;
        input_signal = np.hstack((input_signal[:,output_size:], new_pred))  # θα ενώσω τα δύο κομμάτια μετακινόντας το κάποιες θέσεις
        if new_pred.shape == (1,1): generated_signal.append(np.squeeze(new_pred)) # για output = 1 
        else: generated_signal = generated_signal + list(np.squeeze(new_pred))
    generated_signal = np.array(generated_signal)
    if not(only_generated): generated_signal =  generated_signal[: len(signal)+num_gen_points]
    if only_generated: generated_signal =  generated_signal[:num_gen_points]
    generated_signal = scaler.inverse1d(generated_signal)
    return generated_signal


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visual_fc_comparison(actual_signal, fs, gen_signal1, ml_method1:str, domain, save=save_plots, gen_signal2='None', ml_method2='None'):
    # compare the time-series (i.e. comparison in the time domain)
    if domain in ['time', 'both']: 
        plt. plot(actual_signal, label = 'actual signal')
        plt. plot(gen_signal1, label = ml_method1 +' generated signal')
        if not isinstance(gen_signal2, str): plt. plot(gen_signal2, label = ml_method2 +' generated signal')
        plt.legend()
        plt.title('Visual comparison between actual and generated signal(s)')
        if ml_method2 == 'None': ml_method2 = ''
        if save: plt.savefig(PATH + 'project_files/dipl_images/' + ml_method1 + '-'+ ml_method2 +'_time_domain.png')
        plt.show()
        plt.close()

        # compare the two time-series' frequencies (i.e. comparison in the frequency domain)
    if domain in ['frequency', 'both']:
        f1, Pxx_1 = sn.periodogram(actual_signal, fs=fs, return_onesided=True, scaling='density')
        f2, Pxx_2 = sn.periodogram(gen_signal1, fs=fs, return_onesided=True, scaling='density')
        if not isinstance(gen_signal2, str): f3, Pxx_3 = sn.periodogram(gen_signal2, fs=fs, return_onesided=True, scaling='density')
        plt.plot(f1,Pxx_1, label = 'actual signal')
        plt.plot(f2,Pxx_2, label = ml_method1+' generated signal')
        if not isinstance(gen_signal2, str): plt.plot(f3,Pxx_3, label = ml_method2+' generated signal')
        plt.suptitle('comparison of generated and actual signal frequencies (Fourier-PSD)')
        # plt.title(f'sampling frequency is {fs} due to downsampling', fontsize = 9)
        plt.legend()
        if ml_method2 == 'None': ml_method2 = ''
        if save: plt.savefig(PATH + 'project_files/dipl_images/' + ml_method1 + '-'+ ml_method2 +'_freq_domain.png')
        plt.show()
        plt.close()

    # crate the diagram of the cross-correlation of the singals (actual & model_generated)
    if domain == 'cross-correlation':
        if gen_signal2!='None' or ml_method2!='None': print('This method plots cross-correlation only for one signal')
        sn_corr = norm_cross_cor(actual_signal, gen_signal1)
        plt.plot(sn_corr); plt.title('normalized cross correlation'); plt.show(); plt.close() # η corss-correlation ΕΙΝΑΙ κανονικοποιήμένη στο [-1,1]


#--------------------------------------------------------------------------------------------

def produce_metric_list(model, model_type, test_series, starting_points_list, num_gen_points, input, output, scaling_method, metric, make_barplot):
    """This function recieves an ML model, a test series and a list of initiating points. Then it generates a signal from each initiating point and produces a metric of 
    comparison of the generated signal and the actual following signal. It then collects all these metrics from all the initiating points to a list. This is a random samling 
    method for collecting a sample of metrics from random initiating points, in order to be used for statistical hypotheses."""
    
    starting_points_list = np.array(starting_points_list)
    metric_array_list = np.zeros(len(starting_points_list))
    fs = 1/(test_series[1,3] - test_series[1,2])
    test_signal = test_series[0,:]

    for idx, starting_point in enumerate(starting_points_list):
        actual_signal = test_signal [starting_point : starting_point + num_gen_points]
        if model_type == 'lstm': gen_signal = lstm_generate_lfp(model, test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=True)
        if model_type == 'ml': gen_signal = ml_generate_lfp(model, test_signal[:starting_point], input, output, num_gen_points, scaling_method, only_generated = True)
        if model_type == 'white_noise': noise = produce_noise (beta = 0, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
        if model_type == 'pink_noise': noise = produce_noise (beta = 1, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
        if model_type == 'brownian_noise': noise = produce_noise (beta = 2, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
        if model_type in ['white_noise', 'pink_noise', 'brownian_noise']: gen_signal = noise * (actual_signal.std()/noise.std()) # bring the noise to the same scale as the signal

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
        # # άλλες μετρικές που είχες χρησιμοποιήσει, αλλά τις άφησες στην άκρη
        # np_correlate = np.correlate(pure_base_signal/pure_base_signal.std(), pure_gen_signal/pure_gen_signal.std())/len(pure_base_signal) # η μορφή αυτή είναι κανονικοποιήμένη
        # np_corrcoef = np.corrcoef(pure_base_signal, pure_gen_signal)[0,1]
        # spearman_rho = stats.spearmanr(pure_base_signal, pure_gen_signal)

        metric_array_list[idx] = model_metric
    if make_barplot: make_metric_barplot(starting_points_list, metric_array_list, metric)
    return metric_array_list

#--------------------------------------------------------------------------------------------

def produce_metric_samples(test_series, lstm_model, comparing_model, number_of_starting_points, num_gen_points, scaling_method, metric):
    """This function takes a test_series, an lstm_model, a number of starting points and the name of a metric and it creates two lists. The function crates random starting
    points for forecasting, forecasts the test series with the LSTM and the other comparing methods, and it produces a list of the calculated metric produced by the
    comparison of the forecasted signal and the actual signal in each starting point"""

    input_size = lstm_model.seq_len
    output_size = lstm_model.output_size

    index_range = np.arange(input_size, test_series.shape[1] - num_gen_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    starting_points_list = np.random.choice(index_range, size = number_of_starting_points, replace=False)

    lstm_metric_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    if comparing_model in ['white_noise', 'pink_noise', 'brownian_noise']:
        comparing_metric_list = produce_metric_list('None', comparing_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    else:
        comparing_metric_list = produce_metric_list(comparing_model, 'ml', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    
    return lstm_metric_list, comparing_metric_list

#--------------------------------------------------------------------------------------------

def statistical_comparison(lstm_metric_list, comparing_metric_list, normality_test, comparing_name:str, metric_name:str, plot_visuals = True):
    '''Υπαρχουν 3 κριτήρια που πρέπει να πληρούνται για τη χρήση παραμετρικών κριτηρίων όπως το t-test: 1) οι κατανομές των δειγμάτων να είναι κανονικές, 2) οι κατανομές να 
    έχουν ίσες διακυμάνσεις (κάτι που δε χρειάζεται στα εξαρτημένα δείγματα), και τα δεοδμένα να είναι ποσοτικά. Οπότε ουσιαστικά εδω πρέπει να ελεγχθει μόνο η κανονικότητα'''
    '''The Shapiro–Wilk test is more appropriate method for small sample sizes (<50 samples) although it can also be handling on larger sample size while Kolmogorov–Smirnov 
    test is used for n ≥50'''

    print(f'\nCOMPARISON OF LSTM FORECASTING WITH: {comparing_name} | METRIC USED: {metric_name}')
    lstm_metric_list = np.array(lstm_metric_list)
    comparing_metric_list = np.array(comparing_metric_list)
    print('LSTM metric computed mean is', lstm_metric_list.mean())
    print('2nd method metric computed mean is', comparing_metric_list.mean())
    print('LSTM metric computed median is', np.median(lstm_metric_list))
    print('2nd method metric computed meadian is', np.median(comparing_metric_list))
    diffs = lstm_metric_list - comparing_metric_list

    if plot_visuals:
        # make histograms to inspect distributions
        fig, axes = plt.subplots(2)
        axes[0].hist(lstm_metric_list)
        axes[0].set_title(f'LSTM {metric_name} list - distribution')
        axes[0].set_xlabel('metric values')
        axes[0].set_ylabel('frequencies')
        axes[1].hist(comparing_metric_list)
        axes[1].set_title(comparing_name + metric_name + ' list - distribution')
        axes[1].set_xlabel('metric values')
        axes[1].set_ylabel('frequencies')
        fig.tight_layout()
        plt.show()
        plt.close()

        # make histogram of differences
        plt.hist(diffs)
        plt.title(f'Distribution of {metric_name} differences')
        plt.xlabel('metric_values')
        plt.ylabel('frequencies')
        plt.show()
        plt.close()
        
        # make boxplots
        dict_boxplot = {'LSTM metric list':lstm_metric_list, comparing_name+' metric list':comparing_metric_list}
        plt.boxplot(dict_boxplot.values(), labels=dict_boxplot.keys(), patch_artist=True)#, boxprops=dict(color='darkblue'), medianprops=dict(color='red'), whiskerprops=dict(color='yellow'))
        plt.xlabel('metrics by ML algorithm')
        plt.ylabel('metric values')
        plt.axhline(y=np.median(lstm_metric_list), linestyle = '--', color = '0.5')
        plt.axhline(y=np.median(comparing_metric_list), linestyle = '--', color = '0.5')
        plt.title(f'Metric: {metric_name}')
        plt.show()
        plt.close()


    if normality_test == 'SW':
        _ , p_norm = stats.shapiro(diffs)
    if normality_test == 'KS':
        _, p_norm = stats.kstest(diffs, 'norm')
    # print('NORMALITY P  ==== ', p_norm)
    _, p_skew = stats.skewtest(diffs)
    # print('SKEWNESS P  ==== ', p_skew)

    if p_norm > 0.01: # the null hypothesis that the diferences follow a normal distribution, cannot be rejected
        print ('The differences of metrics in random starting points are distributed normaly. t-test of related samples will be carried out')
        str_test = "Test: t-test"
        stat_test = stats.ttest_rel(lstm_metric_list, comparing_metric_list)
        p_test = stat_test.pvalue
        effect_size_method = "Cohen's d"
        effect_size = Cohens_d(lstm_metric_list, comparing_metric_list) # inerpretation: d=0.01 => no effect, d=0.2 => small effect, d=0.5 => medium effect, d=0.8 => large effect, d=1.2 => very large effect, d=2.0 => huge effect
    elif p_skew > 0.01: # the null hypothesis that the diferences come from a symmetric distribution, cannot be rejected
        print ('The differences of metrics in random starting points are not distributed normaly but are distributed symmetrically. Wilcoxon (T) will be carried out')
        str_test = "Test: Wilcoxon r"
        # if the null hypothesis is that a difference of a pair of samples is zero then the symmetry assumption is not required. However if the null hupothesis is the more general that differences as a whole have zero mean (thus the mean of the two paired samples ...
        # are equal) and that's the case here, then on that occastion, the symmetry assumption is required
        stat_test = stats.wilcoxon(lstm_metric_list, comparing_metric_list, method = 'approx') # method='approx' is used in order to return the z-statistic which is required for the effect size
        p_test = stat_test.pvalue
        z = stat_test.zstatistic
        effect_size_method = "Wilcoxon r effect size"
        effect_size = Wilcoxon_r(lstm_metric_list, comparing_metric_list, z) # inerpretation: abs(r)<0.1 => no effect, abs(r)=0.1 => small effect, abs(r)=0.3 => medium effect, abs(r)=0.5 => medium effect
    else: # the data are irregular. The test remained for usage is the sign-test
        print ('The differences of metrics in random starting points are neither distributed normaly nor distributed symmetrically. Sign-test will be carried out')
        str_test = "Test: sign test"
        stat_test = stats_ds.sign_test(diffs, mu0 = 0)
        p_test = stat_test[1]
        effect_size_method  = "Cohen's g"
        effect_size = Cohens_g(lstm_metric_list, comparing_metric_list) # inerpretation: g<0.05 => negligible, 0.05<g<0.15 => small, 0.15<g<0.25 => medium, g>0.25 => large
    print('results:', stat_test)
    print('p-value is', p_test)
    if p_test < 0.05: print('Thus there is statistically significant difference between the two means of the metrics')
    elif p_test >= 0.05: print('Thus the null hypothesis that means of metrics are equal, cannot be rejected')
    if p_test < 0.05: print(f'Effect size ({effect_size_method}) is {effect_size}')
    p_str = f'p-value: {p_test:.3f}'
    if p_test < 0.05: effect_size_str = f'Effect size ({effect_size_method}): {effect_size:.3f}'
    else: effect_size_str = 'Effect size: - '

    # make barplots with confidence intervals
    lstm_conf_int = stats.norm.interval(confidence=0.95, loc=lstm_metric_list.mean(), scale= stats.sem(lstm_metric_list))
    comp_conf_int = stats.norm.interval(confidence=0.95, loc=comparing_metric_list.mean(), scale=stats.sem(comparing_metric_list))
    lstm_conf_int = np.array(lstm_conf_int); comp_conf_int = np.array(comp_conf_int) # turn them to arrays for logical indexing
    comp_conf_int[comp_conf_int<0]=0; lstm_conf_int[lstm_conf_int<0]=0 # μηδενισμός αρνητικών ορίων εμπιστοσύνης
    labels = ['LSTM metric', f'{comparing_name} metric']
    means = [lstm_metric_list.mean(), comparing_metric_list.mean()]
    ci_min = [lstm_conf_int.min(), comp_conf_int.min()]
    ci_max = [lstm_conf_int.max(), comp_conf_int.max()]
    ci_list = [ci_min, ci_max]
    plt.bar(x=labels, height=means, yerr = ci_list, color = ['lightblue', 'green'])
    plt.xlabel('metric means')
    plt.title(f'Metric: {metric_name}')
    plot_str = str_test + '\n' + p_str + '\n' + effect_size_str
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.text(0.5, 0.9, s=plot_str, bbox=bbox, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    if save_plots: plt.savefig(PATH + 'project_files/dipl_images/'+ comparing_name +'_' + metric_name + '_barplot.png')
    plt.show()
    plt.close()

def Cohens_d(x1, x2):
    """This function computes the Cohen's d of two pairs  of samples. Cohen's d is the effect size metric used for paired t-test"""
    nx1 = len(x1)
    nx2 = len(x2)
    dof = nx1 + nx2 - 2
    s = np.sqrt(((nx1-1)*np.std(x1, ddof=1) ** 2 + (nx2-1)*np.std(x2, ddof=1) ** 2) / dof)
    cohen_d= np.abs(x1.mean()-x2.mean())/s
    return cohen_d

def Wilcoxon_r(x1, x2, z):
    """This function computes the Wilcoxon-r of two pairs  of samples. Wilcoxon-r is the effect size metric used for Wilcoxon signed-rank test"""
    nx1 = len(x1)
    nx2 = len(x2)
    r = z / (nx1 + nx2)
    return r

def Cohens_g(x1,x2):
    """This function computes the Cohen's g of two pairs  of samples. Cohen's g is proposed in literature as a way of assesing the effect size in the sign test"""
    diffs = x1-x2
    positive_values = np.sum(diffs>0)
    all_values = len(x1)
    proportion = positive_values / all_values
    cohen_g = np.abs(proportion - 0.5)
    return cohen_g
#--------------------------------------------------------------------------------------------

# Αυτή η μέθοδος αναπαριστά γραφικά, πόσο καλή είναι η προβλεψη κατά μήκος του test σήματος LFP (εφόσον χρησιμοποιείται ένα σήμα για testing)
def make_metric_barplot(starting_points_list, lstm_metric_list, metric, save=save_plots):
    """This function recicieves the initiating points for forcasting, the metric produced by the comparison of the forecasted and the actual signal, and plots a
    barplot of the metric values according to the initiating points. The purpose of this function is to visually present how effective is the forecasting method in different
    parts across the signal"""
    starting_points_list = np.array(starting_points_list)
    lstm_metric_list = np.array(lstm_metric_list)
    if metric == 'norm-cross-corr': lstm_metric_list = np.abs(lstm_metric_list) # κάνει τα αποτελέσματα μόνο θετικά ώστε αν είναι εύκολα ερμηνεύσιμα
    indices = starting_points_list.argsort() # it returns the indces that sort the starting_point_list
    starting_points_list_sorted = starting_points_list[indices] # the starting_point_list is sorted with the indeces
    starting_points_list_sorted_str = starting_points_list_sorted.astype(str) # the starting_point_list is made to strings in order to be used in barplot
    lstm_metric_list_sorted = lstm_metric_list[indices] # the lstm_metric_list is sorted with the indeces
    if len(lstm_metric_list_sorted) < 200: plt.bar(starting_points_list_sorted_str, lstm_metric_list_sorted)
    if len(lstm_metric_list_sorted) >= 200: plt.plot(starting_points_list_sorted, lstm_metric_list_sorted)
    plt.title(f'Metric: {metric}')
    if metric == 'max Cross-cor': plt.suptitle('The metric values have been tranformed into absolute values')
    plt.xticks(rotation = 'vertical' )
    if save: plt.savefig(PATH + 'project_files/dipl_images/lstm_'+ metric +'_.png')
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------

def produce_noise(beta, samples, fs, freq_range):
    """produces signal in the form of noise """
    def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
        sos = sn.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = sn.sosfiltfilt(sos, data)
        return filtered_data

    # beta ->  the exponent  [1.f^(beta)] -> beta = 0 => white noise, beta = 1 => pink noise, beta = 2 => brownian noise
    # samples -> number of samples to generate
    noise = cn.powerlaw_psd_gaussian(beta, samples)
    if fs>500: # αν το σήμα είναι έντονα downsampled (πάνω από 30 φορές) δεν έχει νόημα το bandpass στο θόρυβο, επειδή έχει μικρύνει πολύ και στο πραγματικό σήμα LFP
        noise = bandpass(noise, freq_range, fs, poles = 5) # κάνει bandpass το θόρυβο στις συχνότητες που έχει γίνει και το LFP για να είναι όμοιος με αυτό
    return noise

def norm_cross_cor(a,b):
    """computes normalised cross-correlation of two signals"""
    if bool(a.std()) and bool(b.std()):
        norm_a = (a - a.mean())/a.std()#* max(len(a),len(b))
        norm_b = (b - b.mean())/b.std()
        norm_a_to_length = norm_a/max(len(a),len(b))
        norm_cor = sn.correlate(norm_a_to_length, norm_b, 'full')
        return norm_cor
    else:
        print("Warning! One of the two signals is constant with std zero and it cannot be normalized. Un-normalized cross-correlation will be returned ")
        cor = sn.correlate(a, b, 'full')
        return cor

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if  __name__ == "__main__":
    
    if execute == 'main()':
        if save_terminal_output_to_file:
            output_file = PATH + 'project_files/dipl_images/fc_LFP_output.txt'
            with open(output_file, 'w') as file: 
                sys.stdout = file
                main()
                sys.stdout = sys.__stdout__
        else:
            main()
    if execute == 'multiple_trainings':
        # multiple LSTM trainings for remote computer
        tag= 'All_EA_WT_0Mg'
        downsample_scale = 10
        sliding_window_step = 3
        input_size = 300 
        hidden_state_dim = 64
        num_layers = 1 
        output_size = 1
        batch_size = 1024
        lr = 0.1
        momentum = 0.9
        epochs = 20
        scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
        scaling_method = scaling_method_list[4]
        save_load_model_number = 0 # καθορίζει ποιο LSTM μοντέλο θα φορτωθεί (η αποθήκευση γίνεται στο φάκελο και τα μεταφέρεις manually στους φακέλους model)

        # lstm_model, _ = LSTM_train(tag, downsample_scale, sliding_window_step, hidden_state_dim, input_size, output_size, num_layers, batch_size, lr, momentum, epochs, scaling_method, save_load_model_number)

        val_scores_list = []
        loop_parameter_list = ['robust_scaling', 'decimal_scaling', 'None']
        loop_parameter_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        # loop_parameter_list = [(50,1), (100,1), (150,10), (200,10), (200,50), (500,100), (500,150), (500,1)]
        parameter_tuned = 'neurons'
        for idx, hidden_state_dim in enumerate(loop_parameter_list):
            save_load_model_number = idx
            lstm_model, val_score = LSTM_train(tag, downsample_scale, sliding_window_step, hidden_state_dim, input_size, output_size, num_layers, batch_size, lr, momentum, epochs, scaling_method, save_load_model_number)
            val_scores_list.append(val_score)
        loop_parameter_list = [str(element) for element in loop_parameter_list] # turns xticks in strings    
        plt.plot(loop_parameter_list, val_scores_list)
        plt.title(f'Validation metric scores of the paremeters: {parameter_tuned}')
        plt.savefig(PATH + 'project_files/training_barplot')
        plt.show()
        print('mutiple training run has been completed')

        # save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
        # lfp_data = np.load(save_load_path)
        # print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών
        # train_loader, val_loader, _  = prepare_data2(lfp_data, input_size, output_size, sliding_window_step, batch_size, scaling_method, cut_with_numpy=0)

    