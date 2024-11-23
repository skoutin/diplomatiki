"""This file uses an LSTM-RNN for the forecasting of the LFP signal"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import signal_handler
torch.backends.cudnn.benchmark = True # added for the -> cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.This error may appear if you passed in a non-contiguous input

from sklearn import model_selection
import sklearn.linear_model as lrm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
import scipy.signal as sn 
import colorednoise as cn

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Παραλαγές της εκπαίδευσης οι οποίες μάλλον ΔΕ θα κρατηθούν στον τελικό κώδικα (GLOBAL VARIABLES)

remote_PC = False
if not(remote_PC): PATH = 'D:/Files/peirama_dipl/' # my PC path
if remote_PC: PATH = '/home/skoutinos/' # remote PC path

run_to_gpu_all = 0 # στέλνει όλα τα δεδομένα στη gpu πριν την εκπαίδευση, !!!!!! ΠΡΟΣΟΧΗ!! όπως έχεις γράψει τον κώδικα αν στείλεις όλα τα δεδομένα στη gpu τότε το
# normalization θα γίνει στη gpu που παίρνει πάρα πολύ χρόνο. Δες το training loop για να το καταλάβεις.
run_to_gpu_batch = 0 # στέλνει τα δεδομένα στη gpu ανά batch επειδή δε χωράνε όλα με τη μία
if run_to_gpu_all or run_to_gpu_batch : device = 'cuda' if torch.cuda.is_available() else 'cpu'
if run_to_gpu_all or run_to_gpu_batch : print(torch.cuda.get_device_name())

fc_move_by_one = 0 # generates the same number of points but moved by one position -> e.g. takes a 100 points and forecasts the last 99 points and 1 new point

## 4 επιλογές για scaling 
# 1) κάνεις  scaling όλα τα σήματα lfp σρην αρχή πριν τα κόψεις σε παράθυρα
# 2) κάνεις scaling τα batches πριν εισαχθούν στην εκπαίδευση
# 3) κάνεις layer normalization των input batches μέσα στο LSTM, οπότε θα πρέπει να κάνεις χωριστό normalization στο output batch
# 4) Δεν κάνεις καθόλου scaling στα δεδομένα
scalling_manner_list = ['norm_all_data', 'norm_batches', 'input_layer_norm_and_output_batch_norm', 'None'] #γενικά το 'input_layer_norm_and_output_batch_norm' δε λειτουργεί καλά
scalling_manner = scalling_manner_list[1]

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    tag= 'All_WT_0Mg'  # determines which groups of files will be loaded, and used for training
    downsample_scale = 10000 # determines how many time will the signal be downsampled
    sliding_window_step = 1 # this is the number of the window sliding for the creation of the windows that will be used for training

    input_size = 300 # this is the number of the input_data of the LSTM, i.e. the number of points used for forecasting
    hidden_state_dim = 2 # the size of the hidden/cell state of LSTM
    num_layers = 1 # the number of consecutive LSTM cells the nn.LSTM will have (i.e. number of stacked LSTM's)
    output_size = 100 # this is th number of output_data of the LSTM, i.e. the future points forecasted by the LSTM

    extract_data = 0
    batch_size = 1
    epochs = 2
    lr = 0.1 # optimizers learning rate
    train_LSTM = 1 # for True it trains the model, for False it loads a saved model # ΠΡΟΣΟΧΗ αν κάνεις load μοντέλο που το έχεις εκπαιδεύσει με άλλο output_type προφανώς θα προκύψει σφάλμα
    train_older = 1 # trains linear (autoregresson) and dummy regresson
    save_load_model_number = 0 # καθορίζει ποιο LSTM μοντέλο θα φορτωθεί (η αποθήκευση γίνεται στο φάκελο και τα μεταφέρεις manually στους φακέλους model)

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    

    if fc_move_by_one: input_size = 100 
    if fc_move_by_one: output_size = input_size # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση


    # # Import data
    # if not(remote_PC): save_load_path= PATH + 'project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    # if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path) 
    # if not(extract_data): lfp_data = np.load(save_load_path)
    # print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    # # prepare data
    # if train_LSTM: train_loader, val_loader, scaler  = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size)

    # ## NN instance creation
    # lstm_model_init = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size)
    # criterion = nn.MSELoss()
    # # optimizer = optim.SGD(lstm_model_init.parameters(), lr)
    # optimizer = optim.Adam(lstm_model_init.parameters(), lr)
    # # optimizer = optim.LBFGS(lstm_model.parameters(), lr) # for it to work u have to craete a closure function. See pytorch documentation fo more info

    # # # try forward method with a (εχεις φτιάξει ένα LSTM που παίρνει ένα τενσορα fc_num στοιχείων και επιστρέφει ένα τενσορα 1 στοιχείου
    # # a=np.linspace(0,3,input_size); a=torch.tensor(a, dtype=torch.float32); a=torch.unsqueeze(a,0); a=torch.unsqueeze(a,0);print(a.shape)
    # # arr = lstm_model(a) # input must be dims (batch_size, sequence_length, input_size)
    # # print('arr output shape is', arr.shape); print(arr)

    if train_LSTM:
        lstm_model = LSTM_train(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, epochs, sliding_window_step, scaling_method, tag, input_size, output_size, save_load_model_number, extract_data)
    #     if run_to_gpu_all or run_to_gpu_batch : lstm_model_init = lstm_model_init.to(device)
    #     lstm_model, training_string = training_lstm(lstm_model_init, criterion, optimizer, epochs, train_loader, val_loader, scaling_method, save_load_model_number, measure_train_time=True) 
    #     create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string)
    #     save_params(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string)
    # del lstm_model_init # delete this variable in order to save memory
    if not(train_LSTM):
        dict_train = load_params(save_load_model_number)
        input_size, hidden_state_dim, num_layers, output_size = dict_train['input_size'], dict_train['hidden_state_dim'], dict_train['num_layers'], dict_train['output_size'] # χρειάζονται για το loading του LSTM
        downsample_scale, scaling_method = dict_train['downsample_scale'], dict_train['scaling_method'] # χρειάζονται για το generate/compare
        # scaler.change_scaling_method(scaling_method) # αυτό ήταν πριν αρχίσεις να χρησιμοποιήσει 2 scalers για να μη προκαλείται σφάλμα tensor*ndarray
        lstm_model = load_lstm(save_load_model_number, input_size, hidden_state_dim, num_layers, output_size)

    # if train_LSTM:
    #     ###  test trained LSTM
    #     print('\nTest trained LSTM: Compare actual and generated signal')
    #     generate_or_compare = 'compare' # 'generate' , 'compare'
    #     starting_point = 4000 # 4000 350000 700000
    #     if starting_point < lstm_model.input_size: starting_point = lstm_model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
    #     num_gen_points = output_size # 5 * output_size
    #     test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
    #     if remote_PC: test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
    #     print('length of test series is ', test_series.shape)
    #     fs = 1/(test_series[1,3] - test_series[1,2])
    #     test_signal = torch.from_numpy(test_series[0,:]).clone().float()
    #     if generate_or_compare == 'generate': gen_signal = lstm_generate_lfp(lstm_model, test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=1); plt.plot(gen_signal); plt.show(); plt.close()
    #     if generate_or_compare == 'compare':
    #         base_signal, gen_signal, = produce_comparing_signals(lstm_model, test_signal, starting_point, num_gen_points, scaler)
    #         compare_for_trained_lstm(base_signal, gen_signal, starting_point, fs)
    #         # compare_noise(base_signal, fs)
    #     del test_series; del test_signal; del base_signal; del gen_signal # delete these varibles in order to save memory

    num_gen_points = 3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
    test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
    if remote_PC: test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
    check_lstm_metrics(lstm_model, test_series, num_gen_points, scaling_method)


    # train linear and dummy regressors with the same data
    linear_save_name = 'lineeear'
    dummy_save_name = 'duuuumy' #'None'
    if train_older:
        linear = train_older_methods('linear', tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step, batch_size, model_save_name=linear_save_name)
        dummy = train_older_methods('dummy', tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step, batch_size, model_save_name=dummy_save_name)
    if not(train_older):
        with open(PATH + 'project_files/' + linear_save_name + '.pkl', 'rb') as file: linear = pickle.load(file)
        with open(PATH + 'project_files/' + dummy_save_name + '.pkl', 'rb') as file: dummy = pickle.load(file)
    
    

    # load test series
    # μπορείς να προσθέσεις περισσότερες από μια testing χρονοσειρές, για ακόμα μεγαλύτερη γενικευσιμότητα των αποτελεσμάτων
    test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
    if remote_PC: test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
    print('length of test series is ', test_series.shape)

    # compare statisticaly different methods
    number_of_st_points = 40 # καθορίζει πόσα τυχαία σημεία έναρξης της πρόβλεψης θα παρθούν για τη στατιστική σύγκριση των μεθόδων
    num_gen_points = 3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
    print('\n\n\nCOMPARISON OF LSTM FORECASTING WITH AUTOREGRESSIC FORECASTING')
    lstm_MAE_list, linear_MAE_list = produce_metric_samples(test_series, lstm_model, linear, number_of_st_points, num_gen_points, scaling_method, metric ='MAE')
    statistical_comparison(lstm_MAE_list, linear_MAE_list, normality_test ='SW')
    print('\n\n\nCOMPARISON OF LSTM FORECASTING WITH DUMMY REGRESSOR')
    lstm_MAE_list, dummy_MAE_list = produce_metric_samples(test_series, lstm_model, dummy, number_of_st_points, num_gen_points, scaling_method, metric ='max Cross-cor')
    statistical_comparison(lstm_MAE_list, dummy_MAE_list, normality_test ='SW')
    print('\n\n\nCOMPARISON OF LSTM FORECASTING WITH PINK NOISE')
    lstm_MAE_list, noise_MAE_list = produce_metric_samples(test_series, lstm_model, 'pink_noise', number_of_st_points, num_gen_points, scaling_method, metric ='RMS-PSD')
    statistical_comparison(lstm_MAE_list, noise_MAE_list, normality_test ='SW')

    # visual presentations of forecasting
    starting_point = np.random.randint(lstm_model.input_size, test_series.shape[1])# size = number_of_st_points)
    num_gen_points = output_size # 5 * output_size
    test_signal = test_series[0,:]
    tensor_test_signal = torch.from_numpy(test_signal).clone().float()
    fs = 1/(test_series[1,3] - test_series[1,2])
    actual_signal = test_signal [starting_point : starting_point + num_gen_points]
    lstm_gen_signal = lstm_generate_lfp(lstm_model, tensor_test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=1)
    visual_fc_comparison(lstm_gen_signal, actual_signal, fs, domain = 'both')
    linear_gen_signal = ml_generate_lfp(linear, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
    visual_fc_comparison(linear_gen_signal, actual_signal, fs, domain ='both')
    dummy_gen_signal = ml_generate_lfp(dummy, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
    visual_fc_comparison(dummy_gen_signal, actual_signal, fs, domain ='cross-correlation')



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def LSTM_train(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, epochs, sliding_window_step, scaling_method, tag, input_size, output_size, save_load_model_number, extract_data = False):
    print('\nTrain LSTM-RNN:')

    # creates a folder to save the new trained model
    created_folder = False
    if save_load_model_number == 0: created_folder = True # ο φάκελος 0 θα είναι ο προτυπος φάκελος για να μη φτιάχνοντα ασκοπα καινούργιοι
    while created_folder == False:
        newpath = PATH + '/project_files/models/model' + str(save_load_model_number) + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            created_folder = True
        else:
            save_load_model_number = save_load_model_number + 1

    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    # Import data
    if not(remote_PC): save_load_path = PATH + 'project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path) 
    if not(extract_data): lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    # prepare data
    train_loader, val_loader, scaler  = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size)
    # if (train_LSTM or train_older): train_loader, val_loader, x_train, y_train, x_test, y_test, tensor_scaler = prepare_data_mem(lfp_data, tensor_scaler, input_size, output_size, sliding_window_step, batch_size)

    ## NN instance creation
    lstm_model_init = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(lstm_model_init.parameters(), lr)
    optimizer = optim.Adam(lstm_model_init.parameters(), lr)
    # optimizer = optim.LBFGS(lstm_model.parameters(), lr) # for it to work u have to craete a closure function. See pytorch documentation fo more info

    # # try forward method with a (εχεις φτιάξει ένα LSTM που παίρνει ένα τενσορα fc_num στοιχείων και επιστρέφει ένα τενσορα 1 στοιχείου
    # a=np.linspace(0,3,input_size); a=torch.tensor(a, dtype=torch.float32); a=torch.unsqueeze(a,0); a=torch.unsqueeze(a,0);print(a.shape)
    # arr = lstm_model(a) # input must be dims (batch_size, sequence_length, input_size)
    # print('arr output shape is', arr.shape); print(arr)

    # train lstm and save it
    if run_to_gpu_all or run_to_gpu_batch : lstm_model_init = lstm_model_init.to(device)
    lstm_model, training_string = training_lstm_loop(lstm_model_init, criterion, optimizer, epochs, train_loader, val_loader, scaling_method, save_load_model_number, measure_train_time=True) 
    create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string, save_load_model_number)
    save_params(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string, save_load_model_number)

    return lstm_model




### data preparation
def prepare_data(lfp_data_matrix, scaler, input_size, output_size, window_step, batch_size, cut_with_numpy=False, return_loaders=True):
    """This function prepares the data (i.e. normalizes, divide long signals, dreates windowed data, wraps them into loaders) and returs the train_loader and val_loader 
    objects that will be used to feed the batces in the LSTM during training process"""

    # scaling_data
    if scalling_manner == 'norm_all_data':
        lfp_data_matrix = scaler.normalize2d(lfp_data_matrix) # κανονικοποιεί το σήμα
        scaler.fit2d(lfp_data_matrix) # εξάγει κοινές παραμέτρους κανονικοποίησης για όλο το σήμα

    # Δημιουργία παραπάνω batches
    if lfp_data_matrix.shape[1]>10**6: # κόβει τα σήματα και φτιάχνει νέα batches για να επιλύσει πορβλήματα μνήμης. Μπορείς να το κάνεις και συνάρτηση
        batch_multiplier = 10 # θα κόψει κάθε σήμα τόσες φορές και θα δημιοργήσει τόσα νέα batches για κάθε σήμα
        new_cutted_length = lfp_data_matrix.shape[1] - (lfp_data_matrix.shape[1] % batch_multiplier) 
        lfp_data_matrix = lfp_data_matrix[:, 0:new_cutted_length] # κόβω τα τελευταία στοιχεία για να είναι διαιρέσιμο με το 10 (ή γεντικότερα με το batch_multiplier)
        lfp_data_split = np.hsplit(lfp_data_matrix, batch_multiplier)
        lfp_data_matrix= np.vstack(lfp_data_split)
        print('After batch multiplication lfp_data have shape: ', lfp_data_matrix.shape)

    window_size = input_size + output_size

    # παρακάτω οι variables data, windowed_data, input_data, target_data, dataset, train_data, val_data, train_loader, val_loader είναι views και έτσι μαλλον δεν
    # καταλαμβάνουν επιπρόσθετη μνήμη

    if cut_with_numpy: # το cut with numpy είναι τεχνικά άχρηστο αλλά το κρατάς για ιστορικούς λόγους σαν άλλη μέθοδο δημιουργίας παραθύρων
        # Δημιουργία δεδομένων εκπαίδευσης με κόψιμο τους σε παράθυρα όπου κάθε παράθυρο περιλαμβάνει τα target_data και input_data μιας forecasting δοκιμής/εκπαίδευσης
        windowed_data = np.lib.stride_tricks.sliding_window_view(lfp_data_matrix, window_size, axis=1, writeable=True)[:,::window_step,:]
        if fc_move_by_one: windowed_data = np.lib.stride_tricks.sliding_window_view(lfp_data_matrix, input_size + 1, axis=1, writeable=True)[:,::window_step,:]
        print('numpy windowed data are', windowed_data.shape)
        # windowed_data = torch.from_numpy(windowed_data).float() # με το που εκτελείς αυτή την εντολή, τα windows παύουν να είναι views του numpy και αυτό αυξάνει σημαντικά τις ανάγκες σε μνήμη. Αυτό είναι το πιο συχνό σημείο για Runtime errors
    if not(cut_with_numpy): # εδώ τα παράθυρα κόβονται αφού είναι tensors, οπότε παραμένουν views. Βέβαια μάλλον παύουν να είναι views όταν εισάγονται στους loaders
        data = torch.from_numpy(lfp_data_matrix).float()
        windowed_data = data.unfold(dimension=1, size = window_size, step = window_step) 
        print('torch windowed data are', windowed_data.shape)
    input_data = windowed_data[:,:, 0:input_size]
    target_data = windowed_data[:,:,input_size:window_size]

    if not(return_loaders): return input_data, target_data, scaler
    if fc_move_by_one: target_data = windowed_data[:,:,1:output_size+1] # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση
    if run_to_gpu_all: input_data=input_data.to(device); target_data=target_data.to(device) # εδώ τα data σίγουρα παύουν να είναι views. Για αυτό πιο κάτω στο training μόνο σε αυτή την περίπτωση τα bathces δεν αντιγράφονται
    dataset = torch.utils.data.TensorDataset(input_data, target_data); del input_data; del target_data
    train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.9, 0.1]); del dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=train_data.__len__()
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=val_data.__len__()
    if return_loaders: return train_loader, val_loader, scaler




### Architecture (class) of the LSTM-based-neural-network
class LSTM_fc(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depends from the output size and the target data """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_fc, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.output_size = output_size

        self.norm_layer = nn.LayerNorm(input_size)
        self.lstm=nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True) # nn.LSTM has dynamic layers throught the num_layer parameter which creates stacked LSTM
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # batch_size = x.size(0)
        if scalling_manner == 'input_layer_norm_and_output_batch_norm': x=self.norm_layer(x)
        out, _ = self.lstm(x) # out dims (batch_size, L, hidden_size) if batch_first=True
        out = self.linear(out)
        return out
    



### train & validate the LSTM model
def training_lstm_loop(model, criterion, optimizer, epochs, train_loader, val_loader, scaling_method, model_number, measure_train_time:bool):
    """This is the NN training function. It recieves the typical parameters fo model, loss function (criterion), optimizer and epochs. 
    It also recieves as input the torch dataloader objects for training ana validation data. 
    save_name must be a string. If it's none then the trained model is not saved, otherwise the model is saved with file name the string.
    measure_train_time must be True or False. If it's True then the training time is calculated and printed"""

    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    print('start training')
    if measure_train_time: tic = time.perf_counter()
    
    num_epochs = epochs

    losses_list=[]
    val_losses_list=[]
    val_mean_loss = 10^6 # initialization for the loop
    training_string =''
    time_str='' # initialization if measure_time = False

    for epoch in range(num_epochs):
        if measure_train_time: t0 = time.perf_counter()
        ### training
        model.train()
        train_losses = []
        if measure_train_time: norm_time = []
        if measure_train_time: train_time = []
        for x_batch, y_batch in train_loader:
            # x_batch, y_batch = x_batch.detach().clone(), y_batch.detach().clone() # να τα batces είναι viewws των data θα πρέπει να γίνουν ανεξάρτητα για να μη δημιουργηθούν σφάλματα
            if not(run_to_gpu_all): x_batch, y_batch = x_batch.detach().clone().numpy(), y_batch.detach().clone().numpy() # να τα batches είναι viewws των data θα πρέπει να γίνουν ανεξάρτητα για να μη δημιουργηθούν σφάλματα
            if scalling_manner in ['norm_batches', 'input_layer_norm_and_output_batch_norm']:
                if measure_train_time: t1 = time.perf_counter()
                with torch.no_grad():
                    #print('Batches normalization begins')
                    if scalling_manner != 'input_layer_norm_and_output_batch_norm': x_batch = scaler.normalize3d(x_batch)
                    y_batch = scaler.normalize3d(y_batch)
                    #print('Batches normalized')
                if measure_train_time: t2 = time.perf_counter()
                if measure_train_time: norm_time.append(t2-t1)
            if not(run_to_gpu_all): x_batch = torch.from_numpy(x_batch).requires_grad_()
            if not(run_to_gpu_all): y_batch = torch.from_numpy(y_batch).requires_grad_()
            if run_to_gpu_batch: x_batch = x_batch.to(device); y_batch = y_batch.to(device) 
            train_pred = model(x_batch)
            train_pred = torch.squeeze(train_pred)
            y_batch = torch.squeeze(y_batch)
            # print(y_batch - train_pred) # αν η διαφορά είναι πολύ μικρή ίσως τα δεδομένα χρειάζονται κανονικοποίηση
            loss = criterion (y_batch, train_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item()) # list of train_losses for every batch
        mean_loss = sum(train_losses)/len(train_losses) # mean of train_losses of all batches in every epoch
        if measure_train_time: t3 = time.perf_counter()
        if measure_train_time: train_time.append(t3-t0)
        del x_batch; del y_batch; del train_pred # these variables are deleted in order to save memory

        ### validation
        val_mean_loss_old = val_mean_loss #; print(val_mean_loss_old)
        model.eval()
        with torch.no_grad():
            val_losses =[]
            for x_val, y_val in val_loader:
                if run_to_gpu_batch: x_val = x_val.to(device); y_val = y_val.to(device)   
                test_pred = model(x_val)
                test_pred = torch.squeeze(test_pred)
                y_val = torch.squeeze(y_val)
                val_loss = criterion (y_val, test_pred)
                val_losses.append(val_loss.item()) # list of val_losses for every batch
            val_mean_loss = sum(val_losses)/len(val_losses) # mean of val_losses of all batches in every epoch
            if measure_train_time: train_time = np.array(train_time).mean(); norm_time = np.array(norm_time).mean(); whole_time = train_time + norm_time
        # print(f'Epoch:{epoch+1}/{num_epochs} -> train (batch mean) loss = {mean_loss} - val (batch mean) loss = {val_mean_loss}')
        epoch_str = f'Epoch:{epoch+1}/{num_epochs} -> train (batch mean) loss = {mean_loss} - val (batch mean) loss = {val_mean_loss}'; print(epoch_str)
        if measure_train_time: time_str = f'Computation times -> epoch_time: {whole_time} - train_time: {train_time} - norm_time: {norm_time}'; print(time_str)
        training_string = training_string + '\n' + epoch_str + '\n' + time_str
        losses_list.append(mean_loss)
        val_losses_list.append(val_mean_loss)

        ## save the model of the best epoch (with the smallest val_mean_loss) -> δεν κάνει μεγάλη διαφορά με το να αποθήκευεται στο τέλος επειδη το loss σχεδόν πάντα μειώνεται
        # check = val_mean_loss < val_mean_loss_old
        # print(val_mean_loss_old, val_mean_loss, check)
        if  val_mean_loss < val_mean_loss_old: torch.save(model.state_dict(), PATH + 'project_files/models/model' + str(model_number) + '/LSTM_forecasting_model.pt')
        
    if measure_train_time: toc = time.perf_counter()
    if measure_train_time: print ('whole training time is', toc - tic)

    ### plot train and validation losses
    plt.plot(range(num_epochs), losses_list, label = 'Train loss')
    plt.plot(range(num_epochs), val_losses_list, label = 'Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('LSTM training - Loss to Epochs diagram')
    plt.legend()
    plt.show()
    plt.close()

    return model, training_string



### creates and saves report of the training of the LSTM model to a text file
def create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, window_step, scaling_method, tag, input_size, output_size, training_string, model_number):
    ds_string = f'Downsampling: {downsample_scale}'
    hiiden_size_string = f'Size of LSTM hidden state: {hidden_state_dim}'
    layers_string = f'Number of stacked LSTM layers: {num_layers}'
    batch_string = f'Size of batches: {batch_size}'
    lr_string = f'Learning rate: {lr}'
    window_string = f'sliding window step: {window_step}'
    scaling_string = f'normalization method: {scaling_method}'

    files_string = f'files used: {tag}'
    input_string = f'input size: {input_size}'
    output_string = f'output size: {output_size}'
    
    fc_move_by_one_string = f'fc_move_by_one: {fc_move_by_one}'
    scaling_manner_string = f'scaling_manner: {scalling_manner}'
    # if device == 'cpu': training_method_string = f'training_method: {device}'
    if run_to_gpu_all == 1: training_method_string = 'training_method: All data passed to GPU in the beggining'
    elif run_to_gpu_batch == 1: training_method_string = 'training_method: Batches are passed to GPU seperately'
    else: training_method_string = f"training_method: 'cpu'"

    whole_string = (ds_string + '\n'+hiiden_size_string + '\n'+layers_string + '\n'+batch_string + '\n'+lr_string + '\n'+window_string + '\n'+scaling_string + 
                    '\n\n'+files_string + '\n'+input_string + '\n'+output_string + '\n\n'+fc_move_by_one_string + '\n'+scaling_manner_string + '\n'+training_method_string + 
                    '\n\n'+training_string)
    with open(PATH + '/project_files/models/model' + str(model_number) + '/training_log.txt', "w+") as file: file.write(whole_string)




# load the LSTM model if you have saved it, in order not to run training again if its time-consuming
def load_lstm(model_number, input_size, hidden_state_dim, num_layers, output_size):
    model = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size) 
    model.load_state_dict(torch.load(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_forecasting_model.pt'))
    print('LSTM model has been loaded')
    return model



### saves the parameters of the LSTM model to a dictionary and then saves the dictionary to a picle file
def save_params(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, window_step, scaling_method, tag, input_size, output_size, training_string, model_number):
    dict_param = {'downsample_scale':downsample_scale, 'hidden_state_dim':hidden_state_dim, 'num_layers':num_layers, 'batch_size':batch_size, 'lr':lr, 
                  'window_step':window_step, 'scaling_method':scaling_method, 'tag':tag, 'input_size':input_size, 'output_size':output_size,  'training_string':training_string}
    with open(PATH + '/project_files/models/model' + str(model_number) + '/LSTM_params.pkl', "wb") as file: pickle.dump(dict_param, file)


### loads the saved parameters of the LSTM model from the picled dictionary file
def load_params(model_number):
    with open(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_params.pkl', 'rb') as file: dict_param = pickle.load(file)
    return dict_param




### Generate/forecast LFP signal with the LSTM model
def lstm_generate_lfp(model, signal, num_gen_points:int, scaling_method, only_gen_signal:bool):
    """""This function ganerates a number of points (num_gen_points), in the end of the given LFP signal (signal) by using a trained NN (model)
        1) model -> is the LSTM forecasting model
        2) signal -> must be an lfp signal in tensor form bigger in length than the input_size of the LSTM model
        3) num_gen_points -> is the number of the points that will be generated/forecasted in total
        4) scaler -> is the scaler object used for scaling/unscaling the data. This class exist in the signal_handler module
    """""

    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    model = model.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
    if torch.is_tensor(signal):
        signal = signal.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
        signal = signal.cpu().numpy()
    else: signal = np.float32(signal)
    if scalling_manner in ['norm_all_data', 'norm_batches']: signal = scaler.fit_transform1d(signal) # κάνονικοποιεί τo σήμα-input με τον ίδιο τρόπο που έχει μάθει να δέχεται κανονικοποιημένα inputs το LSTM
    if not(only_gen_signal): generated_signal= list(signal) # αν θέλουμε το παραγώμενο σήμα να περιέχει το input
    if only_gen_signal: generated_signal=[] # αν θέλουμε το παραγώμενο σήμα να περιέχει μονο το generated χωρίς το input
    fc_repeats = int(num_gen_points/model.output_size) + 1 # παράγει μερικά παραπάνω σημεία και κόβει τα τελευταία για να μπορεί να παράγει σημεία που δεν είναι πολλπλάσια του output_size
    if fc_move_by_one: fc_repeats = num_gen_points
    
    model.eval()
    signal =  torch.from_numpy(signal)
    starting_signal = signal[(len(signal)-model.input_size):] # παiρνει τα τελευταία σημεία του σήματος (τόσα όσο είναι το input του model) για να παράξει τη συνέχεια του σήματος
    for i in range(fc_repeats): # παράγει το output, το κάνει λίστα αριθμών και επεικείνει με αυτό, τη generated_signal
        starting_signal_input=torch.unsqueeze(starting_signal, 0)
        starting_signal_input=torch.unsqueeze(starting_signal_input, 0)
        output = model(starting_signal_input)
        output = torch.squeeze(output)
        if fc_move_by_one: output = output[-1]
        if output.shape == torch.Size([]): output = torch.unsqueeze(output, 0) # αυτό χρειάζεται όταν το output εχει διάσταση 1
        generated_signal = generated_signal + list(output.detach().numpy()) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης
        if not(fc_move_by_one): starting_signal = torch.cat((starting_signal, output), dim=0)[model.output_size:] # κατασκευή νέου input για το model
        if fc_move_by_one: starting_signal = torch.cat((starting_signal, output), dim=0)[1:] # κατασκευή νέου input για το model
    generated_signal = np.array(generated_signal) # η λίστα generated_signal μετατρέπεται σε np.ndarray
    if not(only_gen_signal): generated_signal = generated_signal[: signal.shape[0] + num_gen_points] # κρατιοούνται μόνο τα σημεία που ζητήθηκαν να παραχθούν (είχαν παραχθεί λίγα περισσότερα, που είνια πολλαπλάσια του LSTM output)
    if only_gen_signal: generated_signal = generated_signal[:num_gen_points] # αν θέλουμε το παραγώμενο σήμα να περιέχει μονο το generated χωρίς το input χρειάζεται κι αυτή η εντολή
    if scalling_manner != 'None': generated_signal = scaler.inverse1d(generated_signal) # αποκανονικοποίηση του τελικού αποτελέσματος για να είναι στην κλίμακα του LFP
    return generated_signal



# ## αυτή η συνάρτηση είναι πιο παλιά και πλέον χρησιμοποιείται μόνο για τα αποτελεσματα της εκπαίδευσης του LSTM. Θα μπορούσε να αντικατασταθεί με τις νεότερες ...
# # ...συναρτήσεις, αλλά αυτό θα ήθελε λίγη επιπρόσθετη ενασχόληση
# ### παίρνει ένα σήμα και επιστρέφει δύο σήματα ίδιου μήκους -> 1) το αρχικό σήμα κομμένο, 2) το generated σήμα -> ώστε να μπορούν να συγκριθούν για το foracasting τους
# def produce_comparing_signals(model, signal:torch.Tensor, starting_point:int, num_gen_points:int, scaler):
#     """"This functions recieves the necessary parameters and produces with the LSTM model, the signals which will be compared in the compare function
#     Starting point must be bigger than model.iput_size because we can't start producing point i there isn't enough input points before the starting point"""
#     if starting_point < model.input_size: starting_point = model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
#     gen_signal = lstm_generate_lfp(model, signal[:starting_point], num_gen_points, scaler, only_gen_signal = False) # παράγει το forecasted σήμα ξεκινόντας από το starting point
#     signal_baseline = signal[: len(gen_signal)].numpy() # παίρνει κομμάτι από το αρχικό σήμα, που να έχει ίδιο μήκος με το gen_signal -> έτσι μπορούν να συγκριθούν
#     return signal_baseline, gen_signal



# ## αυτή η συνάρτηση είναι πιο παλιά και πλέον χρησιμοποιείται μόνο για τα αποτελεσματα της εκπαίδευσης του LSTM. Θα μπορούσε να αντικατασταθεί με τις νεότερες ...
# # ...συναρτήσεις, αλλά αυτό θα ήθελε λίγη επιπρόσθετη ενασχόληση
# ## compare the similarity of 2 signals. Used for comparing the LSTM-forecasted signal, with the actual signal. It calculates different metrics and also produces some related plots.
# def compare_for_trained_lstm(baseline_signal:np.ndarray, generated_signal:np.ndarray, starting_point:int, fs):
#     """This function takes as input the forecasted signal and compares it with the actual follwoing signal in order to measure the similarity of the two signals. The 
#     greater the similarity of the two signals, the most accurate will be the forecasting (ideally the forecasted signal would be completely the same as the actual (baseleine)
#     signal. The variable 'generated signal' is the forecasted signal that has been generated by the ML model. The "baseline_signal" is the actual signal that followed and
#     'fs' is the sampling_frequency of the baseline_signal)
#     Stating point is the point in which forecasting starts. It is importana because before the starting point signals are the same and souldn't be compared because they
#     falsely give better comparing results. However the previus parts are ploted, because they give a better optical impression of the forecasting, and that is the reason
#     they weren't ommited completely from this function"""

#     pure_gen_signal = generated_signal[starting_point:] # με αυτή και την επόμενη γραμμή παίρνουμε τα σήματα χώρίς το input signal που χρησιμοποιηθηκε για την παραγωγη του
#     pure_base_signal = baseline_signal[starting_point:] # σήματος και επειδή είναι το ίδιο και στα δύο, αλλοιώνει τις μετρικές, βελτιώνοντας τις
    
#     MAE = mean_absolute_error(pure_base_signal, pure_gen_signal)
#     print ('Mean absolute error is', MAE)
#     RMSE = np.sqrt(mean_squared_error(pure_base_signal, pure_gen_signal))
#     print ('Root mean square error is', RMSE)
#     np_correlate = np.correlate(pure_base_signal/pure_base_signal.std(), pure_gen_signal/pure_gen_signal.std())/len(pure_base_signal) # η μορφή αυτή είναι κανονικοποιήμένη
#     print('Cross-correlation (np.correlate) is', np_correlate) # η cross-correlation ΕΙΝΑΙ κανονικοποιήμένη στο [-1,1]
#     np_corrcoef = np.corrcoef(pure_base_signal, pure_gen_signal)[0,1]
#     print('Cross-correlation (np.corrcoef) is', np_corrcoef) # takes values in [-1,1]. The closer to 1 the greter the similarity. Is a Pearson corealation coeficient so is almost the same as st.pearsonr
#     # sn_corr = sn.correlate(signal_baseline/signal_baseline.std(), gen_signal/gen_signal.std())/len(signal_baseline) # παραλλαγή του από κάτω
#     sn_corr = sn.correlate(pure_base_signal, pure_gen_signal)/(pure_base_signal.std()*pure_gen_signal.std()*len(pure_base_signal))
#     plt.plot(sn_corr); plt.title('sn.correlate cross correlation'); plt.show();  plt.close() # η corss-correlation ΕΙΝΑΙ κανονικοποιήμένη στο [-1,1]
#     pearson_r = stats.pearsonr(pure_base_signal, pure_gen_signal)
#     print('Pearson (Cross-)correlation (st.pearsonr) is', pearson_r) # takes values in [-1,1]. The closer to 1 the greter the similarity, and the closer is to 0 the less the similarity.
#     spearman_rho = stats.spearmanr(pure_base_signal, pure_gen_signal)
#     print('Spearman rho correlation (st.spearmanr)) is', spearman_rho)


#     # οπτίκή σύγκριση των δύο σημάτων περιλαμβάνοντας τα προηγούμενα τμήματα που παρήγαγαν το σήμα
#     plt. plot(baseline_signal, label = 'original signal')
#     plt. plot(generated_signal, label = 'generated signal')
#     plt.legend()
#     plt.title('Visual comparison with precedent signal')
#     plt.show()
#     plt.close()

#     # οπτίκή σύγκριση μόνο του generated segment με το actual segment
#     plt. plot(pure_base_signal, label = 'original signal')
#     plt. plot(pure_gen_signal, label = 'generated signal')
#     plt.legend()
#     plt.title('Visual comparison of generated and actual segments')
#     plt.show()
#     plt.close()

#     # για σύγκριση σήματων με διαφορετικό μέγεθός ή διαφορετικο downsampling θα μπορούσε να χρησιμοποιηθεί το dynamic time warping

#     # compare signal frequencies (to see if generated signal have approximately the same spectral components)
#     f1, Pxx_1 = sn.periodogram(pure_base_signal, fs=fs, return_onesided=True, scaling='density')
#     f2, Pxx_2 = sn.periodogram(pure_gen_signal, fs=fs, return_onesided=True, scaling='density')
#     plt.plot(f1,Pxx_1, label = 'original signal')
#     plt.plot(f2,Pxx_2, label = 'generated signal')
#     plt.suptitle('compare signals in frequency-domain (Fourier)')
#     plt.title(f'sampling frequency is {fs} due to downsampling', fontsize = 9)
#     plt.legend()
#     plt.show()
#     plt.close()


###  test trained LSTM metrics and visualizations
def check_lstm_metrics(lstm_model, test_series, num_gen_points, scaling_method):
    print('\nTest trained LSTM: Compare actual and generated signal')

    input_size = lstm_model.input_size
    output_size = lstm_model.output_size
    number_of_starting_points = 1 # καθορίζει πόσα τυχαία σημεία έναρξης της πρόβλεψης θα παρθούν για την παραγωγή των λιστών της κάθε μετρικής

    starting_points_list = np.random.randint(input_size, test_series.shape[1] - num_gen_points, size = number_of_starting_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν 
    # από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    
    MAE_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE', make_barplot=False)
    RMSE_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMSE', make_barplot=False)
    pearson_r_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='Pearson r', make_barplot=False)
    max_cross_cor_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='max Cross-cor', make_barplot=False)
    RMS_PSD_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMS-PSD', make_barplot=False)

    print ('Absolute mean error', sum(MAE_list)/len(MAE_list))
    print ('Root mean square error is', sum(RMSE_list)/len(RMSE_list))
    print ('Pearson r (normalized cross-correlation of zero phase) is', sum(pearson_r_list)/len(pearson_r_list))
    print ('Maximum cross-correlation is', sum(max_cross_cor_list)/len(max_cross_cor_list))
    print ('Root mean square error of PSD is', sum(RMS_PSD_list)/len(RMS_PSD_list))
    print('\n\n')
    # βαλε mean και ένα σημείο
    # βάλε και visuals

    # visual representation on a random point of the signal
    starting_point = starting_points_list[0]
    test_signal = test_series[0,:]
    tensor_test_signal = torch.from_numpy(test_signal).clone().float()
    fs = 1/(test_series[1,3] - test_series[1,2])
    actual_signal = test_signal [starting_point : starting_point + num_gen_points]
    lstm_gen_signal = lstm_generate_lfp(lstm_model, tensor_test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=1)
    visual_fc_comparison(lstm_gen_signal, actual_signal, fs, domain = 'both')
    visual_fc_comparison(lstm_gen_signal, actual_signal, fs, domain = 'cross-correlation')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### εκπαιδεύει στο forecasting με άλλες μεθόδους πέραν του LSTM. Μπορείς να προσθέσεις και άλλες μεθόδους αν θέλεις.
# def train_older_methods(lfp_data, scaling_method, input_size, output_size, sliding_window_step, batch_size, linear_save_name:str, dummy_save_name:str):
    
#     scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

#     # αν θέλεις τα δεδομένα να είναι ακριβως τα ίδια με αυτά του LSTM, χρησιμοποίησε αυτές εδώ
#     # x_train = recreate_data(train_loader, 'input')
#     # y_train = recreate_data(train_loader, 'input')
#     # x_test = recreate_data(val_loader, 'input')
#     # y_test = recreate_data(val_loader, 'input')
    
#     # scaling_data -> if scaling_manner = norm_all_data, then data are normazized inside the prepare_data function, else they are normalized here
#     if scalling_manner != 'norm_all_data':
#         lfp_data = scaler.normalize2d(lfp_data) # κανονικοποιεί το σήμα
#         scaler.fit2d(lfp_data) # εξάγει κοινές παραμέτρους κανονικοποίησης για όλο το σήμα

#     x_data, y_data, scaler = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size, cut_with_numpy=1, return_loaders=0)
#     # x_data, y_data = x_data.numpy(), y_data.numpy()
#     x_data=np.reshape(x_data, (x_data.shape[0]*x_data.shape[1], x_data.shape[2]))
#     y_data=np.reshape(y_data, (y_data.shape[0]*y_data.shape[1], y_data.shape[2]))
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.9)


#     linear = lrm.LinearRegression()
#     linear.fit(x_train, y_train)
#     print('Linear regression R^2 score is ', linear.score(x_test, y_test))
#     # pred = linear.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting

#     dummy = DummyRegressor(strategy='mean')
#     dummy.fit(x_train, y_train)
#     print('Dummy regresson R^2 score is ', dummy.score(x_test, y_test))
#     # pred = dummy.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting

#     if linear_save_name != 'None': 
#         with open(PATH + 'project_files/' + linear_save_name + ".pkl", "wb") as file1: pickle.dump(linear, file1)
#     if dummy_save_name != 'None': 
#         with open(PATH + 'project_files/' + dummy_save_name + ".pkl", "wb") as file2: pickle.dump(dummy, file2)
        
#     return linear, dummy



### εκπαιδεύει στο forecasting με άλλες μεθόδους πέραν του LSTM. Μπορείς να προσθέσεις και άλλες μεθόδους αν θέλεις.
def train_older_methods(ml_method, tag, downsample_scale, scaling_method, input_size, output_size, sliding_window_step, batch_size, model_save_name:str):
    print('\nTrain ' + ml_method + ' regressor:')
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    save_load_path= PATH + 'project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών
    x_data, y_data, scaler  = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size, cut_with_numpy=True, return_loaders=False)
    
    # # αν θέλεις τα δεδομένα να είναι ακριβως τα ίδια με αυτά του LSTM (αλλίως είναι διαφορετικά λόγω διαφορετικού train-test split), χρησιμοποίησε αυτές εδώ
    # def recreate_data(loader, extracted_data):
    #     if extracted_data == 'input': s=0
    #     if extracted_data == 'target': s=1
    #     data = loader.dataset[0][s].numpy()
    #     for idx, batch in enumerate(loader):
    #         if idx > 0:
    #             data = np.vstack((data, torch.squeeze(batch[s]).numpy()))
    #     return data
    # train_loader, val_loader, scaler  = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size)
    # x_train = recreate_data(train_loader, 'input')
    # y_train = recreate_data(train_loader, 'target')
    # x_test = recreate_data(val_loader, 'input')
    # y_test = recreate_data(val_loader, 'target')
    
    # scaling_data -> if scaling_manner = norm_all_data, then data are normazized inside the prepare_data function, else they are normalized here
    # if scalling_manner != 'norm_all_data':
    #     lfp_data = scaler.normalize2d(lfp_data) # κανονικοποιεί το σήμα
    #     scaler.fit2d(lfp_data) # εξάγει κοινές παραμέτρους κανονικοποίησης για όλο το σήμα
    # υπάρχει ο 'κίνδυνος' να ξανακανονικοποιηθούν τα δεδομένα αν έχει γίνει scaling_manner = norm_all_data, αλλά αυτό μάλλον δεν αποτελεί μεγάλο πρόβλημα επειδή δε θα αλλάξουν ιδιαίτερα μορφή
    x_data = lfp_data = scaler.normalize2d(x_data) 
    y_data = lfp_data = scaler.normalize2d(y_data)

    # x_data, y_data, scaler = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size, cut_with_numpy=1, return_loaders=0)
    # x_data, y_data = x_data.numpy(), y_data.numpy()
    x_data=np.reshape(x_data, (x_data.shape[0]*x_data.shape[1], x_data.shape[2])) # transforms the data in the sklearn format
    y_data=np.reshape(y_data, (y_data.shape[0]*y_data.shape[1], y_data.shape[2])) # transforms the data in the sklearn format
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.9)

    if ml_method == 'linear':
        model = lrm.LinearRegression()
        name_string = 'Linear regression'
    elif ml_method == 'dummy':
        model = DummyRegressor(strategy='mean')
        name_string = 'Dummy regressor'

    model.fit(x_train, y_train)
    print(name_string + ' R^2 score is ', model.score(x_test, y_test))
    # pred = model.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting

    if model_save_name != 'None': 
        with open(PATH + 'project_files/' + model_save_name + ".pkl", "wb") as file: pickle.dump(model, file)
        
    return model



def recreate_data(loader, extracted_data):
    if extracted_data == 'input': s=0
    if extracted_data == 'target': s=1
    data = loader.dataset[0][s].numpy()
    for idx, batch in enumerate(loader):
        if idx > 0:
            data = np.vstack((data, torch.squeeze(batch[s]).numpy()))
    return data




def ml_generate_lfp(model, signal:np.ndarray, input_size:int, output_size:int, num_gen_points:int, scaling_method, only_generated:bool):
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    num_predictions =  int(num_gen_points/output_size) + 1
    signal = scaler.fit_transform1d(signal) # κανονικοποίηση. Θα μπορούσε να χρησιμοποιηθεί και σκέτη η scaler.transform1d(signal) επειδή ο scaler εχει γίνει fit για ολα τα training data
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




def visual_fc_comparison(model_generated_signal, actual_signal, fs, domain):
    
    # compare the two time-series (i.e. comparison in the time domain)
    if domain in ['time', 'both']: 
        plt. plot(actual_signal, label = 'actual_signal')
        plt. plot(model_generated_signal, label = 'generated signal')
        plt.legend()
        plt.title('Visual comparison between generated and actual signal')
        plt.show()
        plt.close()

    # compare the two time-series' frequencies (i.e. comparison in the frequency domain)
    if domain in ['frequency', 'both']:
        f1, Pxx_1 = sn.periodogram(actual_signal, fs=fs, return_onesided=True, scaling='density')
        f2, Pxx_2 = sn.periodogram(model_generated_signal, fs=fs, return_onesided=True, scaling='density')
        plt.plot(f1,Pxx_1, label = 'actual_signal')
        plt.plot(f2,Pxx_2, label = 'generated signal')
        plt.suptitle('comparison of generated and actual signal frequencies (Fourier-PSD)')
        plt.title(f'sampling frequency is {fs} due to downsampling', fontsize = 9)
        plt.legend()
        plt.show()
        plt.close()

    if domain == 'cross-correlation':
        sn_corr = sn.correlate(actual_signal, model_generated_signal)/(actual_signal.std()*model_generated_signal.std()*len(actual_signal))
        plt.plot(sn_corr); plt.title('normalized cross correlation'); plt.show(); plt.close() # η corss-correlation ΕΙΝΑΙ κανονικοποιήμένη στο [-1,1]




def produce_metric_list(model, model_type, test_series, starting_points_list, num_gen_points, input, output, scaling_method, metric, make_barplot):
    metric_list = []
    fs = 1/(test_series[1,3] - test_series[1,2])
    test_signal = test_series[0,:]

    for starting_point in starting_points_list:
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
        elif metric == 'Pearson r': # Pearson r is equal to normalized cross-corelation at zero time-lag
            model_metric, _ = stats.pearsonr(actual_signal, gen_signal)
        elif metric == 'max Cross-cor': # this computes the maximum crsss-corelation between the 2 signals
            model_metric = np.max(sn.correlate(actual_signal, gen_signal)/(actual_signal.std()*gen_signal.std()*len(actual_signal)))
        elif metric == 'RMS-PSD': # this is the root-mean-square-error of the PSD's. It is a metric fo how similar frequencies the two signals have
            _, Pxx_1 = sn.periodogram(actual_signal, fs=fs, return_onesided=True, scaling='density')
            _, Pxx_2 = sn.periodogram(gen_signal, fs=fs, return_onesided=True, scaling='density')
            model_metric = np.sqrt(mean_squared_error(Pxx_1, Pxx_2))

        metric_list.append(model_metric)
    if make_barplot: make_metric_barplot(starting_points_list, metric_list, metric)
    return metric_list




def produce_metric_samples(test_series, lstm_model, comparing_model, number_of_starting_points, num_gen_points, scaling_method, metric):
    """This function takes a test_series, an lstm_model, a number of starting points and the name of a metric and it creates two lists. The function crates random starting
    points for forecasting, forecasts the test series with the LSTM and the other comparing methods, and it produces a list of the calculated metric produced by the
    comparison of the forecasted signal and the actual signal in each starting point"""

    input_size = lstm_model.input_size
    output_size = lstm_model.output_size

    starting_points_list = np.random.randint(input_size, test_series.shape[1] - num_gen_points, size = number_of_starting_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν 
    # από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση

    lstm_metric_list = produce_metric_list(lstm_model, 'lstm', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    if comparing_model in ['white_noise', 'pink_noise', 'brownian_noise']:
        comparing_metric_list = produce_metric_list('None', comparing_model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    else:
        comparing_metric_list = produce_metric_list(comparing_model, 'ml', test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric, make_barplot=False)
    
    make_metric_barplot(starting_points_list, lstm_metric_list, metric)
    return lstm_metric_list, comparing_metric_list


# def produce_metric_samples(test_series, lstm_model, comparing_model, number_of_st_points, num_gen_points, scaling_method, metric):
#     """This function takes a test_series, an lstm_model, a number of starting points and the name of a metric and it creates two lists. The function crates random starting
#     points for forecasting, forecasts the test series with the LSTM and the other comparing methods, and it produces a list of the calculated metric produced by the
#     comparison of the forecasted signal and the actual signal in each starting point"""

#     input_size = lstm_model.input_size
#     output_size = lstm_model.output_size

#     lstm_metric_list = []
#     comparing_metric_list = []
#     # starting_points_list = np.random.randint(1, test_series.shape[1], size = number_of_st_points) # αυτή η εντολύ απαιτούσε τα δυο if στην αρχή του βρόχου for πιο κάτω, για αυτό αντικαταστάθηκε με την επόμενη που δεν τα χρειάζεται
#     starting_points_list = np.random.randint(lstm_model.input_size , test_series.shape[1] - num_gen_points, size = number_of_st_points)
#     fs = 1/(test_series[1,3] - test_series[1,2])
#     test_signal = test_series[0,:]
#     tensor_test_signal = torch.from_numpy(test_signal).clone().float()

#     for starting_point in starting_points_list:
#         # if starting_point < lstm_model.input_size: starting_point = lstm_model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
#         # if starting_point > test_series.shape[1] - num_gen_points: starting_point = test_series.shape[1] - num_gen_points # πρέπει το starting point να απέχει τουλάχιστον num_gen_points από το τέλος του σήματος, για να μπορούν να παραχθούν αρκετά σημεία
#         actual_signal = test_signal [starting_point : starting_point + num_gen_points]
#         lstm_gen_signal = lstm_generate_lfp(lstm_model, tensor_test_signal[:starting_point], num_gen_points, scaling_method, only_gen_signal=True)
#         # linear_gen_signal = ml_generate_lfp(linear, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
#         # dummy_gen_signal = ml_generate_lfp(dummy, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
#         if comparing_model not in ['white_noise', 'pink_noise', 'brownian_noise', 'None']:
#             comparing_gen_signal = ml_generate_lfp(comparing_model, test_signal[:starting_point], input_size, output_size, num_gen_points, scaling_method, only_generated = 1)
#         if comparing_model == 'white_noise':
#             noise = produce_noise (beta = 0, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
#             comparing_gen_signal = noise * (actual_signal.std()/noise.std()) # bring the noise to the same scale as the signal
#         if comparing_model == 'pink_noise': # Actually LFP signal's PSD follows pink noise morphology
#             noise = produce_noise (beta = 1, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
#             comparing_gen_signal = noise * (actual_signal.std()/noise.std()) # bring the noise to the same scale as the signal
#         if comparing_model == 'brownian_noise':
#             noise = produce_noise (beta = 2, samples = actual_signal.shape, fs=fs, freq_range = [0.001, 200]) # the noise is filtered in the range of LFP -> [0,200] Hz
#             comparing_gen_signal = noise * (actual_signal.std()/noise.std()) # bring the noise to the same scale as the signal


#         if metric == 'MAE':
#             metric_lstm = mean_absolute_error(actual_signal, lstm_gen_signal)
#             if comparing_model != 'None': metric_comparing = mean_absolute_error(actual_signal, comparing_gen_signal)
#         elif metric == 'RMSE':
#             metric_lstm = np.sqrt(mean_squared_error(actual_signal, lstm_gen_signal))
#             if comparing_model != 'None':metric_comparing = np.sqrt(mean_squared_error(actual_signal, comparing_gen_signal))
#         elif metric == 'Pearson r': # Pearson r is equal to normalized cross-corelation at zero time-lag
#             metric_lstm, _ = stats.pearsonr(actual_signal, lstm_gen_signal)
#             if comparing_model != 'None':metric_comparing, _ = stats.pearsonr(actual_signal, comparing_gen_signal)
#         elif metric == 'max Cross-cor': # this computes the maximum crsss-corelation between the 2 signals
#             metric_lstm = np.max(sn.correlate(actual_signal, lstm_gen_signal)/(actual_signal.std()*lstm_gen_signal.std()*len(actual_signal)))
#             if comparing_model != 'None':metric_comparing = np.max(sn.correlate(actual_signal, comparing_gen_signal)/(actual_signal.std()*comparing_gen_signal.std()*len(actual_signal)))
#         elif metric == 'RMS-PSD': # this is the root-mean-square-error of the PSD's. It is a metric fo how similar frequencies the two signals have
#             _, Pxx_1 = sn.periodogram(actual_signal, fs=fs, return_onesided=True, scaling='density')
#             _, Pxx_2 = sn.periodogram(lstm_gen_signal, fs=fs, return_onesided=True, scaling='density')
#             if comparing_model != 'None':_, Pxx_3 = sn.periodogram(comparing_gen_signal, fs=fs, return_onesided=True, scaling='density')
#             metric_lstm = np.sqrt(mean_squared_error(Pxx_1, Pxx_2))
#             if comparing_model != 'None':metric_comparing = np.sqrt(mean_squared_error(Pxx_1, Pxx_3))

#         lstm_metric_list.append(metric_lstm)
#         if comparing_model != 'None':comparing_metric_list.append(metric_comparing)
#     make_metric_barplot(starting_points_list, lstm_metric_list,metric)
#     if comparing_model != 'None': return lstm_metric_list, comparing_metric_list
#     else: return lstm_metric_list



def statistical_comparison(lstm_metric_list, comparing_metric_list, normality_test):
    '''Υπαρχουν 3 κριτήρια που πρέπει να πληρούνται για τη χρήση παραμετρικών κριτηρίων όπως το t-test: 1) οι κατανομές των δειγμάτων να είναι κανονικές, 2) οι κατανομές να 
    έχουν ίσες διακυμάνσεις (κάτι που δε χρειάζεται στα εξαρτημένα δείγματα), και τα δεοδμένα να είναι ποσοτικά. Οπότε ουσιαστικά εδω πρέπει να ελεγχθει μόνο η κανονικότητα'''
    '''The Shapiro–Wilk test is more appropriate method for small sample sizes (<50 samples) although it can also be handling on larger sample size while Kolmogorov–Smirnov 
    test is used for n ≥50'''
    lstm_metric_mean = sum(lstm_metric_list)/len(lstm_metric_list)
    other_metric_mean = sum(comparing_metric_list)/len(comparing_metric_list)
    print('LSTM metric computed mean is', lstm_metric_mean)
    print('2nd method metric computed mean is', other_metric_mean)
    if normality_test == 'SW':
        _ , p1 = stats.shapiro(lstm_metric_list)
        _ , p2 = stats.shapiro(comparing_metric_list)
    if normality_test == 'KS':
        _, p1 = stats.kstest(lstm_metric_list, 'norm')
        _, p2 = stats.kstest(comparing_metric_list, 'norm')
    # plt.hist(lstm_metric_list); plt.show()#; plt.close() -> το ιστόγραμμα μπορεί να δείξει που προβλέπει καλύτερα ο κάθε αλγόριθμος το σήμα (εξαρτάται και από το σε τι έχει εκπαιδευτεί)
    # plt.hist(other_metric_list); plt.show()#; plt.close()
    if p1 > 0.05 and p2 > 0.05: # the null hypothesis that the data come from a norma distribution, cannot be rejected
        print ('The metrics in different starting points are distributed normaly. t-test of related samples will be carried out')
        stat_test = stats.ttest_rel(lstm_metric_list, comparing_metric_list)
    else:
        print ('The metrics in different starting points are not distributed normaly. Wilcoxon (T) will be carried out')
        print ('The fact that metrics in different starting points are not distributed normally, hints that foracasting may not be equally effective in different parts of the LFP signal')
        stat_test = stats.wilcoxon(lstm_metric_list, comparing_metric_list)
    print('results:', stat_test)
    print('p-value is', stat_test.pvalue)
    if stat_test.pvalue < 0.05: print('Thus there is statistically significant difference between the two means')
    elif stat_test.pvalue >= 0.05: print('Thus the null hypothesis that means of metrics are equal, cannot be rejected')



# Αυτή η μέθοδος αναπαριστά γραφικά, πόσο καλή είναι η προβλεψη κατά μήκος του test σήματος LFP (εφόσον χρησιμοποιείται ένα σήμα για testing)
def make_metric_barplot(starting_points_list, lstm_metric_list,metric):
    starting_points_list = np.array(starting_points_list)
    lstm_metric_list = np.array(lstm_metric_list)
    indices = starting_points_list.argsort()
    starting_points_list_sorted = starting_points_list[indices]
    starting_points_list_sorted_str = starting_points_list_sorted.astype(str)
    lstm_metric_list_sorted = lstm_metric_list[indices]
    #plt.plot(starting_points_list_sorted, lstm_metric_list_sorted)
    plt.bar(starting_points_list_sorted_str, lstm_metric_list_sorted)
    plt.title(f'Metric: {metric}')
    plt.xticks(rotation = 'vertical' )
    plt.show()
    plt.close()



def produce_noise(beta, samples, fs, freq_range):
    # beta ->  the exponent  [1.f^(beta)] -> beta = 0 => white noise, beta = 1 => pink noise, beta = 2 => brownian noise
    # samples -> number of samples to generate
    noise = cn.powerlaw_psd_gaussian(beta, samples)
    if fs>500: # αν το σήμα είναι έντονα downsampled (πάνω από 30 φορές) δεν έχει νόημα το bandpass στο θόρυβο, επειδή έχει μικρύνει πολύ και στο πραγματικό σήμα LFP
        noise = bandpass(noise, freq_range, fs, poles = 5) # κάνει bandpass το θόρυβο στις συχνότητες που έχει γίνει και το LFP για να είναι όμοιος με αυτό
    return noise



def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sn.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sn.sosfiltfilt(sos, data)
    return filtered_data

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



if  __name__ == "__main__":
    main()
    
    
