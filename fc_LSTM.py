"""This file uses an LSTM-RNN for the forecasting of the LFP signal"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import signal_handler
import compare_module
torch.backends.cudnn.benchmark = True # added for the -> cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.This error may appear if you passed in a non-contiguous input

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Παραλαγές της εκπαίδευσης οι οποίες μάλλον ΔΕ θα κρατηθούν στον τελικό κώδικα (GLOBAL VARIABLES)

PATH = 'D:/Files/peirama_dipl/' # my PC path
# PATH = '/home/skoutinos/' # remote PC path

run_to_gpu_all = 0 # στέλνει όλα τα δεδομένα στη gpu πριν την εκπαίδευση, !!!!!! ΠΡΟΣΟΧΗ!! όπως έχεις γράψει τον κώδικα αν στείλεις όλα τα δεδομένα στη gpu τότε το
#normalization θα γίνει στη gpu που παίρνει πάρα πολύ χρόνο. Δες το training loop για να το καταλάβεις.
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
def main1():
    tag= 'All_EA'  # determines which groups of files will be loaded, and used for training
    downsample_scale = 10000 # determines how many time will the signal be downsampled
    sliding_window_step = 1 # this is the number of the window sliding for the creation of the windows that will be used for training

    input_size = 100 # this is the number of the input_data of the LSTM, i.e. the number of points used for forecasting
    hidden_state_dim = 50 # the size of the hidden/cell state of LSTM
    num_layers = 1 # the number of consecutive LSTM cells the nn.LSTM will have (i.e. number of stacked LSTM's)
    output_size = 10 # this is th number of output_data of the LSTM, i.e. the future points forecasted by the LSTM

    extract_data = 0
    # cut_with_numpy = False # είναι keyword argument πλέον
    # return_loaders = True # είναι keyword argument πλέον
    batch_size = 4
    epochs = 10
    lr = 0.1 # optimizers learning rate
    train_model= 0 # for True it trains the model, for False it loads a saved model # ΠΡΟΣΟΧΗ αν κάνεις load μοντέλο που το έχεις εκπαιδεύσει με άλλο output_type προφανώς θα προκύψει σφάλμα
    model_number = 4

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    if fc_move_by_one: input_size = 100 
    if fc_move_by_one: output_size = input_size # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση

    # Import data
    save_load_path= PATH + 'project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    # file_name = 'LSTM_fc_data_All_ds1000.npy' # remote PC load file
    # save_load_path = PATH + 'project_files/' + file_name # remote PC load file
    if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path) 
    if not(extract_data): lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    # prepare data
    if train_model: train_loader, val_loader, scaler  = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size)

    ## NN instance creation
    lstm_model = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(lstm_model.parameters(), lr)
    optimizer = optim.Adam(lstm_model.parameters(), lr)
    # optimizer = optim.LBFGS(lstm_model.parameters(), lr) # for it to work u have to craete a closure function. See pytorch documentation fo more info

    # # try forward method with a (εχεις φτιάξει ένα LSTM που παίρνει ένα τενσορα fc_num στοιχείων και επιστρέφει ένα τενσορα 1 στοιχείου
    # a=np.linspace(0,3,input_size); a=torch.tensor(a, dtype=torch.float32); a=torch.unsqueeze(a,0); a=torch.unsqueeze(a,0);print(a.shape)
    # arr = lstm_model(a) # input must be dims (batch_size, sequence_length, input_size)
    # print('arr output shape is', arr.shape); print(arr)

    if train_model:
        if run_to_gpu_all or run_to_gpu_batch : lstm_model = lstm_model.to(device)
        model, training_string = training_lstm(lstm_model, criterion, optimizer, epochs, train_loader, val_loader, scaler, save_name='LSTM_forecasting_model', measure_train_time=True) 
        create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string)
        save_params(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, sliding_window_step, scaling_method, tag, input_size, output_size, training_string)
    if not(train_model):
        dict_train = load_params(model_number)
        input_size, hidden_state_dim, num_layers, output_size = dict_train['input_size'], dict_train['hidden_state_dim'], dict_train['num_layers'], dict_train['output_size'] # χρειάζονται για το loading του LSTM
        downsample_scale, scaling_method = dict_train['downsample_scale'], dict_train['scaling_method'] # χρειάζονται για το generate/compare
        scaler.change_scaling_method(scaling_method)
        model = load_lstm(model_number, input_size, hidden_state_dim, num_layers, output_size)



    ###  test trained LSTM
    print('\nTest trained LSTM: Compare actual and generated signal')

    generate_or_compare = 'compare' # 'generate' , 'compare'
    starting_point = 4000 # 4000 350000 700000
    if starting_point < model.input_size: starting_point = model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
    num_gen_points = output_size # 5 * output_size
    test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
    # test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
    print('length of test series is ', test_series.shape)
    fs = 1/(test_series[1,3] - test_series[1,2])
    test_signal = torch.from_numpy(test_series[0,:]).clone().float()
    if generate_or_compare == 'generate': gen_signal = lstm_generate_lfp(model, test_signal[:starting_point], num_gen_points, scaler, only_gen_signal=1); plt.plot(gen_signal); plt.show(); plt.close()
    if generate_or_compare == 'compare':
        base_signal, gen_signal, = produce_comparing_signals(model, test_signal, starting_point, num_gen_points, scaler)
        compare_module.compare_for_trained_lstm(base_signal, gen_signal, starting_point, fs)
        # compare_noise(base_signal, fs)


#------------------------------------------------------------------------------------------------------------------------------------------------------------

def main2():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import scipy.stats as stats 

    tag= 'All_EA'  # determines which groups of files will be loaded, and used for training
    downsample_scale = 10000
    sliding_window_step = 1 # this is the number of the window sliding for the creation of the windows that will be used for training
    batch_size = 4
    model_number = 4 # καθορίζει ποιο LSTM μοντέλο θα φορτωθεί
    number_of_st_points = 40 # καθορίζει πόσα τυχαία σημεία έναρξης της πρόβλεψης θα παρθούν για τη στατιστική σύγκριση των μεθόδων

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]
    ndarray_scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    tensor_scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    # Import data
    save_load_path= PATH + 'project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy' # my PC load file
    lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape)

    
    # load LSTM model
    dict_train = load_params(model_number)
    input_size, hidden_state_dim, num_layers, output_size = dict_train['input_size'], dict_train['hidden_state_dim'], dict_train['num_layers'], dict_train['output_size'] # χρειάζονται για το loading του LSTM
    downsample_scale, scaling_method = dict_train['downsample_scale'], dict_train['scaling_method'] # χρειάζονται για το generate/compare
    tensor_scaler.change_scaling_method(scaling_method)
    ndarray_scaler.change_scaling_method(scaling_method)
    model = load_lstm(model_number, input_size, hidden_state_dim, num_layers, output_size)

    # train linear and dummy with the same data
    linear, dummy = train_older_methods(lfp_data, ndarray_scaler, input_size, output_size, sliding_window_step, batch_size)

    # load test series and random starting points
    test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale) # το τεστ είναι ένα ολόκληρο σήμα, ώστε διαφορετικά starting points να ελέγξουν το forecasting σε όλες τις καταστάσεις (EA, IA, SLA)
    # test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy') # for remote pc
    print('length of test series is ', test_series.shape)
    starting_points_list = np.random.randint(1, test_series.shape[1], size = number_of_st_points)

    # generate signals and compare them with actual signal
    # for starting_point in starting_points_list:

    lstm_metric_list = []
    linear_metric_list = []
    dummy_metric_list = []

    for starting_point in starting_points_list:
        if starting_point < model.input_size: starting_point = model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
        num_gen_points = 3 * output_size + 25 # 5 * output_size
        fs = 1/(test_series[1,3] - test_series[1,2])
        test_signal = test_series[0,:]
        tensor_test_signal = torch.from_numpy(test_signal).clone().float()

        # δε χριεάζεται. try_lfp_norm = scaler.fit_transform1d(test_signal.numpy()) # κανονικοποίηση του σήματος. Μια πολύ μιρκή λαθροχειρία είναι ότι η μεση τιμή εξάγεται και από το σήμα που θα προβλεφθεί, οπότε λαμβάνεται μια πληροφορία από την πρόβλεψη
        linear_gen_signal = ml_generate_lfp(linear, test_signal[:starting_point], input_size, output_size, num_gen_points, ndarray_scaler, only_generated = 1)
        # δε χριεάζεται linear_gen_signal  = scaler.inverse1d(linear_gen_signal_norm)
        dummy_gen_signal = ml_generate_lfp(dummy, test_signal[:starting_point], input_size, output_size, num_gen_points, ndarray_scaler, only_generated = 1)
        # δε χριεάζεται dummy_gen_signal  = scaler.inverse1d(dummy_gen_signal_norm)
        lstm_gen_signal = lstm_generate_lfp(model, tensor_test_signal[:starting_point], num_gen_points, tensor_scaler, only_gen_signal=1)
        actual_signal = test_signal [starting_point : starting_point + num_gen_points]

        
        MAE_lstm = mean_absolute_error(actual_signal, lstm_gen_signal)
        MAE_linear = mean_absolute_error(actual_signal, linear_gen_signal)
        MAE_dummy = mean_absolute_error(actual_signal, dummy_gen_signal)
        lstm_metric_list.append(MAE_lstm)
        linear_metric_list.append(MAE_linear)
        dummy_metric_list.append(MAE_dummy)

    print(sum(lstm_metric_list)/len(lstm_metric_list), sum(linear_metric_list)/len(linear_metric_list))
    print(stats.ttest_rel(lstm_metric_list, linear_metric_list))
    print('a')
    


    



def train_older_methods(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size):
    from sklearn import model_selection
    import sklearn.linear_model as lr
    from sklearn.dummy import DummyRegressor

    # scaling_data -> if scaling_manner = norm_all_data, then data are normazized inside the prepare_data function, else they are normalized here
    if scalling_manner != 'norm_all_data':
        lfp_data = scaler.normalize2d(lfp_data) # κανονικοποιεί το σήμα
        scaler.fit2d(lfp_data) # εξάγει κοινές παραμέτρους κανονικοποίησης για όλο το σήμα

    x_data, y_data, scaler = prepare_data(lfp_data, scaler, input_size, output_size, sliding_window_step, batch_size, cut_with_numpy=1, return_loaders=0)
    x_data, y_data = x_data.numpy(), y_data.numpy()
    x_data=np.reshape(x_data, (x_data.shape[0]*x_data.shape[1], x_data.shape[2]))
    y_data=np.reshape(y_data, (y_data.shape[0]*y_data.shape[1], y_data.shape[2]))
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.9)

    linear = lr.LinearRegression()
    linear.fit(x_train, y_train)
    print('Linear regression R^2 score is ', linear.score(x_test, y_test))
    # pred = linear.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(x_train, y_train)
    print('Dummy regresson R^2 score is ', dummy.score(x_test, y_test))
    # pred = dummy.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting

    return linear, dummy


def ml_generate_lfp(model, signal:np.ndarray, input_size:int, output_size:int, num_gen_points:int, scaler, only_generated:bool):
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


### παλία μέθοδος -> διέγραψέ τη
# def ml_generate_lfp(model, signal:np.ndarray, input_size:int, output_size:int, num_gen_points:int, scaling_method:str, only_generated:bool):
#     num_predictions =  int(num_gen_points/output_size) + 1
#     # scaling_power = 6
#     # signal = signal*(10**scaling_power)
#     signal, _ = signal_handler.normalize_signal(signal.reshape(-1, 1), method= scaling_method, direction = 'normalize', scaler= 'None', scaling_power = 3) 
#     input_signal = signal[(len(signal)-input_size):]
#     _, denorm_scaler = signal_handler.normalize_signal(input_signal[:output_size].reshape(-1, 1), method= scaling_method, direction = 'normalize', scaler= 'None', scaling_power = 3)
#     input_signal = input_signal.reshape(1,-1) # πρέπει να είναι σε αυτή τη μορφή για να εισαχθεί στο predict
#     if only_generated: generated_signal = []
#     if not(only_generated): generated_signal= list(signal)

#     for i in np.arange(num_predictions):
#         new_pred = model.predict(input_signal)
#         new_pred = signal_handler.normalize_signal(new_pred.reshape(-1, 1), method= scaling_method, direction = 'inverse', scaler= denorm_scaler, scaling_power = 3)
#         if new_pred.shape == (): new_pred = new_pred[np.newaxis]; 
#         else: new_pred = new_pred[np.newaxis,:]
#         if new_pred.shape == (1,): new_pred = new_pred.reshape(1,1) # για output = 1 
#         input_signal = np.hstack((input_signal[:,output_size:], new_pred))  # θα ενώσω τα δύο κομμάτια μετακινόντας το κάποιες θέσεις
#         if new_pred.shape == (1,1): generated_signal.append(np.squeeze(new_pred)) # για output = 1 
#         else: generated_signal = generated_signal + list(np.squeeze(new_pred))
#     generated_signal = np.array(generated_signal)
#     if not(only_generated): generated_signal =  generated_signal[: len(signal)+num_gen_points]
#     if only_generated: generated_signal =  generated_signal[:num_gen_points]
#     return generated_signal #*(10**(-scaling_power))
# --------------------------------------------------------- DELETE IF YOU HAVE REPLACED IT WITH NEWER CODE  ----------------------------------------------------------------
# #### This part trains an autoregreesive model and a dummy regressor, and compares the results with the test signal. It also produces noise a compares the noise with the signal
# #### The purpose is to examine if generatde signals from autoregressor, dummy regressor and noise signal, are more similar with the base_signal, than the 
# #### genereatd signal from LFP
# def main2(): ## -> ΑΥΤΗ ΦΤΙΑΞΤΗ ΜΟΛΙΣ ΦΤΙΑΞΕΙΣ ΤΟ compare_module
#     from sklearn import model_selection
#     downsample_scale =10000
#     input_size =1000
#     output_size =100
#     tag ='All_EA'
#     downsample_scale = 10000
#     scaler1 = signal_handler.lfp_scaler(scaling_power=4)

#     x_data, y_data, scaler1  = prepare_data(downsample_scale, extract_data =1, tag=tag, numpy_data_float32 =0, scaler = scaler1 , input_size=input_size, 
#                                             output_size = output_size,  window_step =1, return_loaders = 0, cut_with_numpy = 1)
#     x_data, y_data = x_data.numpy(), y_data.numpy()
#     x_data=np.reshape(x_data, (x_data.shape[0]*x_data.shape[1], x_data.shape[2]))
#     y_data=np.reshape(y_data, (y_data.shape[0]*y_data.shape[1], y_data.shape[2]))
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.9)

#     test_time_series = signal_handler.time_series('test1', downsample_scale)
#     print('test_time_series shape is', test_time_series.shape)
#     starting_point = 1000
#     num_gen_points = 200

#     ### normalize  training data
#     scaling_method = 'z_normalization'
#     scaler = signal_handler.lfp_scaler(scaling_power=4)
#     x_train = scaler.normalize2d(x_train, scaling_method); y_train = scaler.normalize2d(y_train, scaling_method)
#     x_test = scaler.normalize2d(x_test, scaling_method); y_test = scaler.normalize2d(y_test, scaling_method)

#     ### produce results for compare
#     compare_module.dummy(x_train, y_train, x_test, y_test, test_time_series, starting_point, num_gen_points, scaling_method)
#     compare_module.linear_autoregression(x_train, y_train, x_test, y_test, test_time_series, starting_point, num_gen_points, scaling_method)
#     fs = 1/(test_time_series[1,3] - test_time_series[1,2])
#     compare_module.compare_noise(test_time_series[0,:], fs, downsample_scale)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### data preparation
def prepare_data(lfp_data_matrix, scaler, input_size, output_size, window_step, batch_size, cut_with_numpy=0, return_loaders=1):
    """This function prepares the data (i.e. normalizes, divide long signals, dreates windoed data, wraps them into loaders) and returs the train_loader and val_loader 
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

    if cut_with_numpy:
        # Δημιουργία δεδομένων εκπαίδευσης με κόψιμο τους σε παράθυρα όπου κάθε παράθυρο περιλαμβάνει τα target_data και input_data μιας forecasting δοκιμής/εκπαίδευσης
        data = np.lib.stride_tricks.sliding_window_view(lfp_data_matrix, window_size, axis=1, writeable=True)[:,::window_step,:]
        if fc_move_by_one: data = np.lib.stride_tricks.sliding_window_view(lfp_data_matrix, input_size + 1, axis=1, writeable=True)[:,::window_step,:]
        print('numpy windowed data are', data.shape)
        ## transform the data to a tensor, and then put them in loaders
        data = torch.from_numpy(data).float() # με το που εκτελείς αυτή την εντολή, τα windows παύουν να είναι views του numpy και αυτό αυξάνει σημαντικά τις ανάγκες σε μνήμη
        input_data = data[:,:, 0:input_size]
        target_data = data[:,:,input_size:window_size]
    if not(cut_with_numpy): # εδώ τα παράθυρα κόβονται αφού είναι tensors, οπότε παραμένουν views. Βέβαια μάλλον παύουν να είναι views όταν εισάγονται στους loaders
        data = torch.from_numpy(lfp_data_matrix).float()
        windowed_data = data.unfold(dimension=1, size = window_size, step = window_step)
        input_data = windowed_data[:,:, 0:input_size]
        target_data = windowed_data[:,:,input_size:window_size]
        print('torch windowed data are', windowed_data.shape)

    if not(return_loaders): return input_data, target_data, scaler
    if fc_move_by_one: target_data = windowed_data[:,:,1:output_size+1] # με πρόβλεψη επόμενων σημείων πλήθους ίσου με το input, μετατοπισμένων κατά μία θέση
    if run_to_gpu_all: input_data=input_data.to(device); target_data=target_data.to(device) # εδώ τα data σίγουρα παύουν να είναι views. Για αυτό πιο κάτω στο training μόνο σε αυτή την περίπτωση τα bathces δεν αντιγράφονται
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.9, 0.1]) 
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
def training_lstm(model, criterion, optimizer, epochs, train_loader, val_loader, scaler, save_name, measure_train_time:bool):
    """This is the NN training function. It recieves the typical parameters fo model, loss function (criterion), optimizer and epochs. 
    It also recieves as input the torch dataloader objects for training ana validation data. 
    save_name must be a string. If it's none then the trained model is not saved, otherwise the model is saved with file name the string.
    measure_train_time must be True or False. If it's True then the training time is calculated and printed"""

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
        if save_name != 'None' and val_mean_loss < val_mean_loss_old: torch.save(model.state_dict(), PATH + 'project_files/' + save_name + '.pt')
        
    if measure_train_time: toc = time.perf_counter()
    if measure_train_time: print ('whole training time is', toc - tic)

    ### plot train and validation losses
    plt.plot(range(num_epochs), losses_list, label = 'Train loss')
    plt.plot(range(num_epochs), val_losses_list, label = 'Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    return model, training_string



### creates and saves report of the training of the LSTM model to a text file
def create_training_report(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, window_step, scaling_method, tag, input_size, output_size, training_string):
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
    with open(PATH + '/project_files/training_log.txt', "w+") as file: file.write(whole_string)




# load the LSTM model if you have saved it, in order not to run training again if its time-consuming
def load_lstm(model_number, input_size, hidden_state_dim, num_layers, output_size):
    model = LSTM_fc(input_size, hidden_state_dim, num_layers, output_size) 
    model.load_state_dict(torch.load(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_forecasting_model.pt'))
    print('LSTM model has been loaded')
    return model



### saves the parameters of the LSTM model to a dictionary and then saves the dictionary to a picle file
def save_params(downsample_scale, hidden_state_dim, num_layers, batch_size, lr, window_step, scaling_method, tag, input_size, output_size, training_string):
    dict_param = {'downsample_scale':downsample_scale, 'hidden_state_dim':hidden_state_dim, 'num_layers':num_layers, 'batch_size':batch_size, 'lr':lr, 
                  'window_step':window_step, 'scaling_method':scaling_method, 'tag':tag, 'input_size':input_size, 'output_size':output_size,  'training_string':training_string}
    with open(PATH + '/project_files/LSTM_params.pkl', "wb") as file: pickle.dump(dict_param, file)



### loads the saved parameters of the LSTM model from the picled dictionary file
def load_params(model_number:int):
    with open(PATH + 'project_files/models/model' + str(model_number) + '/LSTM_params.pkl', 'rb') as file: dict_param = pickle.load(file)
    return dict_param




### Generate/forecast LFP signal with the LSTM model
def lstm_generate_lfp(model, signal, num_gen_points:int, scaler, only_gen_signal:bool):
    """""This function ganerates a number of points (num_gen_points), in the end of the given LFP signal (signal) by using a trained NN (model)
        1) model -> is the LSTM forecasting model
        2) signal -> must be an lfp signal in tensor form bigger in length than the input_size of the LSTM model
        3) num_gen_points -> is the number of the points that will be generated/forecasted in total
        4) scaler -> is the scaler object used for scaling/unscaling the data. This class exist in the signal_handler module
    """""
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




### παίρνει ένα σήμα και επιστρέφει δύο σήματα ίδιου μήκους -> 1) το αρχικό σήμα κομμένο, 2) το generated σήμα -> ώστε να μπορούν να συγκριθούν για το foracasting τους
def produce_comparing_signals(model, signal:torch.Tensor, starting_point:int, num_gen_points:int, scaler):
    """"This functions recieves the necessary parameters and produces with the LSTM model, the signals which will be compared in the compare function
    Starting point must be bigger than model.iput_size because we can't start producing point i there isn't enough input points before the starting point"""
    if starting_point < model.input_size: starting_point = model.input_size # δεν πρέπει το starting point να είναι μικρότερο από το input, επειδή δε θα υπάρχουν αρκετά σημεία για input πίσω από το starting point
    gen_signal = lstm_generate_lfp(model, signal[:starting_point], num_gen_points, scaler, only_gen_signal = False) # παράγει το forecasted σήμα ξεκινόντας από το starting point
    signal_baseline = signal[: len(gen_signal)].numpy() # παίρνει κομμάτι από το αρχικό σήμα, που να έχει ίδιο μήκος με το gen_signal -> έτσι μπορούν να συγκριθούν
    return signal_baseline, gen_signal


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if  __name__ == "__main__":
    # main1()
    main2()
    # main3()
    # main_LSTM_train()
    
