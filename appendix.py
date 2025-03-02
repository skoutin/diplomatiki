

import numpy as np
import matplotlib.pyplot as plt
import signal_handler
# import compare_module # deprecated code file
# import fc_LSTM # deprecated code file
import pickle
import time

import sklearn.linear_model as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor

from hmmlearn import hmm

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing # holt-winters is an expantion of ExpSmoothing that accounts for trend (Holt) and seasonality (winters)/ LFP has neither
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
import scipy.signal as sn 
from fc_LFP import norm_cross_cor, visual_fc_comparison

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

remote_PC = False
if not(remote_PC): PATH = 'D:/Files/peirama_dipl/' # my PC path
if remote_PC: PATH = '/home/skoutinos/' # remote PC path


def main(fc_method, data_type_for_hmm_or_classic):
    tag= 'All_WT_0Mg'
    downsample_scale = 10
    sliding_window_step = 12000
    input_size = 499
    output_size = 1

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]

    data_form_list = ['windowed_parts', 'concatenated_windowed_parts', 'single_lfp']
    # data_form = data_form_list[0] # not needed, every method calls its own

    extract_data = 0
    train_model = True #True #False
    save_load_name = saveloader(fc_method, saveload='None') # if you change <saveload='None'> to <saveload='save'> the trained model will be saved

    if train_model:
        if fc_method in ["exponential_smoothing", "autoregressive", "moving average", "arma", "arima", "optimal_arima"]:
            time_lags = 7 # ο σκοπός είναι οι μεταβλητές να μη βελτιστοποιηθούν με κλασσικές μεθόδους (πχ adfuller test) αλλά με GridSearchCV
            diff = 1 # ο σκοπός είναι οι μεταβλητές να μη βελτιστοποιηθούν με κλασσικές μεθόδους (πχ adfuller test) αλλά με GridSearchCV -> αλλωστε το LFP μάλλον δε μπορεί να γίνει stationary
            q_lag = 4 # ο σκοπός είναι οι μεταβλητές να μη βελτιστοποιηθούν με κλασσικές μεθόδους (πχ adfuller test) αλλά με GridSearchCV
            if data_type_for_hmm_or_classic == 'combined windows': 
                data_form = data_form_list[1] # for training with concatenated parts
                data, _ = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form, extract_data=extract_data)
            if data_type_for_hmm_or_classic == 'whole_series': 
                data_form = data_form_list[2] # for training with a whole lfp signal
                data = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form, extract_data=extract_data)
            # plot_afc_pafc(data)
            # find_diff_order(data)
            # _, diff = find_diff_order(data)
            model = train_classic_fc_models(data, fc_method, time_lags, diff, q_lag, model_save_name=save_load_name)


        elif fc_method in ["linear", "decision_tree", "random_forest", "K_nearest_regression", "dummy", "polynomial", "support_vector_regression-muti", 'support_vector_regression-single']:
            data_form = data_form_list[0]
            x_data, y_data = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form='windowed_parts', extract_data=extract_data)
            # x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size=0.9)
            model = train_ml_model(fc_method, x_data, y_data, model_save_name=save_load_name)


        elif fc_method == 'hmm':
            with_mixture_emissions = 1 # if True GMMHMM is used which is an HMM with Gaussian mixture emmision probabilities
            n_components = 10 # this is the number of hidden_states that the HMM will have (HMM's have hidden states and observal states. Observable states are the values of the signal)
            n_mix = 7 # number of states in the GMM (Gaussian Mixture Models) i.e. how many gaussians will be used for each emmision

            if data_type_for_hmm_or_classic == 'whole_series':
                data_form = data_form_list[2] # for training with a whole lfp signal
                data = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form, extract_data=extract_data)
                lengths = 'None'
            if data_type_for_hmm_or_classic == 'combined windows':
                data_form = data_form_list[1] # for training with concatenated parts
                data,  lengths = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form, extract_data=extract_data)
            model = train_hmm(data, n_components, n_mix, with_mixture_emissions, data_form, lengths, model_save_name=save_load_name)


        elif fc_method == 'mlp':
            num_layers = 4
            hidden_size = 100
            num_epochs = 60
            device = 'cpu'
            batch_size = 256
            learning_rate = 0.001

            data_form = data_form_list[0]
            train_data, val_data = prepare_training_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, data_form, extract_data=extract_data)
            train_tensor = torch.from_numpy(train_data).float(); val_tensor = torch.from_numpy(val_data).float()

            dataset = torch.utils.data.TensorDataset(train_tensor, val_tensor)
            # το γεγονός ότι γίνεται random split στα παράθυρα προκαλεί temporal leakage, και καθιστά τσ validation scores μη αξιόπιστα. Θα έπρεπε είτε τα validation παράθυρα να είναι μετά από όλα τα trainin παράθυρα, είτε να προέρχονται από μια νέα χρονοσειρά όπως γίνται στο testing
            train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.8, 0.2])
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=train_data.__len__()
            val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=val_data.__len__()
            
            # #try forward method
            # arr = torch.ones((7,input_size))
            # model = MLP_autoregr(input_size, hidden_size, num_layers, output_size)
            # res = model(arr)

            mlp_model = MLP_autoregr(input_size, hidden_size, num_layers, output_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(mlp_model.parameters(), learning_rate)

            model, _ = mlp_reg_training(mlp_model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
            model.eval()

    elif not(train_model):
        if fc_method == 'mlp': 
            checkpoint = torch.load(PATH + 'project_files/MLP_fc_regressor.pt')
            args = checkpoint['model_args']
            model = MLP_autoregr(**args) 
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else: model = saveloader(fc_method, saveload='load')



    num_gen_points = output_size #3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
    if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό
    starting_point = 322
    test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale)
    # gen_signal = generate_lfp(model, test_series, starting_point, num_gen_points, input_size, output_size, fc_method, scaling_method)

    number_of_starting_points = 100
    index_range = np.arange(input_size, test_series.shape[1] - num_gen_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    starting_points_list = np.random.choice(index_range, size = number_of_starting_points, replace=False)
    # MAE_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    # print(MAE_list)
    create_metric_results(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method)
    

    plot_starting_point = np.random.randint(input_size, test_series.shape[1])
    actual_signal = test_series[0, plot_starting_point : plot_starting_point + num_gen_points]
    model_generated_signal = generate_lfp(model, test_series, starting_point, num_gen_points, input_size, output_size, fc_method, scaling_method)
    fs = 1/(test_series[1,3] - test_series[1,2])
    visual_fc_comparison(actual_signal, fs, model_generated_signal, ml_method1=fc_method, domain = 'both', save=False)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prepare_training_data(tag, downsample_scale, input_size, output_size, window_step, scaling_method, data_form, extract_data, float_32 = True):
    save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
    if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path)
    if not(extract_data): lfp_data = np.load(save_load_path)
    if float_32: lfp_data = lfp_data.astype(np.float32) # κάνει τα δεδομένα float32 λόγω προβλημάτων μνήμης στην εκπαίδευση των αλγόριθμων
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    window_size = input_size + output_size
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    windowed_data = np.lib.stride_tricks.sliding_window_view(lfp_data, window_size, axis=1, writeable=True)[:,::window_step,:]
    print(f'windowed data have shape {windowed_data.shape}')

    if data_form == 'windowed_parts': # για ML μεθόδους, ή MLP χρησιμοποίησε αυτά τα δεδομένα
        windowed_data = np.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1],window_size)) # αυτή η εντολή κόβει τα παράθυρα και τα μπλέκει. Είναι καλύ για ml αλγόριθμους
        if scaling_method!= 'None': windowed_data = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά
        x_data = windowed_data[:,0:input_size]
        y_data = windowed_data[:,input_size:window_size]
        return x_data, y_data
    
    if data_form == 'concatenated_windowed_parts': # για HMM και παραδοσιακους αλγοριθμους χρησιμοποίησε αυτά τα δεδομένα
        if scaling_method!= 'None': windowed_data = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά
        training_data = np.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1]*window_size)) # όλα τα παράθυρα ενώνονται σε μια χρονοσειρά
        lengths = np.ones(windowed_data.shape[0]*windowed_data.shape[1])*window_size
        lengths = lengths.astype(int)
        return training_data, lengths

    if data_form == 'single_lfp': # για HMM και παραδοσιακους αλγοριθμους χρησιμοποίησε αυτά τα δεδομένα
        training_data = np.hstack([lfp_data[i,:] for i in range(6)])
        if scaling_method!= 'None': training_data = scaler.fit_transform1d(training_data)
        return training_data

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stationarity is a prerequisite for the MA, AR/AutoReg and ARMA model. ARIMA model doesn't need stationarity because it transforms the data (the 'I' part), making them stationary
def check_stationarity (time_series:np.ndarray, tranformation:str):
    if tranformation == 'diff': time_series = np.diff(time_series, n=1)
    if tranformation == 'log':
        for idx in range(time_series.shape[0]):
            if time_series[idx] > int(0) : time_series[idx] = np.log(time_series[idx])
            if time_series[idx] < int(0) : time_series[idx] = -np.log(-time_series[idx])
            if time_series[idx] == int(0) : time_series[idx] = 0

    sig = adfuller(time_series)[1]
    print('p-value of ADF is', sig)
    if sig < 0.05: print('Time-series is stationary, with transformation: ' + tranformation)
    else: print('Time-series is non-stationary, with transformation: ' + tranformation)


# differences the series until they become stationary. Stationarity is a prerequisite for the MA, AR/AutoReg and ARMA model, so the differented series should be inserted 
# in these models ideally. Also this function return the i factor in order to be used in an ARIMA model.
def find_diff_order(time_series):
    i=1
    sig = adfuller(time_series)[1]
    while sig >=0.05:
        i=i+1
        time_series = diff(time_series, k_diff=i, k_seasonal_diff=None, seasonal_periods=None) # this function uses np.diff in its code
        # time_series = np.diff(time_series, n=i) # another method of differencing by numpy
        sig = adfuller(time_series)[1]
    # print('p-value of ADF is', sig)
    return time_series, i


# try to decompose lfp time-series -> lfp probably cannnot be decomposed with traditional time series decomposition because of its complexity(constantly chnaging seasonality)
def decompose_lfp(time_series:np.ndarray, period:float):
    lfp_decomposition = STL(time_series, period = period).fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax1.plot(lfp_decomposition.observed); ax1.set_ylabel('Observed')
    ax2.plot(lfp_decomposition.trend); ax2.set_ylabel('Trend')
    ax3.plot(lfp_decomposition.seasonal); ax3.set_ylabel('Seasonal')
    ax4.plot(lfp_decomposition.resid); ax4.set_ylabel('Residuals')
    plt.tight_layout()
    plt.show()


# acf and pafc plots help in selecting the optimal q, and p parameters for MA(q), AR(p), ARMA(p,q)
def plot_afc_pafc(time_series:np.ndarray):
    print('start creating acf & pafc plots')
    fig, axs = plt.subplots(nrows=2, ncols=1)
    plot_acf(time_series, ax = axs[0])
    plot_pacf(time_series, ax = axs[1])
    plt.tight_layout()
    plt.show()


def train_classic_fc_models(data, fc_method, time_lags, diff, q_lag, model_save_name='None'):
    tic = time.perf_counter()
    if fc_method == "exponential_smoothing":
        model = ExponentialSmoothing(data)
        results = model.fit(smoothing_level = 0.2, optimized = False)

    if fc_method == "autoregressive": model = SARIMAX(data, order=(time_lags,0,0), trend=None)
    if fc_method == "moving average": model = SARIMAX(data, order=(0,0,q_lag), trend=None)
    if fc_method == "arma": model = SARIMAX(data, order=(time_lags, 0, q_lag), trend=None)
    if fc_method == "arima": model = SARIMAX(data, order=(time_lags, diff, q_lag), trend= None)
    if fc_method in ["autoregressive", "moving average", "arma", "arima"]: results = model.fit(disp =False, maxiter = 10)

    if fc_method == "optimal_arima":
        results = auto_arima(data, start_P=1, start_q=1, test='adf', max_p=100, max_q=100, information_criterion='aic', max_order=5, seasonal=False)
        print(results.summary())
        p = results.arparams().size; q = results.maparams().size
        # b_coefficients = results.arparams() # these are the coefficients of the linear autoregression model
        print(f'p = {p}, q = {q}. Thus the number of parameters used for the forecsting is {p+q}')

        # trains a new arima model with the optimal_parameters (the reason is that auto_arima cant be used in out-of-sample data e.g. new time-series)
        optimal_order = results.order
        # optimal_seasonal_order = results.seasonal_order # that will be used for training a SARIMAX model (and not an ARIMA)
        model = SARIMAX(data, order=optimal_order, trend=None)
        results = model.fit(disp =False, maxiter = 100)

    toc = time.perf_counter()
    print(f'train_time = {toc-tic}')


    if model_save_name != 'None': 
        with open(PATH + 'project_files/' + model_save_name + ".pkl", "wb") as file: pickle.dump(results, file)

    return results


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# train sklearn machine learning models
def train_ml_model(ml_method, x_train, y_train, model_save_name='None'):

    ## τα 5 μοντέλα λειτουργούν με τον ακριβώς ίδιο τρόπο
    if ml_method == 'linear': model = lr.LinearRegression()
    elif ml_method == 'decision_tree': model = DecisionTreeRegressor(min_samples_split=2, max_depth= 20)
    elif ml_method == 'random_forest': model = RandomForestRegressor(max_depth= 5, n_estimators=50, max_leaf_nodes=50)
    elif ml_method == 'K_nearest_regression': model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    elif ml_method == 'dummy': model = DummyRegressor(strategy='mean')

    ## το polynomial reggression διαφέρει πολύ από τα αλλα καθώς δεν είναι χωριστό μοντέλο, αλλά linear_regression πάνω σε polynomial features
    ## οι απαιτήσεις μνήμης των polynomial features καθιστούν το μοντέλο πρακτικά άχρηστο για time series forecasting
    if ml_method == 'polynomial':
        # στην πράξη η polynomial regression είναι linear regression αφού μετατρέψουμε τα x_train σε polynomial features δηλαδή σε ένα πίνακα με στήλες για 
        # #κάθε βαθμό (x, x^2, x1*x2, x^3)
        # Τα PolynomialFeatures υπάρχουν και σε άλλες μεθόδους ML, πχ ως kernel στα SVM
        poly = PolynomialFeatures(degree = 2)
        x_poly = poly.fit_transform(x_train) # Τα PolynomialFeatures απαιτούν ΠΑΡΑ ΠΟΛΛΗ μνήμη, ακόμα και για σχετικά μικρούς βαθμούς επειδή αυξάνουν σημαντικά τα δεδομενα
        model = lr.LinearRegression()
        model.fit(x_poly, y_train)

    ## """σε αντίθεση με τα προηγούμενα μοντέλα, το SVM μπορεί να έχει μόνο έναν αριθμό σαν output και όχι ένα διάνυσμα. Με τον multi-output-regressor κάνει μια παλινδρόμιση 
    # για κάθε στοιχείο της μεταβλητής y, αλλά αυτό ίσως να μην είναι πολύ οφέλιμο σε μια χρονοσειρα"""
    if ml_method == 'support_vector_regression-muti': 
        # SVR_initial = SVR(kernel='linear', C =1)
        SVR_initial = LinearSVR(epsilon=0, C=1, max_iter=10**3)
        model = MultiOutputRegressor(SVR_initial)

    if ml_method == 'support_vector_regression-single':
        # if y_train.shape[1]==1: model = SVR(kernel='linear', C =1)
        if y_train.shape[1]==1: model = LinearSVR(epsilon=0, C=1, max_iter=10**3)
        else: print('support_vector_regression-single cannot have regression ouput more than one point, no model will be trained' )
        y_train = np.squeeze(y_train)
    
    
    model.fit(x_train, y_train)
    # print(ml_name_string + ' R^2 score is ', model.score(x_test, y_test))
    # y_pred = model.predict(x_test[0].reshape(1,-1)) # με αυτή την εντολή θα γίνει τελικά το forecasting, αλλά εδώ τεθηκε μόνο για έλεγχο
    
    if model_save_name != 'None': 
        with open(PATH + 'project_files/' + model_save_name + ".pkl", "wb") as file: pickle.dump(model, file)

    return model



## Architecture (class) of the Multi-Layer Perceptron Neural Network with a dynamic number of layers
class MLP_autoregr(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depends from the output size and the target data """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLP_autoregr, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) # without ReLU the whole MLP would only be a linear regressor
        x = self.output_layer(x)
        return x

# training loop for the above MLP regressor
def mlp_reg_training(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
        epoch_val_loss = 10^6 # initialization for the loop

        for epoch in range(num_epochs):
            train_time = []
            model.train()
            t1 = time.perf_counter()
            batch_train_losses = []
            for x_train, y_train in train_loader:
                x_train = x_train.to(device); y_train = y_train.to(device)
                train_pred = model(x_train)
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
                    x_val = x_val.to(device); y_val = y_val.to(device)
                    test_pred = model(x_val)
                    val_loss = criterion (y_val, test_pred)
                    batch_val_losses.append(val_loss.item()) # list of val_losses for every batch
                epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses) # mean of train_losses of all batches in every epoch
            epoch_str = f'Epoch:{epoch+1}/{num_epochs} -> train (batch mean) loss = {epoch_train_loss} - val (batch mean) loss = {epoch_val_loss}'
            time_str = f'train_time: {train_time[0]}'
            print(epoch_str + ' - ' + time_str)

            if  epoch_val_loss < epoch_val_loss_old: 
                torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.input_size, 'hidden_size':model.hidden_size, 'num_layers':model.num_layers, 'output_size':model.output_size}}, PATH + 'project_files/MLP_fc_regressor.pt')
                best_model = model
                best_val_score = epoch_val_loss
        
        return best_model, best_val_score


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# use Hidden Markov Models with Gaussian emissions (or GMM emissions) for time-series forecasting (gaussian or GMM emmision allow HMM to be used on continuus non-discrete 
# sequential data)
def train_hmm(data, n_components, n_mix, with_mixture_emissions, data_form, lengths, model_save_name='None'):
    if not(with_mixture_emissions): hmm_model = hmm.GaussianHMM(n_components=n_components)
    if with_mixture_emissions: hmm_model = hmm.GMMHMM(n_components=1, n_mix=n_mix)

    if data_form == 'single_lfp': hmm_model.fit(data.reshape(-1,1))
    if data_form == 'concatenated_windowed_parts': hmm_model.fit(data.reshape(-1,1), lengths)

    ## save the model
    if model_save_name!='None': 
        with open(PATH + 'project_files/'+ model_save_name + ".pkl", "wb") as file: pickle.dump(hmm_model, file)

    return hmm_model



##### other things that HMM can do (except forecasting)  -----------------------------------------
other_things = 0
model = 'the above trained hmm model'
downsampling = 1
if other_things:
    test_signal = signal_handler.time_series('WT1_2in6', downsampling)[0, :]
    prob, states = model.decode(test_signal.reshape(-1,1)) # decode() and predict() methods do the same thing, but decode also returns the log_probability of the observble state 
                                                        # being produced by the output hidden state, and allows different algorithms used for the finding of the 
                                                        # hidden_state. This is a form of unsupervised learning as hidden_states are a forms of tags assigned to the data
    print(states)
    prob = model.score(test_signal.reshape(-1,1)) # computes the log probbalility of the input being produced by the HMM model. It can be used for classification or for 
                                                # simmilarity testing, because is a metric of how close the input data is, to the data that have being used for the training
                                                # of the HMM model
    print(prob)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def generate_lfp(model, test_series, starting_point, num_gen_points, input_size, output_size, model_name, scaling_method):
    
    if starting_point<input_size:
        print ("starting_point can't be smaller than input_size. There aren't enough points to be used as input. Starting point wiil change")
        starting_point = input_size
        print(f'New starting point is {starting_point} (equal to input size)')

    lfp_segment = test_series[0, starting_point-input_size : starting_point]
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    if scaling_method!= 'None': lfp_segment  = scaler.fit_transform1d(lfp_segment)

    if model_name in ["exponential_smoothing", "autoregressive", "moving average", "arma", "arima", "optimal_arima"]:
        model = model.apply(lfp_segment) # εφαρμόζει τα νέα δεδομένα για το forecasting
        gen_signal = model.forecast(steps = num_gen_points) # κάνε forecasting
    if model_name == 'hmm':
        starting_point_value = lfp_segment[-1] # finds the observable state (in order to find the hiiden state in the next line)
        starting_point_state = model.predict(starting_point_value.reshape(-1,1)) # finds the most probbable hiden state for the value of the starting point
        gen_signal, _ = model.sample(n_samples = num_gen_points, currstate = starting_point_state[0])
        gen_signal = np.squeeze(gen_signal)
    if model_name in ["linear", "decision_tree", "random_forest", "K_nearest_regression", "dummy", "polynomial", "support_vector_regression-muti", 'support_vector_regression-single']: 
        num_predictions =  int(num_gen_points/output_size) + 1
        input_signal = lfp_segment [(len(lfp_segment)-input_size) :] # λήψη των τελευταιων σημείων του σήματος για forecasting
        input_signal = input_signal.reshape(1,-1) # πρέπει να είναι σε αυτή τη μορφή για να εισαχθεί στο predict
        gen_signal = []
        for i in np.arange(num_predictions):
            new_pred = model.predict(input_signal)
            if new_pred.shape == (1,): new_pred = new_pred.reshape(1,1) # για output = 1;;;;
            input_signal = np.hstack((input_signal[:,output_size:], new_pred))  # θα ενώσω τα δύο κομμάτια μετακινόντας το κάποιες θέσεις
            if new_pred.shape == (1,1): gen_signal.append(np.squeeze(new_pred)) # για output = 1 
            else: gen_signal = gen_signal + list(np.squeeze(new_pred))
    if model_name == 'mlp':
        model = model.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
        num_predictions =  int(num_gen_points/output_size) + 1
        input_signal = lfp_segment [(len(lfp_segment)-input_size) :] # λήψη των τελευταιων σημείων του σήματος για forecasting
        input_signal = torch.from_numpy(input_signal).float()
        gen_signal = []
        model.eval()
        with torch.no_grad():
            for i in np.arange(num_predictions):
                output = model(input_signal)
                gen_signal = gen_signal + list(output.detach().numpy()) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης
                input_signal = torch.cat((input_signal, output), dim=0)[model.output_size:] # κατασκευή νέου input για το model

    gen_signal = np.array(gen_signal)
    gen_signal =  gen_signal[:num_gen_points]
    if scaling_method!= 'None': gen_signal  = scaler.inverse1d(gen_signal)
    return gen_signal


def saveloader(method:str, saveload:str):
    if method == "exponential_smoothing": save_load_name = 'ExpSmooth_model'
    if method == "autoregressive": save_load_name = 'AR_model'
    if method == "moving average": save_load_name = 'MA_model'
    if method == "arms": save_load_name = 'ARMA_model'
    if method == "arima": save_load_name = 'ARIMA_model'
    if method == "optimal_arima": save_load_name = 'opt_ARIMA_model'
    if method == 'linear': save_load_name = 'LinearRegr_model'
    if method == 'decision_tree': save_load_name = 'DT_model'
    if method == "random_forest": save_load_name = 'RF_model'
    if method == "K_nearest_regression": save_load_name = 'KNN_model'
    if method == "dummy": save_load_name = 'Dummy_model'
    if method == "polynomial": save_load_name = 'PolynomialRegr_model'
    if method == "support_vector_regression-muti": save_load_name = 'SVR_multi_model'
    if method == 'support_vector_regression-single': save_load_name = 'SVR_single_model'
    if method == "hmm": save_load_name = 'HMM_model'

    if saveload == 'save': return save_load_name
    if saveload == 'load':
        with open(PATH + 'project_files/' + save_load_name + '.pkl', 'rb') as file: model = pickle.load(file)
        return model
    if saveload == 'None': return 'None'


def create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric):
    starting_points_list = np.array(starting_points_list)
    metric_array_list = np.zeros(len(starting_points_list))
    fs = 1/(test_series[1,3] - test_series[1,2])
    #test_signal = test_series[0,:]
    for idx, starting_point in enumerate(starting_points_list):
        actual_signal = test_series[0, starting_point : starting_point + num_gen_points]
        gen_signal = generate_lfp(model, test_series, starting_point, num_gen_points, input_size, output_size, fc_method, scaling_method)

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


def create_metric_results(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method):
    MAE_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    RMSE_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMSE')
    norm_cross_corr_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='norm-cross-corr')
    max_cross_cor_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='max Cross-cor')
    RMS_PSD_list = create_metric_list(model, fc_method, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMS-PSD')

    MAE_str = f'Absolute mean error {MAE_list.mean()}'
    RMSE_str = f'Root mean square error is {RMSE_list.mean()}'
    norm_cross_corr_str = f'Pearson r (normalized cross-correlation of zero phase) is {norm_cross_corr_list.mean()}'
    max_cross_cor_str = f'Maximum cross-correlation is {max_cross_cor_list.mean()}'
    RMS_PSD_str = f'Root mean square error of PSD is {RMS_PSD_list.mean()}'

    print(MAE_str +'\n'+ RMSE_str +'\n'+ norm_cross_corr_str +'\n'+ max_cross_cor_str +'\n'+ RMS_PSD_str)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if  __name__ == "__main__":
    fc_method_list = ["exponential_smoothing", "autoregressive", "moving average", "arma", "arima", "optimal_arima", "linear", "decision_tree", "random_forest", 
                      "K_nearest_regression", "dummy", "polynomial", "support_vector_regression-muti", 'support_vector_regression-single', 'hmm', 'mlp']
    # fc_method = fc_method_list[-1]
    fc_method = fc_method_list[4]
    data_type_for_hmm_classic_list = ['whole_series', 'combined windows']
    data_type_for_hmm_classic = data_type_for_hmm_classic_list[1]
    # fc_method = 'mlp'
    main(fc_method, data_type_for_hmm_classic)