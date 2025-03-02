
import numpy as np
import signal_handler
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
import scipy.signal as sn 
from fc_LFP import norm_cross_cor, visual_fc_comparison


remote_PC = False
if not(remote_PC): PATH = 'D:/Files/peirama_dipl/' # my PC path
if remote_PC: PATH = '/home/skoutinos/' # remote PC path

device = 'cpu'
move_to_gpu_list = ['None', 'all', 'batches'] # 'None'-> training is done in the cpu, 'all'-> all data are being moved in gpu at once, 'batches'-> data are move to gpu one batch at a time
move_to_gpu = move_to_gpu_list[2]
if move_to_gpu != 'None': device = 'cuda' if torch.cuda.is_available() else 'cpu'
if move_to_gpu != 'None': print(torch.cuda.get_device_name())

dl_model_list = ['cnn', 'cnn_lstm', 'lstm_encoder_decoder', 'lstm_encoder_decoder+attention', 'resnet_lstm']
dl_model = dl_model_list[4]

seq = 'seq2seq' # only for CNN_LSTM. It can be 'seq2one' or 'seq2seq'


def main():
     
    tag= 'All_WT_0Mg'
    downsample_scale = 10
    sliding_window_step = 10
    input_size = 500
    output_size = 100
    if dl_model in ['lstm_encoder_decoder', 'lstm_encoder_decoder+attention']: output_size = input_size

    scaling_method_list = ['min_max', 'max_abs', 'z_normalization', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaling_method = scaling_method_list[2]
    train_model = False

    
    if dl_model == 'cnn':

        num_kernels = 5
        kernel_size=int(input_size/100)
        kernel_step = 1 # kernel_step =  kernel_size
        dilation = 2
        
        # # try CNN-LSTM forward method
        # arr = torch.ones((99,1,input_size))
        # model = CNN_fc(num_kernels, kernel_size, kernel_step, dilation, input_size, output_size)
        # res = model(arr)

        model_type_input ='cnn'
        model = CNN_fc(input_size, output_size, num_kernels, kernel_size, kernel_step, dilation)



    if dl_model == 'cnn_lstm':
        num_kernels = 5
        kernel_size=int(input_size/100)
        kernel_step = 1 # kernel_step =  kernel_size
        lstm_hidden_size = 512
        num_lstm_layers = 1

        # #try CNN-LSTM forward method
        # arr = torch.ones((99,1,input_size))
        # model = CNN_LSTM_fc(input_size, output_size, lstm_hidden_size, num_kernels, kernel_size, kernel_step, num_lstm_layers)
        # res = model(arr)
        
        model_type_input ='cnn'
        model = CNN_LSTM_fc(input_size, output_size, lstm_hidden_size, num_kernels, kernel_size, kernel_step, num_lstm_layers)



    if dl_model == 'lstm_encoder_decoder':
        hidden_size = 512
        num_layers = 1
        
        # # try encoder-decoder forward method
        # arr = torch.ones((99,input_size,1))
        # model = Encoder_Decoder_fc(hidden_size, num_layers)
        # model.train()
        # res1 = model(arr)
        # print(res1.shape)
        # model.eval()
        # res2 = model(arr,32)
        # print(res2.shape)

        model_type_input ='lstm'
        model = Encoder_Decoder_fc(hidden_size, num_layers)

    if dl_model == 'lstm_encoder_decoder+attention':
        hidden_size = 128
        num_layers = 1

        # # try encoder-decoder-attention forward method
        # arr = torch.ones((99,input_size,1))
        # model = Encoder_Decoder_attention_fc(hidden_size, num_layers)
        # model.train()
        # res1 = model(arr)
        # print(res1.shape)
        # model.eval()
        # res2 = model(arr,num_gen = 32)
        # print(res2.shape)

        model_type_input ='lstm'
        model = Encoder_Decoder_attention_fc(hidden_size, num_layers)

    if dl_model == 'resnet_lstm':
        hidden_size = 512
        num_layers = 1
        bidirectional = True

        # # try residual LSTM forward method
        # arr = torch.ones((99,input_size,1))
        # model = resid_LSTM_fc(input_size, output_size, hidden_size, num_layers, bidirectional)
        # res = model(arr)
        # print(res.shape)

        model_type_input ='lstm'
        model = resid_LSTM_fc(input_size, output_size, hidden_size, num_layers, bidirectional)

    


    ### training the model
    batch_size = 128
    lr = 0.001
    num_epochs = 2
    
    
    extract_data = 0
    if train_model: train_loader, val_loader, _ = prepare_data(tag, downsample_scale, input_size, output_size, sliding_window_step, scaling_method, extract_data, batch_size, inserted_in = model_type_input)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    if train_model: model_trained, _ = train_fc_nn(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
    if not(train_model): model_trained = load_NN(dl_model, model)

    num_gen_points = 234# output_size #3 * output_size + 25 # 5 * output_size # αυτή η μεταβλητή καθορίζει πόσα σημεία θα γίνουν forecasting
    if output_size < 5: num_gen_points = 150 # για την περίπτωση που το ouput είναι πολύ μικρό
    starting_point = 322
    # test_series = signal_handler.combine (signal_handler.lists_of_names('test'), downsample_scale)
    # np.save(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy', test_series)
    test_series = np.load(PATH + 'project_files/test_series_ds'+ str(downsample_scale)  + '.npy')
    fc_signal = generate_fc_signal(model_trained, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)
    

    number_of_starting_points = 10
    index_range = np.arange(input_size, test_series.shape[1] - num_gen_points) # τα όρια είναι αυτα για τον εξής λόγο. Πριν από το starting point πρέπει να υπάρχει αρκετό input για generate, και μετά το generate πρέπει να υπάρχει αρκετή test_series για τη σύγκριση
    starting_points_list = np.random.choice(index_range, size = number_of_starting_points, replace=False)
    #MAE_list = create_metrics(model_trained, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    #print(MAE_list)
    produce_metric_means(model_trained, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method)

    plot_starting_point = np.random.randint(input_size, test_series.shape[1])
    actual_signal = test_series[0, plot_starting_point : plot_starting_point + num_gen_points]
    model_generated_signal = generate_fc_signal(model_trained, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)
    fs = 1/(test_series[1,3] - test_series[1,2])
    visual_fc_comparison(actual_signal, fs, model_generated_signal, ml_method1=dl_model, domain = 'both', save=False)

    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prepare_data(tag, downsample_scale, input_size, output_size, window_step, scaling_method, extract_data, batch_size, inserted_in = 'cnn'):
    save_load_path = PATH + 'project_files/fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy'
    if extract_data: lfp_data = signal_handler.extract_data(tag, downsample_scale, save_load_path)
    if not(extract_data): lfp_data = np.load(save_load_path)
    print('Extracted/Loaded data have shape:', lfp_data.shape) # πρόκειται για αρχεία (καταγραφές LFP) 20 λεπτων. Οι γραμμές είναι ο αριθμός των καταγραφών και οι στήλες είναι το μήκος των καταγραφών

    window_size = input_size + output_size
    if dl_model in ['lstm_encoder_decoder', 'lstm_encoder_decoder+attention']: window_size = input_size + 1
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)

    data = torch.from_numpy(lfp_data).float()
    windowed_data = data.unfold(dimension=1, size = window_size, step = window_step)
    print('torch windowed data have dimensions', windowed_data.shape)
    # windowed_data = windowed_data.contiguous()
    # windowed_data = windowed_data.view((windowed_data.shape[0]*windowed_data.shape[1],window_size,1))
    windowed_data = torch.reshape(windowed_data, (windowed_data.shape[0]*windowed_data.shape[1],window_size)) # πλέον με το reshape τα windows παύουν έτσι κι αλλιώς να είναι views
    print('torch reshaped data have dimensions', windowed_data.shape)

    if scaling_method != 'None':
        windowed_data = windowed_data.numpy()
        windowed_data  = scaler.normalize2d(windowed_data) # κανονικοποιεί τα input σειρά προς σειρά -> δηλαδή κατά παράθυρο με input και outpur όπως αναφέρεται στη βιβλιογραφία
        windowed_data = torch.from_numpy(windowed_data).float()
    
    input_data = windowed_data[:,0:input_size]
    target_data = windowed_data[:,input_size:window_size]
    if dl_model in ['lstm_encoder_decoder', 'lstm_encoder_decoder+attention']: target_data = windowed_data[:,1:input_size+1]
    
    if inserted_in == 'lstm': input_data = input_data = torch.unsqueeze(input_data, 2); target_data = torch.unsqueeze(target_data, 2)
    if inserted_in == 'cnn': input_data = input_data = torch.unsqueeze(input_data, 1); target_data = torch.unsqueeze(target_data, 1)
        
    if move_to_gpu == 'all': input_data=input_data.to(device); target_data=target_data.to(device)
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    # το γεγονός ότι γίνεται random split στα παράθυρα προκαλεί temporal leakage, και καθιστά τσ validation scores μη αξιόπιστα. Θα έπρεπε είτε τα validation παράθυρα να είναι μετά από όλα τα trainin παράθυρα, είτε να προέρχονται από μια νέα χρονοσειρά όπως γίνται στο testing
    train_data, val_data = torch.utils.data.dataset.random_split(dataset, [0.8, 0.2]); del dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=train_data.__len__()
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size) # αν θέλεις να τρέξει όλα τα δεδομένα σε ένα batch, βάλε -> batch_size=val_data.__len__()
    return train_loader, val_loader, scaler
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
### Architecture of CNN-LSTM
class CNN_LSTM_fc(nn.Module): 
    def __init__(self, input_size, output_size, lstm_hidden_size, num_kernels, kernel_size, kernel_step, num_lstm_layers):
        super(CNN_LSTM_fc, self).__init__()
        self.seq_len = input_size
        self.num_kernels = num_kernels
        self.hidden_size = lstm_hidden_size
        self.num_layers=num_lstm_layers
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.kernel_step = kernel_step
        self.cnn_out_seq = int(((input_size - kernel_size)/kernel_step) + 1)

        self.cnn = nn.Conv1d(in_channels=1, out_channels=num_kernels, kernel_size=self.kernel_size, stride=self.kernel_step, padding=0, dilation=1)
        self.lstm=nn.LSTM(num_kernels, self.hidden_size, self.num_layers, batch_first=True) # nn.LSTM has dynamic layers throught the num_layer parameter which creates stacked LSTM
        self.output_layer1 = nn.Linear(self.hidden_size, 1)
        self.output_layer2 = nn.Linear(self.cnn_out_seq, self.output_size)

    def forward(self, x):
        x = self.cnn(x)
        # out, (h_n,c_n) = self.lstm(x)
        x = torch.transpose(x, 1, 2)
        if seq == 'seq2seq': x, _ = self.lstm(x)
        if seq == 'seq2one': _ , (x,_) = self.lstm(x) # Better to not use it. It loses important temporal information
        x = self.output_layer1(x)
        x =torch.squeeze(x)
        x = self.output_layer2(x)
        return x



### Architecture of LSTM Encoder_Decoder
class Encoder_Decoder_fc(nn.Module):
    def __init__(self, hidden_size, enc_num_layers):
        super(Encoder_Decoder_fc, self).__init__()
        #self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = enc_num_layers
        #self.output_size = output_size
        #if bidirectional: self.bidirectional = 2
        #if not(bidirectional): self.bidirectional = 1

        self.encoder = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTMCell(1, self.hidden_size)
        self.decoder_lstm_linear = nn.Linear(hidden_size, 1)

    def forward(self, x, num_gen = 0):
        _ , (context_vector, _) = self.encoder(x) # it encodes the context vector in order to feed it to the decoder

        if self.training: # in training teacher forcing is being used
            output = []
            hx = torch.squeeze(context_vector)
            cx = torch.zeros(hx.shape).to(next(self.parameters()).device)
            for i in range(x.shape[1]):
                hx, cx = self.decoder_lstm(x[:,i,:], (hx, cx))
                result = self.decoder_lstm_linear(hx)
                output.append(result)
            output = torch.stack(output, dim=0)
            output = torch.transpose(output, 0, 1)
            return output
        
        else: # when it generates points it uses one point to generate the next etc.
            if num_gen==0: num_gen = x.shape[1] # if num_gen is not given it generates as many point as the input sequence length
            x = x[:, -1, :] # it takes just the last value of each batch to start generating points
            if x.shape[0]==1: x = x.squeeze(0) # if there is one batch it removes one dimension, because hx recognises it as batched input and an error emerges
            output = []
            hx = torch.squeeze(context_vector)
            cx = torch.zeros(hx.shape).to(next(self.parameters()).device)
            for i in range(num_gen):
                hx, cx = self.decoder_lstm(x, (hx, cx))
                x = self.decoder_lstm_linear(hx)
                output.append(x.clone())
            output = torch.stack(output, dim=0)
            output = torch.transpose(output, 0, 1)
            return output



### Architecture of LSTM Encoder_Decoder with simple attention mechanism
class Encoder_Decoder_attention_fc(nn.Module):
    def __init__(self, hidden_size, enc_num_layers):
        super(Encoder_Decoder_attention_fc, self).__init__()
        #self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = enc_num_layers
        #self.output_size = output_size
        #if bidirectional: self.bidirectional = 2
        #if not(bidirectional): self.bidirectional = 1

        self.encoder = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTMCell(1, self.hidden_size)
        self.decoder_lstm_linear = nn.Linear(2*hidden_size, 1)

    def forward(self, x, num_gen = 0):
        encoder_states , (first_context_vector, _) = self.encoder(x) # it encodes the context vector in order to feed it to the decoder

        if self.training: # in training teacher forcing is being used
            output = []
            hx = torch.squeeze(first_context_vector)
            cx = torch.zeros(hx.shape).to(next(self.parameters()).device)
            for i in range(x.shape[1]):
                hx, cx = self.decoder_lstm(x[:,i,:], (hx, cx))
                context_vector = self.compute_attention_contex_vector(encoder_states, hx)
                updated_vector = torch.cat((context_vector, hx), dim = 1)
                result = self.decoder_lstm_linear(updated_vector)
                output.append(result)
            output = torch.stack(output, dim=0)
            output = torch.transpose(output, 0, 1)
            return output
        
        else: # when it generates points it uses one point to generate the next etc.
            if num_gen==0: num_gen = x.shape[1] # if num_gen is not given it generates as many point as the input sequence length
            x = x[:, -1, :] # it takes just the last value of each batch to start generating points
            output = []
            hx = first_context_vector.squeeze(0)
            cx = torch.zeros(hx.shape).to(next(self.parameters()).device)
            for i in range(num_gen):
                hx, cx = self.decoder_lstm(x, (hx, cx))
                context_vector = self.compute_attention_contex_vector(encoder_states, hx)
                updated_vector = torch.cat((context_vector, hx), dim = 1)
                x = self.decoder_lstm_linear(updated_vector)
                output.append(x.clone())
            output = torch.stack(output, dim=0)
            output = torch.transpose(output, 0, 1)
            return output

    def compute_attention_contex_vector(self, encoder_states, query_vector):
        # scores = torch.einsum("ijk,ik->ij", encoder_state, hx) # ΔΕΝ ΕΙΣΑΙ ΣΙΓΟΥΡΟΣ ΟΤΙ ΕΧΕΙ ΕΚΤΕΛΕΣΤΕΙ ΚΑΛΑ ΤΟ DOT PRODUCT !!!
        mul_for_scores = torch.transpose(encoder_states, 0, 1) * torch.unsqueeze(query_vector, 0)
        scores = torch.transpose(mul_for_scores.sum(dim=-1), 0, 1)
        attention_weights = F.softmax(scores, dim = 1)
        # context_vector = torch.einsum("ijk,ik->ik", encoder_state, attention_weights) # ΔΕΝ ΕΙΣΑΙ ΣΙΓΟΥΡΟΣ ΟΤΙ ΕΧΕΙ ΕΚΤΕΛΕΣΤΕΙ ΚΑΛΑ ΤΟ DOT PRODUCT !!!
        mul_for_context = encoder_states * torch.unsqueeze(attention_weights, 2)
        context_vector = mul_for_context.sum(dim=1)
        return context_vector
    


    ### Architecture of CNN
class CNN_fc(nn.Module):
    def __init__(self, input_size, output_size, num_kernels, kernel_size, kernel_step, dilation):
        super(CNN_fc, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.kernel_step = kernel_step
        self.dilation = dilation

        self.len1 = ((input_size - dilation*(kernel_size-1)-1)/kernel_step) +1
        self.len2 = ((input_size - int(self.dilation/2)*(2*kernel_size-1)-1)/kernel_step) +1
        #step1 = (min(self.len1,self.len2)-2*num_kernels - 2)
        #step2 = ((min(self.len1,self.len2)-2*num_kernels - 2)/(2*self.num_kernels))
        self.width_result = int(np.ceil(((min(self.len1,self.len2)-2*num_kernels - 2)/(2*self.num_kernels)) + 1))
        
        self.conv1d_short = nn.Conv1d(in_channels=1, out_channels=self.num_kernels, kernel_size=self.kernel_size, stride=self.kernel_step, padding=0, dilation=self.dilation)
        self.conv1d_long = nn.Conv1d(in_channels=1, out_channels=self.num_kernels, kernel_size=2*self.kernel_size, stride=self.kernel_step, padding=0, dilation=int(self.dilation/2))
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = 2*self.num_kernels, stride=2*self.num_kernels, padding=0, dilation=1)
        self.pool = nn.AvgPool2d(kernel_size=6)
        self.output = nn.Linear(self.width_result, output_size)

    def forward(self,x):
        x1 = self.conv1d_short(x)
        x2 = self.conv1d_long(x)
        min_dim = min(x1.shape[2], x2.shape[2])
        x = torch.cat((x1[:,:,:min_dim], x2[:,:,:min_dim]), dim = 1)
        x = torch.unsqueeze(x, 1)
        # x = self.pool(x)
        x = self.conv2d(x)
        x = self.output(x)
        x = torch.squeeze(x)
        return x


### Architecture (class) of the LSTM-based-neural-network
class resid_LSTM_fc(nn.Module): 
    """this model will be a forecasting LSTM model that takes 100 (or more) points and finds some points in the future. 
    How many are the 'some' points depends from the output size and the target data """
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional):
        super(resid_LSTM_fc, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        if bidirectional: self.bidirectional = 2
        if not(bidirectional): self.bidirectional = 1

        self.autoregr = nn.Linear(self.input_size, self.output_size)
        self.lstm=nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional) # nn.LSTM has dynamic layers throught the num_layer parameter which creates stacked LSTM
        self.linear1 = nn.Linear(self.bidirectional*self.hidden_size, 1) # combines straight and inverse h_t's
        self.linear2 = nn.Linear(self.input_size, self.output_size) # combines the points from time steps into final output

    def forward(self, x):
        #if self.training:
        x_init = torch.squeeze(x)
        x_init = self.autoregr(x_init)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = torch.squeeze(x)
        x = self.linear2(x)
        return x + x_init
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_fc_nn(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
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
                if dl_model == 'cnn': torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.input_size, 'output_size':model.output_size,
                                                                                                            'num_kernels':model.num_kernels, 'kernel_size':model.kernel_size, 
                                                                                                            'kernel_step':model.kernel_step, 'dilation':model.dilation}}, 
                                                                                                            PATH + 'project_files/cnn_fc.pt')
                if dl_model == 'cnn_lstm': torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.seq_len, 'hidden_size':model.num_kernels, 
                                                                                                            'num_layers':model.num_layers, 'output_size':model.output_size, 
                                                                                                            'kernel_size':model.kernel_size, 'kernel_step':model.kernel_step, 
                                                                                                            'cnn_out_seq':model.cnn_out_seq}}, PATH + 'project_files/cnn_lstm_fc.pt')
                if dl_model == 'lstm_encoder_decoder': torch.save({'model_state_dict':model.state_dict(), 'model_args':{'hidden_size':model.hidden_size, 
                                                                                                                        'num_layers':model.num_layers, }}, PATH + 'project_files/lstm_enc_dec_fc.pt')
                if dl_model == 'lstm_encoder_decoder+attention': torch.save({'model_state_dict':model.state_dict(), 'model_args':{'hidden_size':model.hidden_size, 
                                                                                                                                  'num_layers':model.num_layers}}, 
                                                                                                                                  PATH + 'project_files/lstm_enc_dec_attention_fc.pt')
                if dl_model == 'resnet_lstm': torch.save({'model_state_dict':model.state_dict(), 'model_args':{'input_size':model.input_size, 'output_size':model.output_size, 
                                                                                                                 'hidden_size':model.hidden_size, 'bidirectional':model.bidirectional,
                                                                                                                                  'num_layers':model.num_layers, }}, 
                                                                                                                                  PATH + 'project_files/resnet_lstm_fc.pt')
                best_model = model
                best_val_score = epoch_val_loss
        
        return best_model, best_val_score

def load_NN(dl_model, initialized_model):
    load_name_dict = {'cnn':'cnn_fc', 'cnn_lstm':'cnn_lstm_fc', 'lstm_encoder_decoder':'lstm_enc_dec_fc', 'lstm_encoder_decoder+attention':'lstm_enc_dec_attention_fc', 'resnet_lstm':'resnet_lstm_fc'}  
    checkpoint = torch.load(PATH + 'project_files/models/' + load_name_dict[dl_model] +'.pt')
    # args = checkpoint['model_args']
    # if dl_model == 'cnn':  model = CNN_fc(**args)
    # if dl_model == 'cnn_lstm':  model = CNN_LSTM_fc(**args)
    # if dl_model == 'lstm_encoder_decoder':  model = Encoder_Decoder_fc(**args)
    # if dl_model == 'lstm_encoder_decoder+attention':  model = Encoder_Decoder_attention_fc(**args)
    # if dl_model == 'resnet_lstm':  model = resid_LSTM_fc(**args)
    initialized_model.load_state_dict(checkpoint['model_state_dict'])
    initialized_model.eval()
    return initialized_model


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_fc_signal(model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method):
    
    if starting_point<input_size:
        print ("starting_point can be smaller than input_size. There aren't enough points to be used as input. Starting point wiil change")
        starting_point = input_size
        print(f'New starting point is {starting_point} (equal to input size)')

    lfp_segment = test_series[0, starting_point-input_size : starting_point]
    scaler = signal_handler.lfp_scaler(scaling_method, scaling_power=4)
    if scaling_method!= 'None': lfp_segment  = scaler.fit_transform1d(lfp_segment)

    model = model.to('cpu') # στέλνει το μοντέλο στη cpu για να γίνει η παραγωγή σήματος. Δε χρειάζεται η gpu για αυτό το task.
    num_predictions =  int(num_gen_points/output_size) + 1
    input_signal = lfp_segment [(len(lfp_segment)-input_size) :] # λήψη των τελευταιων σημείων του σήματος για forecasting
    input_signal = torch.from_numpy(input_signal).float()
    gen_signal = []
    model.eval()
    if dl_model in ['cnn', 'cnn_lstm']:
        with torch.no_grad():
            for i in np.arange(num_predictions):
                input_signal = torch.unsqueeze(input_signal, 0)
                input_signal = torch.unsqueeze(input_signal, 0)
                output = model(input_signal)
                input_signal = torch.squeeze(input_signal)
                output = torch.squeeze(output)
                gen_signal = gen_signal + list(output.detach().numpy()) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης
                input_signal = torch.cat((input_signal, output), dim=0)[model.output_size:] # κατασκευή νέου input για το model
    if dl_model == 'resnet_lstm':
        with torch.no_grad():
            for i in np.arange(num_predictions):
                input_signal = torch.unsqueeze(input_signal, 1)
                input_signal = torch.unsqueeze(input_signal, 0)
                output = model(input_signal)
                input_signal = torch.squeeze(input_signal)
                output = torch.squeeze(output)
                gen_signal = gen_signal + list(output.detach().numpy()) # εδώ επεκτείνεται η λίστα generated_signal, που θα είναι το τελικό output της συνάρτησης
                input_signal = torch.cat((input_signal, output), dim=0)[model.output_size:] # κατασκευή νέου input για το model
    if dl_model in ['lstm_encoder_decoder', 'lstm_encoder_decoder+attention']:
        input_signal = torch.unsqueeze(input_signal, 1)
        input_signal = torch.unsqueeze(input_signal, 0)
        gen_signal = model(input_signal, num_gen=num_gen_points)
        gen_signal = torch.squeeze(gen_signal)
        gen_signal = gen_signal.detach().numpy()

    gen_signal = np.array(gen_signal)
    gen_signal =  gen_signal[:num_gen_points]
    if scaling_method!= 'None': gen_signal  = scaler.inverse1d(gen_signal)
    return gen_signal


def create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric):
    starting_points_list = np.array(starting_points_list)
    metric_array_list = np.zeros(len(starting_points_list))
    fs = 1/(test_series[1,3] - test_series[1,2])
    #test_signal = test_series[0,:]
    for idx, starting_point in enumerate(starting_points_list):
        actual_signal = test_series[0, starting_point : starting_point + num_gen_points]
        gen_signal = generate_fc_signal(model, test_series, starting_point, num_gen_points, input_size, output_size, scaling_method)

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


def produce_metric_means(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method):
    MAE_list = create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='MAE')
    RMSE_list = create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMSE')
    norm_cross_corr_list = create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='norm-cross-corr')
    max_cross_cor_list = create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='max Cross-cor')
    RMS_PSD_list = create_metrics(model, test_series, starting_points_list, num_gen_points, input_size, output_size, scaling_method, metric ='RMS-PSD')

    MAE_str = f'Absolute mean error {MAE_list.mean()}'
    RMSE_str = f'Root mean square error is {RMSE_list.mean()}'
    norm_cross_corr_str = f'Pearson r (normalized cross-correlation of zero phase) is {norm_cross_corr_list.mean()}'
    max_cross_cor_str = f'Maximum cross-correlation is {max_cross_cor_list.mean()}'
    RMS_PSD_str = f'Root mean square error of PSD is {RMS_PSD_list.mean()}'

    print(MAE_str +'\n'+ RMSE_str +'\n'+ norm_cross_corr_str +'\n'+ max_cross_cor_str +'\n'+ RMS_PSD_str)


if  __name__ == "__main__":
    main()

