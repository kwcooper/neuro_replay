# Keiland's edits 191028

import numpy as np
import statsmodels.api as sm
import statsmodels as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from data import *
from theta import *
import time


def analysis_flow(dataPaths, rat_name, theta_thres=20, corr_thres=0.9, verbose=False):
    """
    Perform analysis workflow for a rat and exclude trials during which theta amplitude is too low.
    """
    trial_info, spike_data, lfp_data = load_data(dataPaths, rat_name)
    target, spike_data, lfp_data = clean_data(trial_info, spike_data, lfp_data)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference = lfp_data[:, tetrode_index, :]  # get reference LFP
    # (n: trials by d: cells )
    training_features = process_spike(spike_data, lfp_reference)  # align spike train with LFP

    scaler = StandardScaler()
    training = scaler.fit_transform(training_features)
    
    amp_dist = amplitude_distribution(lfp_reference)
    select_trial = amp_dist > np.percentile(amp_dist, theta_thres)  # select trials by amplitude
    #linear_circular_corr = correlate_time_phase(spike_data, lfp_reference)
    #select_cell = linear_circular_corr > corr_thres  # select cells by correlation
    
    #training_select = training[select_trial][:, select_cell]
    training_select = training[select_trial]
    target_select = target[select_trial]

    
    # train model!
    model = LogisticRegressionCV(cv=5, multi_class='multinomial', 
                                 penalty='l1', solver='saga', 
                                 class_weight='balanced', max_iter=500)
    model = model.fit(training_select, target_select)
    
    if verbose:
        print(np.mean(model.predict(training_select) == target_select))
    # Q: Why 37? 
    nn = 37 # moved to var, origionally 37 | (134/30, 37, 4)
    rolling_preds = np.zeros((target.shape[0], nn, 4))  # rolling prediction during theta cycle
    for i in range(nn):  
        rolling_features = process_spike(spike_data, lfp_reference, [300 + i * 10, 420 + i * 10])
        decoding = scaler.transform(rolling_features)
        #decoding_preds = model.predict_proba(decoding[:, select_cell])
        decoding_preds = model.predict_proba(decoding)
        rolling_preds[:, i, :] = decoding_preds
    
    phase_preds = []
    for i in range(3):
        current_features = process_spike(spike_data, lfp_reference, [360 + i * 120, 480 + i * 120])
        decoding = scaler.transform(current_features)
        #current_preds = model.predict_proba(decoding[:, select_cell])
        current_preds = model.predict_proba(decoding)
        phase_preds.append(current_preds[select_trial])
    
    return rolling_preds[select_trial], phase_preds, target_select

@ignore_warnings(category=ConvergenceWarning)
def analysis_flow_LOO(dataPaths, rat_name, theta_thres=20, corr_thres=0.9, verbose=False):
    """
    Perform analysis workflow for a rat while leaving out one or a subset of neurons
    and exclude trials during which theta amplitude is too low.
    """
    trial_info, spike_data, lfp_data = load_data(dataPaths, rat_name)
    target, spike_data, lfp_data = clean_data(trial_info, spike_data, lfp_data)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference = lfp_data[:, tetrode_index, :]  # get reference LFP
    # (n: trials by d: cells )
    training_features = process_spike(spike_data, lfp_reference)  # align spike train with LFP

    loo_dict = {}
    num_cells = training_features.shape[1]
    start = time.time()
    timeList = []
    for cell_idx in range(0,num_cells):

        cycleStart = time.time()
        #if verbose: 
        if cell_idx % 10 == 0:
            print('On cell %s of %s.'%(cell_idx, num_cells))
    
        # Leave one out (LOO) logic -> cell_idx
        trainf_loo = np.delete(training_features, cell_idx, axis=1)
        
        scaler = StandardScaler()
        training = scaler.fit_transform(trainf_loo)

        # Only keep theta trials above a reasonable aplitude
        amp_dist = amplitude_distribution(lfp_reference)
        select_trial = amp_dist > np.percentile(amp_dist, theta_thres)  # select trials by amplitude
        
        training_select = training[select_trial]
        target_select = target[select_trial]


        
        # train da model!
        # doesn't converge @ 500 iter... if time should expand...
        model = LogisticRegressionCV(cv=5, multi_class='multinomial', 
                                     penalty='l1', solver='saga', 
                                     class_weight='balanced', max_iter=500)
        model = model.fit(training_select, target_select)
        
        nn = 37 # moved to var, origionally 37 | (134/30, 37, 4)
        rolling_preds = np.zeros((target.shape[0], nn, 4))  # rolling prediction during theta cycle
        for i in range(nn):  
            rolling_features = process_spike(spike_data, lfp_reference, [300 + i * 10, 420 + i * 10])
            rf_loo = np.delete(rolling_features, cell_idx, axis=1) # LOO
            decoding = scaler.transform(rf_loo)
            decoding_preds = model.predict_proba(decoding)
            rolling_preds[:, i, :] = decoding_preds
        
        phase_preds = []
        for i in range(3):
            current_features = process_spike(spike_data, lfp_reference, [360 + i * 120, 480 + i * 120])
            cf_loo = np.delete(current_features, cell_idx, axis=1) # LOO
            decoding = scaler.transform(cf_loo)
            current_preds = model.predict_proba(decoding)
            phase_preds.append(current_preds[select_trial])

        # Save our hard work
        loo_dict[cell_idx] = {'rp':rolling_preds[select_trial], 'pp':phase_preds, 't':target_select}
        timeList.append(time.time() - cycleStart)

    totalTime = time.time() - start
    return loo_dict, totalTime

# Function to parse the last theta cycle before a poke out event. 
def analysis_flow_lastTrial(dataPaths, rat_name, theta_thres=20, corr_thres=0.9, verbose=False):
    """
    Perform analysis workflow for a rat and exclude trials during which theta amplitude is too low.
    """

    # Grab the data from the poke in trial
    trial_info_pi, spike_data_pi, lfp_data_pi = load_data2(dataPaths, rat_name, 'PokeIn') 
    target_pi, spike_data_pi, lfp_data_pi = clean_data3(trial_info_pi, spike_data_pi, lfp_data_pi)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference_pi = lfp_data_pi[:, tetrode_index, :]  # get reference LFP
    # (n: trials by d: cells )
    #training_features_pi = process_spike(spike_data_pi, lfp_reference_pi)  # align spike train with LFP
    training_features_pi = process_spike2(spike_data_pi, lfp_reference_pi, 'PokeIn', [480, 600])  # align spike train with LFP
    
    # Grab the data from the poke out trial
    trial_info_po, spike_data_po, lfp_data_po = load_data2(dataPaths, rat_name, 'PokeOut') 
    target_po, spike_data_po, lfp_data_po = clean_data3(trial_info_po, spike_data_po, lfp_data_po)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference_po = lfp_data_po[:, tetrode_index, :]  # get reference LFP
    # (n: trials by d: cells )
    #training_features_po = process_spike(spike_data_po, lfp_reference_po)  # align spike train with LFP
    # TODO: Clean up these LFP Indicies
    training_features_po = process_spike2(spike_data_po, lfp_reference_po, 'PokeOut', [840, 960])  # align spike train with LFP
    
    print('Train shape:', training_features_pi.shape)
    scaler = StandardScaler()
    training = scaler.fit_transform(training_features_pi)
    print('scaled shape:', training.shape)

    # ? Should I do this for the poke out trials or both or...
    amp_dist_pi = amplitude_distribution(lfp_reference_pi)
    select_trial = amp_dist_pi > np.percentile(amp_dist_pi, theta_thres)  # select trials by amplitude
    
    training_select = training[select_trial]
    target_select = target_pi[select_trial]

    
    # train model!
    # ? Should I train the model on the same data (pi theta) to predict po theta? Probably...
    model = LogisticRegressionCV(cv=5, multi_class='multinomial', 
                                 penalty='l1', solver='saga', 
                                 class_weight='balanced', max_iter=500)
    model = model.fit(training_select, target_select)
    
    if verbose:
        print(np.mean(model.predict(training_select) == target_select))

    # Predict the rest of the cycle
    # Q: Why 37?
    print('spike_data_po', spike_data_po.shape)
    print('spike_data_pi', spike_data_pi.shape)
    rolling_preds = np.zeros((target_po.shape[0], 37, 4))  # rolling prediction during theta cycle
    for i in range(37):
        sd = spike_data_po # decode the po data
        rolling_features = process_spike(sd, lfp_reference_po, [300 + i * 10, 420 + i * 10])
        decoding = scaler.transform(rolling_features)
        decoding_preds = model.predict_proba(decoding)
        rolling_preds[:, i, :] = decoding_preds

    phase_preds = []
    for i in range(3):
        sd = spike_data_po # decode the po data
        current_features = process_spike(sd, lfp_reference_po, [360 + i * 120, 480 + i * 120])
        decoding = scaler.transform(current_features)
        current_preds = model.predict_proba(decoding)
        phase_preds.append(current_preds[select_trial])
    
    return rolling_preds[select_trial], phase_preds, target_select



# This guy should give us the extended theta cycle analysis TD:#@ignore_warnings(category=FutureWarning)?
@ignore_warnings(category=ConvergenceWarning)
def multiple_theta_flow(dataPaths, rat_name):
    trial_info, spike_data, lfp_data = load_data(dataPaths, rat_name)
    target, spike_data, lfp_data = clean_data(trial_info, spike_data, lfp_data)
    tetrode_index = reference_tetrode(rat_name)
    lfp_reference = lfp_data[:, tetrode_index, :]  # get reference LFP
   
    training_features = process_spike(spike_data, lfp_reference)  # align spike train with LFP
    scaler = StandardScaler()
    training = scaler.fit_transform(training_features)

    theta_thres = 20
    amp_dist = amplitude_distribution(lfp_reference)
    select_trial = amp_dist > np.percentile(amp_dist, theta_thres)  # select trials by amplitude

    training_select = training[select_trial]
    target_select = target[select_trial]
    # edited from LogisticRegression to LogisticRegressionCV
    model = LogisticRegressionCV(multi_class='multinomial', solver='saga', penalty='l1', class_weight='balanced')
    model = model.fit(training_select, target_select)
   
    rolling_preds = np.zeros((target.shape[0], 200, 4))  # rolling prediction during theta cycle
    for i in range(200):  
        rolling_features = process_spike(spike_data, lfp_reference, [300 + i * 10, 420 + i * 10])
        decoding = scaler.transform(rolling_features)
        decoding_preds = model.predict_proba(decoding)
        rolling_preds[:, i, :] = decoding_preds
       
    return rolling_preds, target


def functional_central_curves(curves):
    """
    Find the functional median curve and the central 50% range.
    """
    depth = sm.graphics.functional.banddepth(curves, method='MBD')
    ix_depth = np.argsort(depth)[::-1]
    #res = sm.graphics.fboxplot(curves)
    #ix_depth = res[2]
    ix_median = ix_depth[0]
    median_curve = curves[ix_median, :]
    ix_central = ix_depth[:int(0.5 * ix_depth.shape[0])]
    central_curves = curves[ix_central, :]
    central_min = np.min(central_curves, axis=0)
    central_max = np.max(central_curves, axis=0)
    central_curves = np.stack([median_curve, central_min, central_max])
    return central_curves


def central_curves(curves, use_median=False):
    """
    Calculate usual central curves.
    """
    n = curves.shape[0]
    error = np.std(curves, axis=0) / np.sqrt(n)
    if use_median:
        median_curve = np.percentile(curves, 50, axis=0)
        #central_min = np.percentile(curves, 25, axis=0)
        #central_max = np.percentile(curves, 75, axis=0)
        central_min = median_curve - 2 * error
        central_max = median_curve + 2 * error
        central_curves = np.stack([median_curve, central_min, central_max])
    else:
        mean_curve = np.mean(curves, axis=0)
        central_min = mean_curve - 2 * error
        central_max = mean_curve + 2 * error
        central_curves = np.stack([mean_curve, central_min, central_max])
    return central_curves
