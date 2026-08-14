# dataset
from trackml.dataset import load_event

# SNN
import torch

# plotting
import matplotlib.pyplot as plt
import scienceplots

# other
import os
import numpy as np
import pandas as pd

# local
import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
from Functions import preprocess_particles, spike_input, SNN_LEAKY, give_weights, plot_cur_mem_spk, plot_hits  # type: ignore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"] = False


event_prefix = "event000001000"
data_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
hits, cells, particles, truth = load_event(os.path.join(data_dir, event_prefix))

# ============= GLOBAL   ======================================================
f_lhc = 40e6
omega = 2 * np.pi * f_lhc
time = 2 * np.pi / omega


#  =============================================================================
#  ================ To be passed in ============================================

thresholds = np.linspace(0,6, 10)    # iterate different thresholds
n_step = (int(time * 1e9) + 5) * 90 //20   #start med 135

# might want to add later to iterate the number of steps. 
base = (int(time * 1e9) + 5)  
n_steps_list = [base, base*3, base*6, base*27, base*90]  # 30, 90, 180, 810, 2700


def efficiency_throwrate(threshold): 
        
    # Lägg till r, phi och weights
    hits["r"] = np.hypot(hits["x"], hits["y"])
    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    all_hits = give_weights(hits, truth)                # everything gets weights * 10000


    #The amount of layers
    layer_radii       = hits.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
    layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
    n_afferents_full  = len(layer_to_afferent)


    #take all the good ones / and bad ones
    particles_conditioned, hits_conditioned, _ = preprocess_particles(10, 10, particles, hits)     
    good = all_hits[truth["particle_id"].isin(particles_conditioned["particle_id"])].copy()
    bad = all_hits[~truth["particle_id"].isin(particles_conditioned["particle_id"])].copy()
    bad["weights"] = bad["weights"] * (6/10) #decrese them some


    # good spike input
    spike_matrix_a, spike_row_a, hits_out_a, t_idx = spike_input(good, n_afferents_full)
    # mem_rec_a, spk_rec_a = SNN_LEAKY(spike_row_a, threshold=2)

    # bad spike input
    spike_matrix_b, spike_row_b, hits_out_b, t_idx = spike_input(bad, n_afferents_full)
    mem_rec_b, spk_rec_b = SNN_LEAKY(spike_row_b, threshold=threshold)

    #togheter
    spike_matrix_ba = spike_matrix_b + spike_matrix_a
    spike_row_ba = spike_row_b + spike_row_a

    mem_rec_ba, spk_rec_ba = SNN_LEAKY(spike_row_ba, threshold=threshold)
    # plot_spike_matrix(spike_matrix_ba)
    plot_cur_mem_spk(spike_row_ba.flatten(), mem_rec_ba, spk_rec_ba, thr_line=threshold)


    hits_out = pd.concat([hits_out_a, hits_out_b], ignore_index=True)
    good_times        = torch.nonzero(spk_rec_ba, as_tuple=True)[0].numpy()
    hits_out["t_idx"] = (hits_out["t"] / time * n_step).round().astype(int)
    good_hits         = hits_out[hits_out["t_idx"].isin(good_times)]

    volumes = [8, 13, 17]
    Guess   = good_hits[good_hits["volume_id"].isin(volumes)]                           #then my guess is out of all the good hits, how many are in volumes 81317


    # Ground truth — true signal hits in volumes 8, 13, 17
    interesting_hits            = hits_out[hits_out["hit_id"].isin(good["hit_id"])]                               #all the actuall good hits
    true_amount                 = interesting_hits[interesting_hits["volume_id"].isin(volumes)]                     #that are in v81317

    # Score
    My_score  = Guess[Guess["hit_id"].isin(true_amount["hit_id"])]
    print(f"out of the 75 correct i guessed: {len(My_score)}")
    efficency = len(My_score)/true_amount.shape[0]

    total     = hits[hits["volume_id"].isin(volumes)].shape[0]
    saved     = Guess.shape[0]
    throwrate = (total - saved) / total

    return efficency, total, saved, throwrate, Guess
    

eff, total, saved, throwrate, guess=  efficiency_throwrate(2)


#this is for plotting the function for one threshold of 5    
eff, total, saved, throwrate, guess = efficiency_throwrate(5.5)
plot_hits(guess)

#now to find the efficiency vs throwrate we can: 
plt.figure(figsize=(10, 6))
efficiencies, throwrates = [], []   

for thr in thresholds:
    eff, total, saved, throwrate, guess = efficiency_throwrate(thr)
    efficiencies.append(eff)
    throwrates.append(throwrate)

plt.plot(throwrates, efficiencies, marker='o', label=f"n_steps={n_step}")

plt.xlabel("Throwrate")
plt.ylabel("Efficiency")
plt.grid(True)
plt.show()



