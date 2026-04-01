#Track Ml datasets necessities
from trackml.dataset import load_event    # These are libraries provided by the organizers to load the data and compute the score.
from trackml.randomize import shuffle_hits
from trackml.score import score_event
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# imports SNN
import snntorch as snn
from snntorch import spikeplot as splt
import torch

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
import seaborn as sns

# Other imports
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import pandas as pd


event_prefix = "event000001000"
data_dir = r"C:\Users\felix\Downloads\Skola\KAND\Kand\ML Challange\Data\train_100_events"
hits, cells, particles, truth = load_event(os.path.join(data_dir, event_prefix))


print(f"hits shape {hits.shape}")



"""
mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
#print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))

"""

print()
print("\n \n Event:", event_prefix, "Alla viktiga shapes: \n")
print("Loaded hits:", hits.shape)
print("Loaded cells:", cells.shape)
print("Loaded particles:", particles.shape)
print("Loaded truth:", truth.shape)
print()


# ================ VISUALIZATION 2D - all hits ==================
g = sns.jointplot(data=hits, x="x", y="y")
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.x, v.y, s=2, label='volume {}'.format(volume))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.close()



# ================ PREPROCESSING FUNKTION ========================
#global parameters 
def preprocess_particles(p_limit, n_hits, particles, hits):
    " Filtrera bort partiklar under p_limit GeV, under n_hits, ger ut en partikellista och en intressant hit lista"

    # Filtrera bort partiklar med p under p_limit
    px = particles['px']
    py = particles['py']
    pz = particles['pz']
    p = np.sqrt(px**2 + py**2 + pz**2)
    particles['p'] = p
    high_energy_particles = particles[p > p_limit]

    # Filtrera bort partiklar med färre än n_hits
    hit = high_energy_particles['nhits']
    high_energy_particles = high_energy_particles[hit > n_hits]

    # Filter hits för de intressanta partiklarna och volymerna
    #interesting_volumes =[8, 13, 17] 
    #hits = hits[hits['volume_id'].isin(interesting_volumes)].copy()

    # Lägg till r och phi i hits
    hits["r"] = np.hypot(hits["x"], hits["y"])
    hits["phi"] = np.arctan2(hits["y"], hits["x"])

    #hits_interesting är alla hits som tillhör en signalbana
    valid_pids = high_energy_particles["particle_id"]                
    valid_hit_ids = truth[truth["particle_id"].isin(valid_pids)]["hit_id"]                                                                                                                                                                       
    hits_interesting = hits[hits["hit_id"].isin(valid_hit_ids)]

    return high_energy_particles.sort_values("p", ascending=False), hits_interesting


def choice_particels(hits_in, n, truth, particle): 
  "Denna funktion tar nte högsta energiska partikelbana och ger specifkt hits kopplade till den, ger ut "
  
  test_particle = particle.loc[n]    # loc = index, iloc = position                                                                                                                           
  pid = test_particle["particle_id"]                                                                                                                             
  hit_ids = truth.loc[truth["particle_id"] == pid, "hit_id"] 
  hits_in["weights"] = truth[truth["particle_id"] == pid]["weight"]*10000
  
  return hits_in.loc[hits_in["hit_id"].isin(hit_ids)].copy(), pid


# ================ PLOTTING FUNCTIONS =====================================
def plot_aff_time(id, hits):
  " this function can plot the afferent vs time for a given particle id, must give aff + t hits in"
                                                                                                                                                                   
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(hits["t"] * 1e9, hits["a"], s=100, color="red", zorder=5)
  ax.set_xlabel("Spike time t (ns)")
  ax.set_ylabel("Afferent a (layer)")
  ax.set_yticks(sorted(hits["a"].unique()))
  ax.set_title(f"SNN Spike Encoding — Particle {[id]}")
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_cur_mem_spk(cur, mem, spk, thr_line=0.5, vline=False, title=False, ylim_max1=1.5, ylim_max2=1.5):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, 280])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  if title:
    ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2]) 
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  if thr_line:
    ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([]) 

  plt.show()


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, num_steps, title):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,7), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input spikes
  splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_title(title)

  # Plot hidden layer spikes
  splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
  ax[1].set_ylabel("Hidden Layer")

  # Plot output spikes
  splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
  ax[2].set_ylabel("Output Spikes")
  ax[2].set_ylim([0, 10])

  plt.show()


def dvs_animator(spike_data):
  fig, ax = plt.subplots()
  anim = splt.animator((spike_data[:,0] + spike_data[:,1]), fig, ax)
  return anim


def histogram_of_particle_momenta(particles, mean):
    plt.figure(figsize=(8, 5))
    plt.hist(particles['p'], bins=50, color='blue', alpha=0.7)
    plt.axvline(x=mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.1f}')
    plt.xlabel('Momentum (GeV/c)')
    plt.ylabel('Number of Particles')
    plt.title('Histogram of Particle Momenta')
    plt.grid(True)
    plt.show()


def histogram_of_particle_hits(particles):
    plt.figure(figsize=(8, 5))
    plt.hist(particles['nhits'], bins=50, color='blue', alpha=0.7)
    plt.xlabel('Number of Hits')
    plt.ylabel('Number of Particles')
    plt.title('Histogram of Number of hits')
    plt.grid(True)
    plt.show()


def plot_particle_hits(particle_hits, h=hits):
    id = particle_hits["hit_id"].values
    plt.figure(figsize=(6, 6))
    plt.scatter(h["x"], h["y"], s=1, alpha=0.2, color='blue')
    plt.scatter(particle_hits["x"], particle_hits["y"], s=7, alpha=0.6, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hits in x-y plane, partikel nr {}'.format(id))
    plt.close()


def plot_hits(h = hits): 
    plt.figure(figsize=(6, 6))
    plt.scatter(h["x"], h["y"], s=1, alpha=0.2, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Hits fom {h}")
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000) 
    plt.show()
   
   


# ================ SNN DEFINITION =====================================
particles_conditioned, hits_conditioned = preprocess_particles(10, 10, particles, hits)         #take conditioned particles and hits
first_particle_ind = particles_conditioned.index                                                #look at indeces if particles
n = first_particle_ind[2]                                                                       #pick index of most energetic and it's hits associated
signal, particle_id = choice_particels(hits_conditioned, n, truth, particles_conditioned)       #pick the hits and turn it into a signal with that specifik hit_ids




layer_radii_full = hits.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
layer_to_afferent_full = {layer: i for i, layer in enumerate(layer_radii_full.index)}
n_afferents_full = len(layer_to_afferent_full)

print(n_afferents_full)                                                                         #understanding how many afferents there are



f_lhc = 40e6               #frequency
omega = 2 * np.pi * f_lhc  #vinkelfrekvens
time = 2*np.pi / omega     # full scan time
#n_afferents_full = 48
#n_steps_full = int(time * 1e9) + 2
      

phi_pos = signal["phi"].values + 2 * np.pi * (signal["phi"].values < 0) 
signal["t"] = phi_pos / omega                                                                    #At what time does the hit occur.                                                                 

layer_radii       = signal.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
n_afferents       = len(layer_to_afferent)           

signal["a"] = pd.MultiIndex.from_frame(
          signal[["volume_id", "layer_id"]]
      ).map(layer_to_afferent)                                                                                         

# wrap-around
aux = signal[phi_pos <= 0.7].copy()
aux["t"] = (phi_pos[phi_pos <= 0.7] + 2 * np.pi) / omega
hits_out = pd.concat([signal, aux], ignore_index=True)

t_bins  = (hits_out["t"].values * 1e9).astype(int)
a_idx   = hits_out["a"].values.astype(int)
n_steps = t_bins.max() + 2

spike_matrix = torch.zeros(n_steps, n_afferents)
spike_matrix[t_bins, a_idx] = torch.tensor(hits_out["weights"].values, dtype=torch.float32)

print(spike_matrix)
print(spike_matrix.shape)


def spike_input(hits_in):
      " Funktionen gör om klupande hits till en matris för input till SNN"

      f_lhc = 40e6               #frequency
      omega = 2 * np.pi * f_lhc  #vinkelfrekvens
      time = 2*np.pi / omega     # full scan time
        #   n_afferents_full = 48
        #   n_steps_full = int(time * 1e9) + 2
      

      phi_pos = hits_in["phi"].values + 2 * np.pi * (hits_in["phi"].values < 0) 
      hits_in["t"] = phi_pos / omega                                                                    #At what time does the hit occur.                                                                 

      layer_radii       = hits_in.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
      layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
      n_afferents       = len(layer_to_afferent)           

      hits_in["a"] = pd.MultiIndex.from_frame(
          hits_in[["volume_id", "layer_id"]]
      ).map(layer_to_afferent)                                                                                             

      # wrap-around
      aux = hits_in[phi_pos <= 0.7].copy()
      aux["t"] = (phi_pos[phi_pos <= 0.7] + 2 * np.pi) / omega
      hits_out = pd.concat([hits_in, aux], ignore_index=True)

      t_bins  = (hits_out["t"].values * 1e9).astype(int)
      a_idx   = hits_out["a"].values.astype(int)
      n_steps = t_bins.max() + 2

      spike_matrix = torch.zeros(n_steps, n_afferents)
      spike_matrix[t_bins, a_idx] = torch.tensor(hits_out["weights"].values, dtype=torch.float32)

      return spike_matrix, n_afferents, n_steps, hits_out


def SNN_LEAKY(spike_input, n_afferents, n_steps, n_neurons = 1): #spike input = spike_matrix
    spike_flat = spike_input.flatten() 

   # Synaptic LIF neuron activation
    lif = snn.Leaky(beta=0.5, threshold=0.5)      # LIF neuron with a decay rate of 0.8

    # setup inputs
    num_steps = n_afferents * n_steps # number of time-steps to simulate
    w = 1 # then run 0.15, 0.20, 0.21

    # Small step current input
    mem = torch.zeros(n_neurons) #1
    spk = torch.zeros(n_neurons) #1

    mem_rec = []
    spk_rec = []

    # neuron simulation
    for step in range(num_steps):
        spk, mem = lif(spike_flat[step], mem)
        mem_rec.append(mem)
        spk_rec.append(spk)

    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    return spike_flat, mem_rec, spk_rec


def SNN_synaptic(spike_matrix, n_afferents, n_steps, n_neurons=1, alpha=0.9, beta=0.8, threshold=1.0):
    """
    Improved SNN using snn.Synaptic — processes all afferents per time step.
    spike_matrix: shape [n_steps, n_afferents]
    Returns: syn_rec, mem_rec, spk_rec  (all shape [n_steps])
    """
    lif = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold, reset_mechanism="subtract")

    # Learnable weights: sum contributions from all afferents
    w = torch.ones(n_afferents) / n_afferents   # uniform weight across afferents

    syn = torch.zeros(n_neurons)
    mem = torch.zeros(n_neurons)

    syn_rec, mem_rec, spk_rec = [], [], []

    for step in range(n_steps):
        cur_input = (spike_matrix[step] * w).sum().unsqueeze(0)   # [1] — weighted sum over afferents
        spk, syn, mem = lif(cur_input, syn, mem)
        syn_rec.append(syn.clone())
        mem_rec.append(mem.clone())
        spk_rec.append(spk.clone())

    syn_rec = torch.stack(syn_rec).squeeze()
    mem_rec = torch.stack(mem_rec).squeeze()
    spk_rec = torch.stack(spk_rec).squeeze()
    return mem_rec, syn_rec, spk_rec



# =========================================================================
# ================ Allmäna funktioner =====================================
particles_conditioned, hits_conditioned = preprocess_particles(10, 10, particles, hits)     
first_particle_ind = particles_conditioned.index                                           

mean = np.mean(particles["p"])           #for all particles plot the histogram of energies
median = np.median(particles["p"])       #for all particles plot the histogram of hits
top = particles["p"].quantile(0.98)      #for all particle take the value for the top 10% in energy


# =======================================================================================
# =============================== Test: Vald Partikel =================================== 
n = first_particle_ind[2]                                                                           #pick index of most energetic
signal, particle_id = choice_particels(hits_conditioned, n, truth, particles_conditioned)           #take that hits and the particle id.  


print("\n \n \n")
print("Vald partikel ID:", particle_id)
print("\n \n \n")


print()

spike_matrix, n_afferents, n_steps, hits_processed = spike_input(signal)                       
spike_flat, mem_rec, spk_rec = SNN_LEAKY(spike_matrix, n_afferents, n_steps)                         #throw that matrix into a SNN function and see the membrane potential


# plot_particle_hits(signal, h=hits)
# plot_aff_time(particle_id, hits_processed )
# histogram_of_particle_momenta(particles, top)
# histogram_of_particle_hits(particles)
# plot_cur_mem_spk(spike_flat, mem_rec, spk_rec, thr_line=1, ylim_max1=2.0, title="snn.Leaky Neuron Model - Signal")

# =======================================================================================
# =============================== Test: Brus ============================================

Noise = hits[~hits["hit_id"].isin(hits_conditioned["hit_id"])] 
Noise["weights"] = truth[ truth["hit_id"].isin(Noise["hit_id"])]["weight"]*10000

spike_matrix_b, n_afferents_b, n_steps_b, hits_processed_b = spike_input(Noise)
spike_flat_bd, mem_rec_bd, spk_rec_bd = SNN_LEAKY(spike_matrix_b, n_afferents_b, n_steps_b)


# print(f"shape of noise {spike_matrix_b.shape}")
# print(f"shape of signal {spike_matrix.shape}")


# plot_cur_mem_spk(spike_flat_bd, mem_rec_bd, spk_rec_bd, thr_line=1.0, ylim_max1=2.0, title="snn.Leaky Neuron Model - Noise")
# plot_hits(Noise)
# plot_hits(signal)


# ======================================================================================
#============================== NOISE + SIGNAL =========================================



spike_matrix_b[:spike_matrix.shape[0], :spike_matrix.shape[1]] += spike_matrix                                                                                                                                                         
spike_combined = spike_matrix_b

# plot_cur_mem_spk(spike_combined, mem_rec_bd, spk_rec_bd, thr_line=1.0, ylim_max1=2.0, title="snn.Leaky Neuron Model - Noise")








# =====================================================================================
# ========================== Plotting the Energies ====================================
 
fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')                                                                                                                                                                  
  
axs[0, 0].hist(particles['p'], bins=50, color='blue', alpha=0.7)
axs[0, 0].set_title('All Particles - Momentum')
axs[0,0].axvline(x=top, color='black', linestyle='dashed', linewidth=2)
axs[0, 0].set_xlabel('Momentum (GeV/c)')

axs[0, 1].hist(particles_conditioned['p'], bins=50, color='blue', alpha=0.7)
axs[0, 1].axvline(x=top, color='black', linestyle='dashed', linewidth=2)
axs[0, 1].set_title('Conditioned Particles - Momentum')
axs[0, 1].set_xlabel('Momentum (GeV/c)')

axs[1, 0].hist(particles['nhits'], bins=50, color='blue', alpha=0.7)
axs[1, 0].set_title('All Particles - Hits')
axs[1, 0].set_xlabel('Number of Hits')

axs[1, 1].hist(particles_conditioned['nhits'], bins=50, color='blue', alpha=0.7)
axs[1, 1].set_title('Conditioned Particles - Hits')
axs[1, 1].set_xlabel('Number of Hits')

# plt.show()


#=====================================================================================
#==========================PLOTTING NOISE VS SIGNAL MEMBRANE =========================


# fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')

# # Row 1: hit locations
# axs[0, 0].scatter(Noise["x"], Noise["y"], s=1, alpha=0.3)
# axs[0, 0].set_title('Noise Hits')

# axs[0, 1].scatter(signal["x"], signal["y"], s=10, color='red')
# axs[0, 1].set_xlim(-1000, 1000)
# axs[0, 1].set_ylim(-1000, 1000)                                                                                                                                                                                                          
# axs[0, 1].set_title('Signal Hits')

#  # Row 2: LIF membrane potential
# axs[1, 0].plot(spike_flat_bd.detach().numpy())
# axs[1, 0].axhline(y=2.0, linestyle='dashed', color='black', alpha=0.5)
# axs[1, 0].set_title('Noise LIF')

# axs[1, 1].plot(spike_flat.detach().numpy())
# axs[1, 1].axhline(y=2.0, linestyle='dashed', color='black', alpha=0.5)
# axs[1, 1].set_title('Signal LIF')

# plt.show()



#=====================================================================================
#========================== 3D because cool ==========================================

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(np.log10(particles["p"]), particles["nhits"], bins=50)
x, y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

ls = LightSource(270, 45)
rgb = ls.shade(hist, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
ax.plot_surface(x, y, hist, rstride=1, cstride=1, facecolors=rgb,
                  linewidth=0, antialiased=False, shade=False)

ax.set_xlabel('log10 Momentum (GeV/c)')
ax.set_ylabel('Number of Hits')
ax.set_zlabel('Count')
ax.set_title('Particle Distribution')
# plt.show()

