# Fungerande för 1 membranpotential

import os
import torch, torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
import numpy as np
import pandas as pd
from trackml.dataset import load_event
import matplotlib.pyplot as plt
import seaborn as sns

print("\n \n ",30*"=","Ny Körning",30*"=")

event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('train_100_events', event_prefix))
#load_event(os.path.join('train_100_events', event_prefix))


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



# ================ VISUALIZATION ==================
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
plt.savefig("Hits_and_Volumes.png", dpi=100, bbox_inches="tight")
plt.close()



# ================ CHOOSING PARTICLES ==================
def over_10_GeV(particles):
    px = particles['px']
    py = particles['py']
    pz = particles['pz']
    p = np.sqrt(px**2 + py**2 + pz**2)
    return particles[p > 10]

def over_10_hits(particle):
    hit = particle['nhits']
    return particle[hit > 10]

def xy_to_r_phi(hits):
    barrel_hits = hits
    barrel_hits["r"] = np.sqrt(barrel_hits["x"]**2 + barrel_hits["y"]**2)
    barrel_hits["phi"] = np.arctan2(barrel_hits["y"], barrel_hits["x"])
    return barrel_hits

def choosing_particle(n, p, h, t):
    # välj partikel
    test_particle = p.loc[n]
    pid = test_particle["particle_id"]

    # hitta hit-id:n
    hit_ids = t.loc[t["particle_id"] == pid, "hit_id"]

    # välj hits
    particle_hits = h.loc[h["hit_id"].isin(hit_ids)].copy()

    # beräkna tid
    f = 40e6
    omega = 2 * np.pi * f

    phi = particle_hits["phi"].values
    phi_wrapped = phi + 2*np.pi * (phi < 0)   # vektoriserad (som i andra koden)

    particle_hits["t"] = phi_wrapped / omega

    print("\nPartikel nr:", n)
    print("Antal hits:", len(particle_hits))

    return particle_hits

def plot_particle_hits(particle_hits, h=hits):
    id = particle_hits["hit_id"].values
    plt.figure(figsize=(6, 6))
    plt.scatter(h["x"], h["y"], s=1, alpha=0.2, color='blue')
    plt.scatter(particle_hits["x"], particle_hits["y"], s=7, alpha=0.6, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hits in x-y plane, partikel nr {}'.format(id))
    plt.savefig("Vald_Partikel.png", dpi=100, bbox_inches="tight")
    plt.close()

    # plot spike encoding
    plt.figure(figsize=(10, 5))
    plt.scatter(particle_hits["t"], particle_hits["layer_id"], s=60)
    plt.xlabel("t (sekunder)")
    plt.ylabel("a (afferent / lager)")
    plt.title("Spike-encoding för vald partikel")
    plt.grid(True)
    plt.savefig("Spike_Encoding.png", dpi=100, bbox_inches="tight")
    plt.close()

barrel_hits = xy_to_r_phi(hits)




def spike_input_for_particle(p_hits, barrel_hits=barrel_hits):

    # Layer -> afferent (sorterad på radie)
    layer_radii = barrel_hits.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
    layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}

    n_afferents = len(layer_to_afferent)

    # Vektoriserad mapping (ersätter iterrows)
    p_hits["a"] = pd.MultiIndex.from_frame(
        p_hits[["volume_id", "layer_id"]]
    ).map(layer_to_afferent)

    # Spike times
    spike_times_ns = (p_hits["t"].values * 1e9).astype(int)
    a_idx = p_hits["a"].values.astype(int)

    n_steps = spike_times_ns.max() + 2

    # Bygg tensor (vektoriserat)
    spike_input = torch.zeros(n_steps, n_afferents)
    spike_input[spike_times_ns, a_idx] = 1.0

    # Flatten (fix: utanför loop)
    spike_flat = spike_input.flatten()

    return spike_input, spike_flat, n_afferents, n_steps



" Definiera funktioner för att välja ut en partikel och plotta dess hits "
high_energy_particles = over_10_GeV(particles)
high_energy_particles_many_hit = over_10_hits(high_energy_particles)
test_partikel_id = 5929
test_p = choosing_particle(test_partikel_id, particles, hits, truth)
test_particle = plot_particle_hits(test_p, h=hits)
plot_particle_hits(test_p, h=hits)
spike_input, spike_flat, n_afferents, n_steps = spike_input_for_particle(test_p, barrel_hits=barrel_hits)






#Beginning the SNN and ploting the membrane function ----------------------------------------------------------------------

#@title Plotting Settings
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.5, ylim_max2=1.5):
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
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([]) 

  plt.savefig("Membrane_Potential.png", dpi=100, bbox_inches="tight")
  plt.close()

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
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
  plt.savefig("Output_Spikes.png", dpi=100, bbox_inches="tight")
  plt.close()

def dvs_animator(spike_data):
  fig, ax = plt.subplots()
  anim = splt.animator((spike_data[:,0] + spike_data[:,1]), fig, ax)
  return anim


lif = snn.Leaky(beta=0.8) # LIF neuron with a decay rate of 0.8

# setup inputs
num_steps = n_steps#280 # number of time-steps to simulate

w = 0.5 # then run 0.15, 0.20, 0.21

mem = torch.zeros(n_afferents) # ERSÄTT 1 med: mem = torch.zeros(n_afferents)
spk = torch.zeros(n_afferents)

mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
  spk, mem = lif(spike_input[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(spike_flat, mem_rec, spk_rec, thr_line=1, ylim_max1=1.0,
                 title="snn.Leaky Neuron Model")



print("\n \n ",45*"=","Klart",45*"=","\n \n")

