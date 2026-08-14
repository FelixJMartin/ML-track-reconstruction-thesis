# dataset
from trackml.dataset import load_event

# SNN
import snntorch as snn
from snntorch import spikeplot as splt
import torch

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scienceplots

# other
import os
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"] = False


# ============= GLOBAL   ======================================================
#vinkel och tid
f_lhc = 40e6                           # frequency
omega = 2 * np.pi * f_lhc              # vinkelhits_out["t"], hits_out["a"]]frekvens
time = 2*np.pi / omega                 # full scan time      
n_steps_full = ((int(time * 1e9) + 5 ) * 90) // 20   # full amount of time
# n_steps_full = (int(time * 1e9) + 5 )     # full amount of time

# ================================================================================

"""
mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
#print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))

"""


# print()
# print("Event:", event_prefix)
# print(f"(Alla viktiga shapes)\n")
# print("Loaded cells:", cells.shape)
# print("Loaded particles:", particles.shape)
# print("Loaded truth:", truth.shape)
# print()



# ================ PREPROCESSING FUNKTION ========================

def preprocess_particles(p_limit, n_hits, particles, hits):
  """
  Filtrera bort partiklar under p_limit GeV, under n_hits, ger ut en partikellista och en intressant hit lista

  Returns:  
    high_energy_particles : pd.DataFrame — partiklar filtrerade på p > p_limit och nhits > n_hits, sorterade efter p
    hits_cond             : pd.DataFrame — hits filtrerade till volymerna [8, 13, 17]
    n_afferents_full      : int          — antal unika lager (afferenter) i hits_cond
  """


  # Filtrera bort partiklar med p under p_limit
  px = particles['px']
  py = particles['py']
  p = np.sqrt(px**2 + py**2)
  particles['p'] = p
  high_energy_particles = particles[p > p_limit]

  # Filtrera bort partiklar med färre än n_hits
  hit = high_energy_particles['nhits']
  high_energy_particles = high_energy_particles[hit > n_hits].sort_values("p", ascending=False)

  # # Filter hits för de intressanta partiklarna och volymerna
  interesting_volumes =[8, 13, 17] 
  hits_cond = hits[hits['volume_id'].isin(interesting_volumes)].copy()

  layer_radii  = hits_cond.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
  layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
  n_afferents_full  = len(layer_to_afferent)

  #hits_interesting är alla hits som tillhör en signalbana
  # valid_pids = high_energy_particles["particle_id"]                
  # valid_hit_ids = truth[truth["particle_id"].isin(valid_pids)]["hit_id"]                                                                                                                                                                       
  # hits_interesting = hits[hits["hit_id"].isin(valid_hit_ids)]

  return high_energy_particles, hits_cond, n_afferents_full


def choice_particels(hits_in,  particles_cond, n,  truth): 
  """
  Denna funktion tar nte högsta energiska partikelbana och ger specifkt hits kopplade till den.
  In : hits_in, particles cond, n och truth.
  
  Returns:
    chosen_hits : pd.DataFrame — hits tillhörande partikeln med index n, med tillagd kolumn 'weights' (truth weight × 10000)
    pid         : int          — particle_id för den valda partikeln
  
  """

  test_particle = particles_cond.loc[n]    # loc = index, iloc = position                                                                                                                           
  pid = test_particle["particle_id"]                                                                                                                             
  hit_ids = truth.loc[truth["particle_id"] == pid, "hit_id"]
  hits_in = hits_in.copy() 
  weight_map = truth.set_index("hit_id")["weight"]
  hits_in["weights"] = hits_in["hit_id"].map(weight_map) * 10000
  chosen_hits = hits_in.loc[hits_in["hit_id"].isin(hit_ids)].copy()

  return chosen_hits, pid


def particle_track_hits(particle_list, truth):
  """
  Kan ge particle conditioned och få ut alla hits med vikter från den.

  Returns:
    track_hits : pd.DataFrame — alla rader från truth som tillhör någon av partiklarna i particle_list
  """
  track_hits = truth[truth["particle_id"].isin(particle_list["particle_id"])].copy()
  track_hits["weight"] = track_hits["weight"] * 10000

  return track_hits


def give_weights(hit_list, truth):
  """
  Take a list of hits and gives it correct weights

  Returns:
    hit_list : pd.DataFrame — same hit list with added 'weight' column (truth weight × 10000)
  """

  weight_map = truth.set_index("hit_id")["weight"]
  hit_list = hit_list.copy()
  hit_list["weights"] = hit_list["hit_id"].map(weight_map) * 10000
  
  return hit_list


# ================ PLOTTING FUNCTIONS =====================================


def plot_aff_time(id, hits):
  """
  Plot the afferent vs time for a given particle id
  
  """
                                                                                                                                                                   
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(hits["t"] * 1e9, hits["a"], s=100, color="red", zorder=5)
  ax.set_xlabel("Spike time t (ns)")
  ax.set_ylabel("Afferent a (layer)")
  ax.set_yticks(sorted(hits["a"].unique()))
  ax.set_title(f"SNN Spike Encoding — Particle {[id]}")
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

def plot_cur_mem_spk(cur, mem, spk, thr_line=1, vline=False, title=False, ylim_max1=8.5, ylim_max2=8.5):
  """
  Plots the 3 windows of input current, membrane function and spikes
  """
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, n_steps_full])
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

def histogram_of_particle_momenta(particles):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(particles['p']), bins=50, color='steelblue', alpha=0.7)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.2,
               label=r'Signal cut ($p_T > 10\,\mathrm{GeV}$)')
    ax.set_xlabel(r'$\log_{10}\,p_T$ (GeV)')
    ax.set_ylabel('Number of Particles')
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    plt.tight_layout()


def histogram_of_particle_hits(particles):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(particles['nhits'], bins=50, color='steelblue', alpha=0.7)
    ax.axvline(x=10, color='black', linestyle='--', linewidth=1.2,
               label=r'Signal cut ($n_\mathrm{hits} > 10$)')
    ax.set_xlabel('Number of Hits')
    ax.set_ylabel('Number of Particles')
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    plt.tight_layout()

def plot_hits_noisy(inter_hit, h):
    """
    Plots the incomming hits against all hits
    """

    _, ax = plt.subplots(figsize=(6, 6))
    background = h[h["r"] > 10]
    ax.scatter(background["x"], background["y"], s=1, alpha=0.04, color='steelblue')
    ax.scatter(inter_hit["x"], inter_hit["y"], s=30, alpha=0.9, color='crimson',
               zorder=5, edgecolors='black', linewidths=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title("Highest $p_T$ particle in event")
    plt.tight_layout()
    plt.show()

def plot_hits_self(hits):
  """
  Plots the incomming hits alone 
  """
  _, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(hits["x"], hits["y"], s=10, alpha=0.9, color='crimson',
               zorder=5, edgecolors='black', linewidths=0.5)
  ax.set_aspect('equal')
  ax.set_xlabel('x (mm)')
  ax.set_ylabel('y (mm)')
  ax.set_title("Highest $p_T$ particle in event - solo")
  ax.set_xlim(-1500, 1500)
  ax.set_ylim(-1500, 1500)
  plt.tight_layout()
  plt.show()

def plot_spike_full(spike_matrix, spike_row, cur, mem, spk, thr_line=1,
                    ylim_max1=8.5, ylim_max2=8.5):
    """Left: spike matrix + flattened row. Right: input current, membrane potential, output spikes."""
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.5], wspace=0.1)

    # left column: matrix (4) + row (1)
    gs_l = gs[0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
    ax_mat = fig.add_subplot(gs_l[0])
    ax_row = fig.add_subplot(gs_l[1])

    # right column: cur (1) + mem (1) + spk (0.4)
    gs_r = gs[1].subgridspec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.02)
    ax_cur = fig.add_subplot(gs_r[0])
    ax_mem = fig.add_subplot(gs_r[1], sharex=ax_cur)
    ax_spk = fig.add_subplot(gs_r[2], sharex=ax_cur)

    # spike matrix
    im_mat = ax_mat.imshow(spike_matrix.numpy()[::-1], cmap="inferno", origin="lower",
                            aspect="auto", norm=mcolors.PowerNorm(gamma=0.7))
    ax_mat.set_ylabel("Afferent (layer)")
    n = spike_matrix.shape[0]
    ax_mat.set_yticks(range(0, n, 5))
    ax_mat.set_yticklabels(range(0, n, 5))
    ax_mat.tick_params(labelbottom=False)  # hide x ticks on matrix, row has them
    fig.colorbar(im_mat, ax=ax_mat, label="Spike", pad=0.01)

    # flattened row
    im_row = ax_row.imshow(spike_row.numpy().reshape(1, -1), cmap="inferno", aspect="auto")
    ax_row.set_xlabel("Time step")
    ax_row.set_yticks([])
    fig.colorbar(im_row, ax=ax_row, label="Summed spikes", pad=0.01)

    # input current
    ax_cur.plot(cur, c="tab:orange", linewidth=0.8)
    ax_cur.set_xlim([0, n_steps_full])
    ax_cur.set_ylim([0, ylim_max1 ])
    ax_cur.set_ylabel("Input Current ($I_{in}$)")
    ax_cur.tick_params(labelbottom=False)

    # membrane potential
    ax_mem.plot(mem, linewidth=0.8)
    ax_mem.set_ylim([2, ylim_max2 +2])
    ax_mem.set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax_mem.axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    ax_mem.tick_params(labelbottom=False)

    # output spikes
    splt.raster(spk, ax_spk, s=400, c="black", marker="|")
    ax_spk.set_xlabel("Time step")
    ax_spk.set_ylabel("Output spikes")
    ax_spk.set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_with_weight(hits):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_facecolor('white')
    
    sc = ax.scatter(hits["x"], hits["y"], s=3, alpha=0.7,
                    c=hits["weights"], cmap='inferno',
                    zorder=5, linewidths=0)
    
    plt.colorbar(sc, ax=ax, label='weight', shrink=0.7)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.show()
   
def geometric(particle_hits, truth, hits, pid = None):
    """
    Plots the 3D track of one particle, annotated with volume_id and layer_id at each hit.

    Parameters:
        particle_hits : pd.DataFrame — hits for one specific particle (from choice_particels or similar)
        truth         : pd.DataFrame — truth table (for tx, ty, tz true positions)
        hits          : pd.DataFrame — full hits table (for volume_id, layer_id)
    """

    # Get true trajectory positions
    p_traj = truth[truth["particle_id"] == pid][["hit_id", "tx", "ty", "tz"]].copy()

    # Merge in volume_id and layer_id
    p_traj = p_traj.merge(hits[["hit_id", "volume_id", "layer_id"]], on="hit_id")
    p_traj = p_traj.sort_values("tz")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_traj["tx"], p_traj["ty"], p_traj["tz"], marker='o', linewidth=1.5)

    for _, row in p_traj.iterrows():
        ax.text(row["tx"], row["ty"], row["tz"],
                f'  v{int(row["volume_id"])}l{int(row["layer_id"])}',
                fontsize=7, alpha=0.8)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D Track — particle_id {pid}")
    plt.tight_layout()
    plt.show()

def plot_hits(hits):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    volumes = sorted(hits.volume_id.unique())
    palette = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', 
           '#C4AD66', '#77BEDB', '#e08f3a', '#888888', '#c44e52']

    for i, volume in enumerate(volumes):
        v = hits[hits.volume_id == volume]
        ax.scatter(v.x, v.y, s=1, alpha=0.2, 
                   label=f"volume {volume}", color=palette[i],
                   rasterized=True)  # important for PDF export

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_xlim(-1100, 1100)
    ax.set_ylim(-1100, 1100)
    ax.set_aspect('equal')  # keeps circles circular!
    
    plt.tight_layout()
    plt.savefig('detector.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hits_rz(hits):
    
    _, ax = plt.subplots(figsize=(8,8))
    
    volumes = sorted(hits.volume_id.unique())
    palette = sns.color_palette("rainbow", len(volumes))

    for i, volume in enumerate(volumes):
      v = hits[hits.volume_id == volume]
      plt.scatter(v.z, v.r,  s=3, alpha=0.3, label=f"volume {volume}", color=palette[i])

    
    plt.xlabel('z (mm)')
    plt.ylabel('r (mm)')
    plt.xlim(-1500,1500)
    plt.ylim(-1500,1500)
    plt.legend(markerscale=5, fontsize=8, loc='upper left', frameon=False)
    plt.show()

def plot_hits_3d(hits, s=1):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')  # 3d projection
    
    volumes = sorted(hits.volume_id.unique())
    palette = sns.color_palette("rainbow", len(volumes))

    for i, volume in enumerate(volumes):
        v = hits[hits.volume_id == volume]
        ax.scatter(v.x, v.y, v.z,          # add z coordinate
                   s=s, alpha=0.3, 
                   label=f"volume {volume}", 
                   color=palette[i])


    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    ax.set_zlim(-1200, 1200)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')                  # add z label
    ax.view_init(elev=20, azim=-60)

    ax.legend(markerscale=5, fontsize=8, 
              loc='upper left', frameon=False)
    plt.show()


# ================ SNN DEFINITIONS =====================================

def spike_input(hits_in, n_afferent):
  """
  Funktionen gör om klupande hits till en matris för input till SNN oavsett size.
  
  Parameters: 
  Hits in      : The hits we wanna use for the spikes.
  n_afferents  : The amount off afferents we wanna use.

  Returns:
  spike_matrix : torch.Tensor [n_afferents, n_steps] — full afferent x time spike matrix
  spike_row    : torch.Tensor [1, n_steps]           — summed input current across all afferents
  hits_out     : pd.DataFrame                        — hits with added columns t (time) and a (afferent index)
  """

  hits_in = hits_in.copy()  
  
  phi_pos = hits_in["phi"].values + 2 * np.pi * (hits_in["phi"].values < 0) 
  hits_in["t"] = phi_pos / omega
                                                                                              
  layer_radii       = hits_in.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
  layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
  hits_in["a"] = pd.MultiIndex.from_frame(hits_in[["volume_id", "layer_id"]]).map(layer_to_afferent)                                                                                             
  hits_out = hits_in

  # aux = hits_in[phi_pos <= 0.7].copy()
  # aux["t"] = (phi_pos[phi_pos <= 0.7] + 2 * np.pi) / omega
  # hits_out = pd.concat([hits_in, aux], ignore_index=True)
  a_idx = hits_out["a"].values.astype(int)
  t_idx = (hits_out["t"] / time * n_steps_full).round().astype(int).clip(0, n_steps_full - 1).values


  spike_matrix = torch.zeros(n_afferent, n_steps_full)
  spike_matrix[ (n_afferent - 1 - a_idx), t_idx] += torch.tensor(hits_out["weights"].values, dtype=torch.float32)
  spike_row = spike_matrix.sum(dim=0, keepdim=True)

  return spike_matrix, spike_row, hits_out, t_idx


def SNN_LEAKY(spike_input, n_neurons = 1, threshold=1.0): 
  """
  Leaky fire and integrate function of one Neuron!
  Parameters:
  spike_input :
  n_neurons :
  threshold :
 
  Returns:
    mem_rec : torch.Tensor [n_steps] — membranpotential över tid
    spk_rec : torch.Tensor [n_steps] — spike output över tid (0 eller 1 per tidssteg)
  """


  # Synaptic LIF neuron activation
  lif = snn.Leaky(beta=0.6, threshold = threshold)      # LIF neuron with a decay rate of 0.6 and reset - threshold

  # setup inputs
  num_steps = n_steps_full # number of time-steps to simulate
  w = 1 # then run 0.15, 0.20, 0.21

  # Small step current input
  mem = torch.zeros(n_neurons) #1
  spk = torch.zeros(n_neurons) #1

  mem_rec = []
  spk_rec = []

  # neuron simulation
  for step in range(num_steps):
      spk, mem = lif(spike_input[0][step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk)

  # convert lists to tensors
  mem_rec = torch.stack(mem_rec)
  spk_rec = torch.stack(spk_rec)
  return mem_rec, spk_rec


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


# ==================================================================================
# ===================== För att kunna importera ====================================


if __name__ == "__main__": 
    
  event_prefix = "event000001000"
  data_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
  hits, cells, particles, truth = load_event(os.path.join(data_dir, event_prefix))

  # Lägg till r och phi i hits
  hits["r"] = np.hypot(hits["x"], hits["y"])
  hits["phi"] = np.arctan2(hits["y"], hits["x"])


  px = particles['px']
  py = particles['py']
  p = np.sqrt(px**2 + py**2)
  particles['p'] = p

  
  histogram_of_particle_hits(particles)
  histogram_of_particle_momenta(particles)
  plot_hits(hits)

  #====================================================================================
  #========================= Test: 1 Vald partikel: aff = 81317 =======================


  #Selection process
  particles_conditioned, hits_conditioned, n_afferent_trim = preprocess_particles(10, 10, particles, hits)     
  first_particle_ind = particles_conditioned.index                                           
  n = first_particle_ind[0]                                       
  signal, particle_id = choice_particels(hits_conditioned, particles_conditioned, n, truth )  

  layer_radii       = hits.groupby(["volume_id", "layer_id"])["r"].mean().sort_values()
  layer_to_afferent = {layer: i for i, layer in enumerate(layer_radii.index)}
  n_afferents_full  = len(layer_to_afferent) 

  #Into LIF
  spike_matrix_s, spike_row_s, hits_out_s, t_idx_s = spike_input(signal, n_afferent_trim)
  mem_rec_s, spk_rec_s = SNN_LEAKY(spike_row_s, threshold= 1.5)

  # Plotting
  plot_spike_full(spike_matrix_s, spike_row_s, spike_row_s.flatten(), mem_rec_s, spk_rec_s,
                  thr_line=1.5)


  # =======================================================================================
  # =============================== ALLA SIGNALER =========================================

  particles_conditioned, hits_conditioned, n_afferent_trim = preprocess_particles(10, 10, particles, hits)     
  all_signal_hits = truth[truth["particle_id"].isin(particles_conditioned["particle_id"])]["hit_id"]
  hit_WW = give_weights(hits_conditioned[hits_conditioned["hit_id"].isin(all_signal_hits)], truth)

  
  # #Into LIF
  spike_matrix_a, spike_row_a, hits_out_a, t_idx_a = spike_input(hit_WW, n_afferent_trim)
  mem_rec_a, spk_rec_a = SNN_LEAKY(spike_row_a, threshold= 1.5)

  # # Plotting
  plot_spike_full(spike_matrix_a, spike_row_a, spike_row_a.flatten(), mem_rec_a, spk_rec_a,
                  thr_line=1.5)

  # print(f"\n{'the indexes where signal appears are'.center(80)}\n{t_idx_a}\n")
  # print(f"\n{'times when spikes occur'.center(80)}\n{torch.nonzero(spk_rec_a)}\n")


  # =======================================================================================
  # =============================== Test: Brus i V81317====================================

  thres = 2.5
  
  #Selection
  good_hit_ids = truth[truth["particle_id"].isin(particles_conditioned["particle_id"])]["hit_id"]
  Noise = hits_conditioned[~hits_conditioned["hit_id"].isin(good_hit_ids)].copy()
  Noise["weights"] = truth[truth["hit_id"].isin(Noise["hit_id"])]["weight"] * 10000

  #Into a LIF
  spike_matrix_n, spike_row_n, hits_out_n, t_idx= spike_input(Noise, n_afferent_trim)
  mem_rec_n, spk_rec_n = SNN_LEAKY(spike_row_n, threshold=thres) 

  #Plotting
  plot_spike_full(spike_matrix_n, spike_row_n, spike_row_n.flatten(), mem_rec_n, spk_rec_n,
                  thr_line=thres)
  # # print(f"\n{'Here Ive taken all the noise and the hits'.center(80)}\n{Noise}\n")
  # print(f"\n{'Noise matrix'.center(80)}\n{spike_matrix_n}\n")
  # print(f"\n{'And this is the flat noisy one'.center(80)}\n{spike_row_n}\n")



  # ======================================================================================
  #============================== NOISE + ONE SIGNAL =========================================

  # #selection
  # spike_matrix_ns = spike_matrix_s + spike_matrix_n
  # spike_row_ns = spike_row_n + spike_row_s
  # mem_rec_ns = mem_rec_s + mem_rec_n
  # spk_rec_ns = spk_rec_s + spk_rec_n

  # #Plotting
  # plot_spike_full(spike_matrix_ns, spike_row_ns, spike_row_ns.flatten(), mem_rec_ns, spk_rec_ns,
  #                 title="Spike matrix (signal + noise)")
  # print(f"\n{'Noise and sig matrix'.center(80)}\n{spike_matrix_n}\n")
  # print(f"\n{'And this is the flat noisy + sign one'.center(80)}\n{spike_row_n}\n")


  # ======================================================================================
  #============================== NOISE ALL + ALL SIGNAL =========================================
  thres = 6.5

  #Selection
  good_hit_ids = truth[truth["particle_id"].isin(particles_conditioned["particle_id"])]["hit_id"]
  Noise = hits[~hits["hit_id"].isin(good_hit_ids)].copy()
  Noise["weights"] = truth[truth["hit_id"].isin(Noise["hit_id"])]["weight"] * 6000

  #Into a LIF
  spike_matrix_n, spike_row_n, hits_out_n, t_idx= spike_input(Noise, n_afferents_full)
  mem_rec_n, spk_rec_n = SNN_LEAKY(spike_row_n, threshold=thres)

  # rebuild all-signal at full volume resolution to match noise (48 afferents)
  hit_WW_full = give_weights(hits[hits["hit_id"].isin(all_signal_hits)], truth)
  spike_matrix_a_full, spike_row_a_full, _, _ = spike_input(hit_WW_full, n_afferents_full)

  #selection
  spike_matrix_na = spike_matrix_n + spike_matrix_a_full
  spike_row_na = spike_row_n + spike_row_a_full
  mem_rec_na = mem_rec_n + mem_rec_a
  spk_rec_na = spk_rec_n + spk_rec_a

  # Plotting
  plot_spike_full(spike_matrix_na, spike_row_na, spike_row_na.flatten(), mem_rec_na, spk_rec_na,
                  thr_line=thres)
  # print(f"\n{'Noise and all sig matrix'.center(80)}\n{spike_matrix_na}\n")
  # print(f"\n{'And this is the flat noisy + all signal ones'.center(80)}\n{spike_row_na}\n")


  # =====================================================================================
  # ========================== Plotting the Energies ====================================
  
  # mean = np.mean(particles["p"])           #for all particles plot the histogram of energies
  # median = np.median(particles["p"])       #for all particles plot the histogram of hits
  # top = particles["p"].quantile(0.98)      #for all particle take the value for the top 10% in energy

  # histogram_of_particle_momenta(particles, top)
  # histogram_of_particle_hits(particles)

  # fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')                                                                                                                                                                  
    
  # axs[0, 0].hist(particles['p'], bins=50, color='blue', alpha=0.7)
  # axs[0, 0].set_title('All Particles - Momentum')
  # axs[0,0].axvline(x=top, color='black', linestyle='dashed', linewidth=2)
  # axs[0, 0].set_xlabel('Momentum (GeV/c)')

  # axs[0, 1].hist(particles_conditioned['p'], bins=50, color='blue', alpha=0.7)
  # axs[0, 1].axvline(x=top, color='black', linestyle='dashed', linewidth=2)
  # axs[0, 1].set_title('Conditioned Particles - Momentum')
  # axs[0, 1].set_xlabel('Momentum (GeV/c)')

  # axs[1, 0].hist(particles['nhits'], bins=50, color='blue', alpha=0.7)
  # axs[1, 0].set_title('All Particles - Hits')
  # axs[1, 0].set_xlabel('Number of Hits')

  # axs[1, 1].hist(particles_conditioned['nhits'], bins=50, color='blue', alpha=0.7)
  # axs[1, 1].set_title('Conditioned Particles - Hits')
  # axs[1, 1].set_xlabel('Number of Hits')

  # plt.show()



  #=====================================================================================
  #========================== 3D because cool ==========================================

  # fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(111, projection='3d')

  # hist, xedges, yedges = np.histogram2d(np.log10(particles["p"]), particles["nhits"], bins=50)
  # x, y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

  # ls = LightSource(270, 45)
  # rgb = ls.shade(hist, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
  # ax.plot_surface(x, y, hist, rstride=1, cstride=1, facecolors=rgb,
  #                   linewidth=0, antialiased=False, shade=False)

  # ax.set_xlabel('log10 Momentum (GeV/c)')
  # ax.set_ylabel('Number of Hits')
  # ax.set_zlabel('Count')
  # ax.set_title('Particle Distribution')
  # plt.show()







