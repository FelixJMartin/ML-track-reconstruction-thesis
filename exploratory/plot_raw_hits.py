# =========== IMPORTS ==============================
import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event    # These are libraries provided by the organizers to load the data and compute the score.
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt           # Just regualr import to plot.
# plt.style.use("dark_background")
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# ============= INITIALIZE ==================================================

event_prefix = "event000001000"
data_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\train_100_events"
hits, cells, particles, truth = load_event(os.path.join(data_dir, event_prefix))




# =========== CIRCULAR PLOT ALL HITS ================================================

g = sns.jointplot(x="x", y="y", data=hits, s=1, height=12)  # data x, y from hits and marker size 1, and size of the plot.
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()

for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.x, v.y, s=3, label='volume {}'.format(volume))  # we iterate over the different volumes and plots the x,y for each and label the
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend(markerscale=5, fontsize=12)
# plt.show()  # Now over to matplot to actually plot the graph.


# Plot from side view:
g = sns.jointplot(x="z", y="y", data=hits, s=1, height=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.z, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.legend(markerscale=5, fontsize=12)
# plt.show()





# ====== 3D SCATTER ALL HITS ==============================================================

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
for volume in volumes:
    v = hits[hits.volume_id == volume]
    ax.scatter(v.z, v.x, v.y, s=1, label='volume {}'.format(volume), alpha=0.5)
ax.set_title('Hit Locations')
ax.set_xlabel('Z (millimeters)')
ax.set_ylabel('X (millimeters)')
ax.set_zlabel('Y (millimeters)')
plt.show()



# =========== THIS PLOTS THE TRAJECTORY OF THE PARTICLE WITH THE MOST HITS IN THIS EVENT ==========

particle = particles.loc[particles.nhits == particles.nhits.max()].iloc[0] #particle_id    1.531254e+17"
p_traj_surface = truth[truth.particle_id == particle.particle_id][['tx', 'ty', 'tz']]  #take it's x,y,x.
row1 = pd.DataFrame([{'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}])
p_traj  = pd.concat([p_traj_surface,  row1], ignore_index=True).sort_values('tz')


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=p_traj.tx, ys=p_traj.ty, zs=p_traj.tz, marker='o')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z  (mm) -- Detection layers')
plt.title('Trajectories of one particles as they cross the detection surface ($Z$ axis).')
plt.show()


# =========== MULTI-TRACK PLOT (reproduces the radiating tracks image) ===========

# Take the top N particles by number of hits; noise particles have particle_id==0
N_TRACKS = 250
top_particles = (particles[particles.particle_id != 0]
                 .nlargest(N_TRACKS, 'nhits'))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for _, p in top_particles.iterrows():
    hits_p = truth[truth.particle_id == p.particle_id][['tz', 'tx', 'ty']]
    if hits_p.empty:
        continue
    vertex = pd.DataFrame([{'tx': p.vx, 'ty': p.vy, 'tz': p.vz}])
    track = pd.concat([vertex, hits_p], ignore_index=True).sort_values('tz')
    ax.plot(track.tz, track.tx, track.ty, lw=0.8, alpha=0.7)

ax.set_axis_off()
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('none')

plt.tight_layout()
plt.show()

