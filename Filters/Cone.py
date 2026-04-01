# In this document i will and filter for each volume etc.

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


# ---------------------- Beginning of the code ------------------------------------------------------------------------------

event_prefix = "event000001000"     # Choose one
data_dir = r"train_100_events"      # To direct later on.

hits, cells, particles, truth = load_event(  # This one unpacks the event and gets the 4 different files from it to get all nicissary info.
    os.path.join(data_dir, event_prefix)
)




# ------------------ Side-cone filter (cylindrical geometry) ------------------
hits = hits.copy()
hits = hits.merge(truth[["hit_id", "tpx", "tpy", "tpz"]], on="hit_id")

hits["p"] = np.sqrt(hits["tpx"]**2 + hits["tpy"]**2 + hits["tpz"]**2)
hits = hits[hits["p"] >= 10]  # Keep only hits with momentum >= 10 MeV

# --- Cone parameters (adjust if needed) ---
cone_opening_angle = 0.8  # radians - half-angle of the cone
apex = np.array([0.0, 0.0, 0.0])  # Cone apex position [x, y, z] in mm

# Pick a transverse direction (cone axis direction):
phi0 = 0.0  # 0 -> +x, pi/2 -> +y, etc.
cone_axis = np.array([np.cos(phi0), np.sin(phi0), 0.0])
cone_axis = cone_axis / np.linalg.norm(cone_axis)

# Calculate vectors from apex to each hit
P = hits[["x", "y", "z"]].to_numpy()
V = P - apex  # vectors from apex to hits
V_norm = np.maximum(np.linalg.norm(V, axis=1), 1e-12)

# Dot product gives cos(angle) between hit vector and cone axis
cos_gamma = (V @ cone_axis) / V_norm
# Filter: hits must be in forward direction AND within cone angle
mask = (cos_gamma > 0) & (cos_gamma > np.cos(cone_opening_angle))

inside = hits[mask]
outside = hits[~mask]

# Save cone-filtered hits as a new dataset
inside.to_csv("cone_hits.csv", index=False)
print(f"Cone hits saved: {len(inside)} hits out of {len(hits)} total")

# --- Visualization with cone wireframe ---
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

# Plot hits (swapped axes for side view: z, x, y)
# ax.scatter(outside["z"], outside["x"], outside["y"], s=1, alpha=0.05, color="green", label="Outside cone")
ax.scatter(inside["z"], inside["x"], inside["y"], s=2, alpha=0.8, color="red", label="Inside cone")



# --- Draw cone wireframe ---
# Generate cone surface for visualization
cone_length = 1500  # mm - extend cone to detector edges
u = np.linspace(0, cone_length, 50)  # along cone axis
theta = np.linspace(0, 2*np.pi, 30)  # around cone

# Create meshgrid for cone surface
u_grid, theta_grid = np.meshgrid(u, theta)
# Radius increases linearly with distance from apex
radius = u_grid * np.tan(cone_opening_angle)

# Cone in local coordinates (axis along x)
x_cone_local = u_grid
y_cone_local = radius * np.cos(theta_grid)
z_cone_local = radius * np.sin(theta_grid)

# Rotate cone to align with cone_axis direction
# Create rotation matrix to rotate from [1,0,0] to cone_axis
from_vec = np.array([1, 0, 0])
if not np.allclose(cone_axis, from_vec):
    v = np.cross(from_vec, cone_axis)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, cone_axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))
else:
    R = np.eye(3)

# Apply rotation and translation
cone_points = np.stack([x_cone_local.flatten(), y_cone_local.flatten(), z_cone_local.flatten()], axis=1)
cone_rotated = (R @ cone_points.T).T + apex
x_cone = cone_rotated[:, 0].reshape(x_cone_local.shape)
y_cone = cone_rotated[:, 1].reshape(y_cone_local.shape)
z_cone = cone_rotated[:, 2].reshape(z_cone_local.shape)

# Plot cone wireframe (swapped for side view)
ax.plot_wireframe(z_cone, x_cone, y_cone, color='blue', alpha=0.15, linewidth=0.5, label="Cone boundary")

ax.set_xlabel("Z (mm)")
ax.set_ylabel("X (mm)")
ax.set_zlabel("Y (mm)")
ax.set_title(f"Cone Filter: angle={np.degrees(cone_opening_angle):.1f}°, apex={apex}, axis={cone_axis}")
ax.legend(loc='upper right')
plt.show()





# # --- 3D scatter of hits (colored by volume) ---
# fig = plt.figure(figsize=(12, 12))

# # volumes = hits.volume_id.unique()


# # print(volumes)

# # ax = fig.add_subplot(111, projection='3d')
# # for volume in volumes:
# #     v = hits[hits.volume_id == volume]
# #     ax.scatter(v.z, v.x, v.y, s=1, label='volume {}'.format(volume), alpha=0.5)  # Alpha is opacity, to make it easier to see the different volumes.
# # ax.set_title('Hit Locations')
# # ax.set_xlabel('Z (millimeters)')
# # ax.set_ylabel('X (millimeters)')
# # ax.set_zlabel('Y (millimeters)')
# # plt.show()

# # colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
# # n_volumes = len(volumes)
# # fig = plt.figure(figsize=(18, 18))
# # for i, volume in enumerate(volumes):
# #     ax = fig.add_subplot(3, 3, i + 1, projection='3d')
# #     v = hits[hits.volume_id == volume]
# #     ax.scatter(v.z, v.x, v.y, s=1, color=colors[i % len(colors)])
# #     ax.set_title('Volume {}'.format(volume))
# #     ax.set_xlabel('Z (mm)')
# #     ax.set_ylabel('X (mm)')
# #     ax.set_zlabel('Y (mm)')
# # fig.suptitle('Hit Locations by Volume', fontsize=16)
# # plt.tight_layout()
# # plt.show()

# # # --- Only volumes 8, 13, 17 ---
# # selected_volumes = [8, 13, 17]
# # hits_filtered = hits[hits.volume_id.isin(selected_volumes)]

# # fig = plt.figure(figsize=(18, 6))
# # for i, volume in enumerate(selected_volumes):
# #     ax = fig.add_subplot(1, 3, i + 1, projection='3d')
# #     v = hits_filtered[hits_filtered.volume_id == volume]
# #     ax.scatter(v.z, v.x, v.y, s=1, color=colors[i % len(colors)])
# #     ax.set_title('Volume {}'.format(volume))
# #     ax.set_xlabel('Z (mm)')
# #     ax.set_ylabel('X (mm)')
# #     ax.set_zlabel('Y (mm)')
# # fig.suptitle('Volumes 8, 13, 17', fontsize=16)
# # plt.tight_layout()
# # plt.show()












# --- Particle trajectories: all particles that pass through the cone ---

# Build a set of valid real particles (non-noise, enough momentum and hits)
HE_particle = particles[particles["particle_id"] != 0].copy()
HE_particle["p_tot"] = np.sqrt(real_particles["px"]**2 + real_particles["py"]**2 + real_particles["pz"]**2)
HE_particle = HE_particle[HE_particle["p_tot"] >= 2]
HE_particle = HE_particle[HE_particle["nhits"] >= 2]


valid_pids = set(HE_particle["particle_id"].values) # all the particle ID's that survive filter








# Method 1: Particles whose MOMENTUM direction points into the cone
mom_vec = HE_particle[["px", "py", "pz"]].to_numpy() # make it numpy
mom_norm = np.maximum(np.linalg.norm(mom_vec, axis=1), 1e-12) # takes momentum vector with a floor of p vector
cos_angle = (mom_vec @ cone_axis) / mom_norm
mom_mask = (cos_angle > 0) & (cos_angle > np.cos(cone_opening_angle))
pids_by_momentum = set(HE_particle[mom_mask]["particle_id"].values) # pick the ones with okey momentum vector. 


# Method 2: Particles that have any HIT physically inside the cone
truth_real = truth[truth["particle_id"] != 0].copy()
T = truth_real[["tx", "ty", "tz"]].to_numpy()
T_vec = T - apex # A vector to all points
T_norm = np.maximum(np.linalg.norm(T_vec, axis=1), 1e-12) #normalize this vector
cos_hit = (T_vec @ cone_axis) / T_norm
hit_in_cone = (cos_hit > 0) & (cos_hit > np.cos(cone_opening_angle))
pids_by_position = set(truth_real[hit_in_cone]["particle_id"].values) & valid_pids

# Combine both sets (union)
all_cone_pids = pids_by_momentum | pids_by_position

print(f"\nParticles by momentum direction: {len(pids_by_momentum)}")
print(f"Particles by hit position in cone: {len(pids_by_position)}")
print(f"Combined (union): {len(all_cone_pids)}")
print(f"Particle IDs: {[int(p) for p in sorted(all_cone_pids)]}")

# Get full trajectories from truth for these particles
truth_full = truth[truth["particle_id"].isin(all_cone_pids)].copy()
truth_full = truth_full.merge(
    particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
)

unique_pids = truth_full["particle_id"].unique()
print(f"Plotting full trajectories for {len(unique_pids)} particles")

# --- 3D trajectory plot ---
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

cmap = plt.cm.tab20
for i, pid in enumerate(unique_pids):
    track = truth_full[truth_full["particle_id"] == pid].copy()
    track["r"] = np.sqrt(track["tx"]**2 + track["ty"]**2 + track["tz"]**2)
    track = track.sort_values("r")

    # Prepend the vertex position
    vx, vy, vz = track.iloc[0][["vx", "vy", "vz"]]
    xs = np.concatenate([[vz], track["tz"].values])
    ys = np.concatenate([[vx], track["tx"].values])
    zs = np.concatenate([[vy], track["ty"].values])

    color = cmap(i % 20)
    ax.plot(xs, ys, zs, '-o', color=color, markersize=3, linewidth=1.2, alpha=0.8)

# Draw cone wireframe
ax.plot_wireframe(z_cone, x_cone, y_cone, color='blue', alpha=0.1, linewidth=0.3)

ax.set_xlabel("Z (mm)")
ax.set_ylabel("X (mm)")
ax.set_zlabel("Y (mm)")
ax.set_title(f"Particle Trajectories in Cone ({len(unique_pids)} particles, p >= 3 MeV)")
plt.tight_layout()
plt.show()





