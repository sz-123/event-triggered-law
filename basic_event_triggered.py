import numpy as np 
import control as ctrl
import copy
import matplotlib.pyplot as plt


L = np.array([[3.4, -3.4, 0., 0.],
              [-3.4, 9.8, -2.1, -4.3],
              [0., -2.1, 3.2, -1.1],
              [0., -4.3, -1.1, 5.4]])

DIM = L.shape[0]

N_iter = 2000 # dt = 0.001
dt = 0.001

def calculate_q_i(x_hat, i, L):
    sum = 0
    for j in range(DIM):
        sum += L[i, j] * (x_hat[j] - x_hat[i]) * (x_hat[j] - x_hat[i])
    return - 1. / 2. * sum

def basic_trigger_condition(e, q_i, i, L, sigma=0.5):
    return True if e[i] * e[i] - sigma / (2 * L[i,i]) * q_i > 0 else False

def update_control_input(x_hat, i, L):
    u_i = 0
    for j in range(DIM):
        u_i -= L[i, j] * x_hat[j]
    return u_i

# basic event-triggered law
x_traj = np.zeros((DIM, N_iter))
triggering_times = np.zeros((DIM, N_iter))

x_average_consensus = 3.8044
x = np.array([[6.2945], [8.1158], [-7.4603], [8.1675]])
x_hat = np.array([[6.2945], [8.1158], [-7.4603], [8.1675]])
u = np.zeros((DIM, 1))
for i in range(DIM):
    u[i] = update_control_input(x_hat, i, L) # init control input

for k in range(N_iter):
    e = x_hat - x
    for i in range(DIM):
        q_i = calculate_q_i(x_hat, i, L)
        if basic_trigger_condition(e, q_i, i, L):
            x_hat[i] = x[i]
            triggering_times[i, k] = 1
            for m in range(DIM):
                u[m] = update_control_input(x_hat, m, L)
    x = x + dt * u
    x_traj[:, k] = x.flatten()

# Plotting subplots1: traj; subplots2: triggering times as dots
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Trajectories')
ax2.set_title('Triggering Times')
for i in range(DIM):
    ax1.plot(x_traj[i, :], label='x' + str(i+1))
    ax2.plot(triggering_times[i, :] * 0.2 * (i+1), '*', label='x' + str(i+1))
ax1.legend()
ax2.legend()
ax1.set_xlabel('Time')
ax2.set_xlabel('Time')
ax1.set_ylabel('Value')
ax2.set_ylabel('Triggering Times')
# plot the consensus value in ax1
ax1.axhline(y=x_average_consensus, color='r', linestyle='--', label='Consensus Value')
ax1.legend()

plt.show()


        

    


