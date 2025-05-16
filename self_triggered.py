import numpy as np 
import control as ctrl
import copy
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar


L = np.array([[3.4, -3.4, 0., 0.],
              [-3.4, 9.8, -2.1, -4.3],
              [0., -2.1, 3.2, -1.1],
              [0., -4.3, -1.1, 5.4]])

DIM = L.shape[0]

# get the minimum positive eigenvalue of L
eigvals = np.linalg.eigvals(L)
print('Eigenvalues of L:', eigvals)
min_eigval = np.min(eigvals[eigvals > 0])
RHO = min_eigval
print('Minimum positive eigenvalue of L:', min_eigval)
ALPHA = 10.
BETA = 1.
CHI = 10.
THETA = 1.
Kn = np.eye(DIM) - np.ones((DIM, DIM)) / DIM
 
X_INIT = np.array([[6.2945], [8.1158], [-7.4603], [8.2675]])

def lyapunov_function(x):
    return 0.5 * np.dot(x.T, np.dot(Kn, x))

k5 = DIM * ALPHA * ALPHA / (RHO - BETA)
k4 = lyapunov_function(X_INIT) - k5

def f_function(t):
    return 2 * np.sqrt(k4 * np.exp(- RHO * t) + k5 * np.exp(- BETA * t))

def integrand(s, i, j, L):
        return (ALPHA / L[i,i] + ALPHA / L[j,j]) * np.exp(- BETA / 2 * s) + f_function(s)

def g_i_rhs(t, i, L):
        return ALPHA / np.sqrt(L[i,i]) * np.exp( - BETA / 2 * t)

def solving_for_the_next_trigger_time(trigger_time, next_trigger_time, x_hat, i, L):
    # t_history = np.linspace(0, 5, int(5 / dt))

    # def k_j(t):
    #     return np.searchsorted(t_history, t) - 1
    
    def g_i(t):
        sum = 0
        t_vec = np.ones((DIM, 1)) * t
        i_trigger_time_vec = np.ones((DIM, 1)) * trigger_time[i]
        tij1 = np.minimum(t_vec, next_trigger_time)
        tij2 = np.maximum(t_vec, next_trigger_time)

        for j in range(DIM):
            sum += np.abs(L[i, j] * (tij1[j] - trigger_time[i]) * (x_hat[j] - x_hat[i]))

        for j in range(DIM):
            if j != i:
                # Create a wrapper function that only takes s as input
                integrand_wrapper = lambda s: integrand(s, i, j, L)

                integral_value, _ = quad(integrand_wrapper, next_trigger_time[j], tij2[j])
                sum += - L[i, j] * integral_value
        
        rhs = g_i_rhs(t, i, L)

        return sum - rhs
    # Use root_scalar to find the root of g_i(t) = 0
    result = root_scalar(g_i, bracket=[trigger_time[i], trigger_time[i] + 5], method='brentq')

    # round result.root to 0.001
    root_round = np.round(result.root, 3)
    
    return root_round

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

N_iter = 2000 # dt = 0.001
dt = 0.001
# basic event-triggered law
x_traj = np.zeros((DIM, N_iter))
triggering_time = np.zeros((DIM, 1))
next_triggering_time = np.zeros((DIM, 1))
# Create a dictionary to store triggering times for each agent
triggering_dict = {i: [] for i in range(DIM)}

x_average_consensus = 3.8044
x = np.array([[6.2945], [8.1158], [-7.4603], [8.2675]])
x_hat = np.array([[6.2945], [8.1158], [-7.4603], [8.2675]])
u = np.zeros((DIM, 1))
for i in range(DIM):
    u[i] = update_control_input(x_hat, i, L) # init control input

for k in range(N_iter):
    for i in range(DIM):
        if next_triggering_time[i] == k:
            # save the next triggering time to the dictionary
            triggering_dict[i].append(next_triggering_time[i])
            triggering_time[i] = next_triggering_time[i]
            x_hat[i] = x[i]
            u[i] = update_control_input(x_hat, i, L)
            next_triggering_time[i] = solving_for_the_next_trigger_time(triggering_time, next_triggering_time, x_hat, i, L)
            # for m in range(DIM):
            #     u[m] = update_control_input(x_hat, m, L)
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
    # plot the triggering times as dots, if it's not empty
    if triggering_dict[i]:
        ax2.plot(triggering_dict[i], (i+1) * 0.2, 'o', label='Triggering Time ' + str(i+1))
    else:
        ax2.plot([], [], 'o', label='Triggering Time ' + str(i+1))  # Empty plot for legend
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


        

    


