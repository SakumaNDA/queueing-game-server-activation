import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcdefaults()  # Reset Linux default settings
plt.rcParams['font.family'] = 'MS Gothic'  # Font setting for Windows
# -------------------------

plt.rcParams['pdf.fonttype'] = 42  # Embed TrueType fonts in PDF output
# -------------------------

# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import time

# ==========================================
# Parameter settings for Colab UI (editable via the form on the right)
# ==========================================
MODEL_TYPE = 'MV' #@param["MV", "SV", "ST", "NT"]
N_POLICY = 3 #@param {type:"integer"}
N_customers = 100 #@param {type:"integer"}
run_ABM = 100 #@param {type:"integer"}
at_least_min_join = 1000 #@param {type:"integer"}

# ==========================================
# Settings for time distributions
# ==========================================
# Arrival distribution (Poisson arrivals)
Lambda = 0.9 #@param {type:"number"}
alpha_poisson = np.array([1])
T_poisson = np.array([[-Lambda]])

# General function for generating phase-type distributions
def get_phase_distribution(dist_type, mean, cv=2.0):
    if dist_type == 'Exponential':
        mu = 1.0 / mean
        alpha = np.array([1.0])
        T = np.array([[-mu]])
    elif dist_type == 'Erlang-2':
        # Erlang distribution of order 2 (E_2)
        lam = 2.0 / mean
        alpha = np.array([1.0, 0.0])
        T = np.array([[-lam, lam],
            [ 0.0, -lam]
        ])
    elif dist_type == 'Hyperexponential-2':
        # Hyperexponential distribution of order 2 (H_2): constructed by the balanced means method
        if cv <= 1.0:
            cv = 1.001 # CV must be greater than 1; avoid error
        p = 0.5 * (1.0 + np.sqrt((cv**2 - 1.0) / (cv**2 + 1.0)))
        lam1 = 2.0 * p / mean
        lam2 = 2.0 * (1.0 - p) / mean
        alpha = np.array([p, 1.0 - p])
        T = np.array([[-lam1, 0.0],[ 0.0, -lam2]
        ])
    return alpha, T

# --- Service time distribution settings ---
SERVICE_DISTR = 'Hyperexponential-2' #@param["Exponential", "Erlang-2", "Hyperexponential-2"]
MEAN_SERVICE = 1.0 #@param {type:"number"}
CV_SERVICE = 2.0 #@param {type:"number"}

alpha_S, T_S = get_phase_distribution(SERVICE_DISTR, MEAN_SERVICE, CV_SERVICE)

# --- Vacation/setup time distribution settings ---
VACATION_DISTR = 'Hyperexponential-2' #@param["Exponential", "Erlang-2", "Hyperexponential-2"]
MEAN_VACATION = 1.0 #@param {type:"number"}
CV_VACATION = 2.0 #@param {type:"number"}

alpha_V, T_V = get_phase_distribution(VACATION_DISTR, MEAN_VACATION, CV_VACATION)

# Reward upon service completion
R = 8  #@param {type:"number"}
# Waiting cost per unit time
C = 1  #@param {type:"number"}

# ==========================================
# Constants for the sigmoid-type function
# ==========================================
eta = 300 #@param {type:"integer"}
c1 = (2 - np.log(3))*np.log(3)/(np.log(3) - 1)
c2 = (1/eta)*np.log(c1+np.sqrt(c1**2+4))/2

INF = float('inf')

# Generate one sample (time to absorption) from a phase-type distribution
def sample_phase_type(alpha, T):
    num_states = T.shape[0]
    t = -np.dot(T, np.ones(num_states))

    state = np.random.choice(num_states, p=alpha)
    time_val = 0.0

    while True:
        rate = -T[state, state]
        if rate <= 0:
            break

        time_val += np.random.exponential(1 / rate)

        probs = np.copy(T[state])
        probs[state] = 0
        probs = np.append(probs, t[state])

        total = np.sum(probs)
        if total == 0:
            break
        probs /= total
        next_state = np.random.choice(num_states + 1, p=probs)

        if next_state == num_states:
            break
        else:
            state = next_state

    return time_val


class Customer:
    def __init__(self, idx):
        self.id = idx
        self.J = {}
        self.W = {}
        self.Ju = {0: 0, 1: 0}
        self.tau = {0: INF, 1: INF}
        self.arrival_state = None
        self.arrival_time = None
        self.departure_time = None

    def theta(self, Ju):
        if Ju == 0:
            return 0.0
        return np.exp(c1 / (1 - np.exp(c2 * Ju)))


def arrival_procedure(t, cust, u, queue, undeparted):
    cust.arrival_time = t
    l = len(queue)
    Ju = cust.Ju[u]
    theta_val = cust.theta(Ju)

    joined = False
    if random.uniform(0, 1) <= 1 - theta_val:
        queue.append(cust)
        cust.arrival_state = (u, l)
        joined = True
    else:
        if l <= cust.tau[u]:
            queue.append(cust)
            cust.arrival_state = (u, l)
            joined = True
        else:
            undeparted.append(cust)

    return joined

def departure_procedure(cust, u, l, w, undeparted):
    key = (u, l)
    cust.J[key] = cust.J.get(key, 0) + 1
    cust.W[key] = cust.W.get(key, 0) + w
    cust.Ju[u] += 1

    payoff = R - C * (cust.W[key] / cust.J[key])

    if payoff < 0:
        cust.tau[u] = min(l, cust.tau[u])
    else:
        cust.tau[u] = max(l, cust.tau[u])

    undeparted.append(cust)


def run_abm():
    time_log = []
    undeparted_log = []

    t = 0.0
    U = 0
    queue = []

    all_customers = [Customer(i) for i in range(N_customers)]
    undeparted = list(all_customers)

    over_A = sample_phase_type(alpha_poisson, T_poisson)
    over_S = INF
    if MODEL_TYPE in ['MV', 'SV']:
        over_V = sample_phase_type(alpha_V, T_V)
    else:
        over_V = INF

    start_time = time.time()
    min_join = 0

    # Variable for progress tracking
    last_printed_min_join = -1

    while min_join <= at_least_min_join:
        min_for_u_0 = min(c.Ju[0] for c in all_customers)
        min_for_u_1 = min(c.Ju[1] for c in all_customers)
        min_join = min(min_for_u_0, min_for_u_1)

        # Display progress only when min_join increases
        if min_join > last_printed_min_join:
            elapsed_time = time.time() - start_time
            print(f"Progress: min_join = {min_join} / {at_least_min_join} (Elapsed: {elapsed_time/60:.2f} min)")
            last_printed_min_join = min_join

        if min_join > at_least_min_join:
            break

        delta = min(over_A, over_V, over_S)
        t += delta

        if delta == over_A:
            # (Same logic as before)
            over_V -= over_A
            over_S -= over_A
            over_A = sample_phase_type(alpha_poisson, T_poisson)

            if len(undeparted) > 0:
                idx = random.randint(0, len(undeparted) - 1)
                cust = undeparted.pop(idx)
                joined = arrival_procedure(t, cust, U, queue, undeparted)

                if joined and U == 0:
                    if MODEL_TYPE == 'ST' and over_V == INF:
                        over_V = sample_phase_type(alpha_V, T_V)
                    elif MODEL_TYPE == 'SV' and over_V == INF:
                        U = 1
                        over_S = sample_phase_type(alpha_S, T_S)
                    elif MODEL_TYPE == 'NT' and len(queue) >= N_POLICY + 1:
                        U = 1
                        over_S = sample_phase_type(alpha_S, T_S)

        elif delta == over_V:
            over_A -= over_V
            over_S -= over_V

            if len(queue) == 0:
                if MODEL_TYPE == 'MV':
                    over_V = sample_phase_type(alpha_V, T_V)
                elif MODEL_TYPE == 'SV':
                    over_V = INF
            else:
                U = 1
                over_V = INF
                over_S = sample_phase_type(alpha_S, T_S)

        else:
            over_A -= over_S
            over_V -= over_S

            cust = queue.pop(0)
            u, l = cust.arrival_state
            w = t - cust.arrival_time
            departure_procedure(cust, u, l, w, undeparted)

            if len(queue) >= 1:
                over_S = sample_phase_type(alpha_S, T_S)
            else:
                U = 0
                over_S = INF
                if MODEL_TYPE in ['MV', 'SV']:
                    over_V = sample_phase_type(alpha_V, T_V)
                else:
                    over_V = INF

        if t < 10000:
            time_log.append(t)
            undeparted_log.append(len(undeparted))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Finished! Total Execution time: {elapsed/60:.2f} min.")
    return {i.id: i.tau for i in undeparted + queue}, time_log, undeparted_log

# ==========================================
# Execution and graph output
# ==========================================
tau_0_all = []
tau_1_all = []

def compute_pmf(data):
    count = Counter(data)
    total = sum(count.values())
    x_vals = sorted(count.keys())
    probs = [count[x] / total for x in x_vals]
    return x_vals, probs

for i in range(run_ABM):
    print(f"ABM {i+1}th / {run_ABM} set (Model: {MODEL_TYPE}, Service: {SERVICE_DISTR}, Vacation: {VACATION_DISTR})")
    results, time_log, undeparted_log = run_abm()

    # --- First graph (transition plot) ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_log, undeparted_log, color='navy', linewidth=2)
    plt.xlabel('Time t', fontsize=25)
    plt.ylabel('Number of undeparted', fontsize=25)
    plt.title(f'Transition ({MODEL_TYPE})', fontsize=25)
    plt.tick_params(labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    tau_0_values = [tau[0] for tau in results.values() if tau[0] < INF]
    tau_1_values = [tau[1] for tau in results.values() if tau[1] < INF]
    tau_0_all.extend(tau_0_values)
    tau_1_all.extend(tau_1_values)

x_0, p_0 = compute_pmf(tau_0_all)
x_1, p_1 = compute_pmf(tau_1_all)

# --- Axis range settings ---
if x_0 and x_1:
    x_min, x_max = min(min(x_0), min(x_1)), max(max(x_0), max(x_1))
elif x_0: x_min, x_max = min(x_0), max(x_0)
elif x_1: x_min, x_max = min(x_1), max(x_1)
else: x_min, x_max = 0, 0

x_common = list(range(int(x_min), int(x_max)+1))

def fill_missing(x_vals, p_vals, common_x):
    p_map = dict(zip(x_vals, p_vals))
    return [p_map.get(x, 0) for x in common_x]

p_0_aligned = fill_missing(x_0, p_0, x_common)
p_1_aligned = fill_missing(x_1, p_1, x_common)
p_0_percent = [round(p * 100, 1) for p in p_0_aligned]
p_1_percent = [round(p * 100, 1) for p in p_1_aligned]

# ==========================================
# Second graph (histogram)
# ==========================================
BAR_WIDTH = 0.4
x_pos_0 = np.array(x_common) - BAR_WIDTH / 2
x_pos_1 = np.array(x_common) + BAR_WIDTH / 2

# Enlarge the figure size to accommodate large font sizes
fig, ax = plt.subplots(figsize=(16, 10))

bars_0 = ax.bar(x_pos_0, p_0_percent, width=BAR_WIDTH, label=r'$\tau_0$ (non-operating)', color='skyblue', alpha=0.8)
bars_1 = ax.bar(x_pos_1, p_1_percent, width=BAR_WIDTH, label=r'$\tau_1$ (in service)', color='salmon', alpha=0.8)

# --- Percentage labels on top of bars ---
# Increased fontsize to 26
# Changed rotation to 60 to avoid overlapping
# Increased xytext vertical offset to 12 to avoid proximity to bars
for bar in bars_0:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=26,
                    fontweight='bold',
                    rotation=60)

for bar in bars_1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=26,
                    fontweight='bold',
                    rotation=60)

# Labels and title font sizes
ax.set_title(f"Frequency distribution of joining thresholds ({MODEL_TYPE})", fontsize=32)
ax.set_xlabel(r'Threshold $\tau$', fontsize=35)
ax.set_ylabel('Percentage (%)', fontsize=35)

# Tick label font size
ax.tick_params(axis='both', labelsize=26)
ax.set_xticks(x_common)

# Legend font size
ax.legend(prop={'size': 28}, loc='upper right')

# Add extra space at the top of the graph for large percentage labels
max_height = max(p_0_percent + p_1_percent) if (p_0_percent + p_1_percent) else 100
ax.set_ylim(0, max_height * 1.5)

ax.grid(True, axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines only
plt.tight_layout()

# ==========================================
# File saving (PDF and text)
# ==========================================

# 1. Set file names (common for PDF and text output)
# Example: ABM_MV_Exponential.pdf / .txt
base_filename = f'ABM_{MODEL_TYPE}_{SERVICE_DISTR}'
pdf_filename = f'{base_filename}.pdf'
txt_filename = f'{base_filename}.txt'

# 2. Save PDF graph
plt.savefig(pdf_filename)
plt.show()
print(f"Saved graph: {pdf_filename}")

# 3. Write distribution data to text file
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write(f"Simulation Results: {MODEL_TYPE}\n")
    f.write(f"Probability Distribution: {SERVICE_DISTR}\n")
    f.write("="*50 + "\n\n")

    # Distribution of tau_0
    f.write("--- Distribution of tau_0 (non-operating) ---\n")
    f.write("Threshold Value\tPercentage (%)\n")
    for val, prob in zip(x_common, p_0_percent):
        if prob > 0:
            f.write(f"{val}\t\t{prob:.1f}%\n")

    f.write("\n" + "-"*40 + "\n\n")

    # Distribution of tau_1
    f.write("--- Distribution of tau_1 (in service) ---\n")
    f.write("Threshold Value\tPercentage (%)\n")
    for val, prob in zip(x_common, p_1_percent):
        if prob > 0:
            f.write(f"{val}\t\t{prob:.1f}%\n")

print(f"Saved text data: {txt_filename}")
