# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

# ==========================================
# Parameter settings for Colab UI
# ==========================================
MODEL_TYPE    = 'NT'      #@param["MV", "SV", "ST", "NT"]
N_POLICY      = 3         #@param {type:"integer"}
M_LIMIT       = 15        #@param {type:"integer"}
Lambda        = 0.9       #@param {type:"number"}

SERVICE_DISTR = 'Hyperexponential-2'        #@param["Exponential", "Erlang-2", "Hyperexponential-2"]
MEAN_SERVICE  = 1.0               #@param {type:"number"}
CV_SERVICE    = 2.0               #@param {type:"number"}

VACATION_DISTR = 'Hyperexponential-2'       #@param["Exponential", "Erlang-2", "Hyperexponential-2"]
MEAN_VACATION  = 1.0              #@param {type:"number"}
CV_VACATION    = 2.0              #@param {type:"number"}

R = 8.0   #@param {type:"number"}
C = 1.0   #@param {type:"number"}

# ==========================================
# Utility for generating phase-type (PH) distributions
# ==========================================
def get_ph_dist(dist_type, mean, cv=2.0):
    if dist_type == 'Exponential':
        mu    = 1.0 / mean
        alpha = np.array([1.0])
        T     = np.array([[-mu]])
    elif dist_type == 'Erlang-2':
        lam   = 2.0 / mean
        alpha = np.array([1.0, 0.0])
        T     = np.array([[-lam, lam], [0.0, -lam]])
    elif dist_type == 'Hyperexponential-2':
        if cv <= 1.0: cv = 1.001
        p     = 0.5 * (1.0 + np.sqrt((cv**2 - 1.0) / (cv**2 + 1.0)))
        lam1  = 2.0 * p       / mean
        lam2  = 2.0 * (1-p)   / mean
        alpha = np.array([p, 1.0-p])
        T     = np.array([[-lam1, 0.0], [0.0, -lam2]])
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
    return alpha, T

# ==========================================
# Stationary distribution computation engine
#
# [Assumption] Service time distribution is independent of the number
#              of customers: U_n = X (common phase-type distribution)
#
# Correspondence to manuscript notation:
#   nu_L[n] : Unnormalized stationary measure nu_{L_n} for state (s=0, n)
#             = stationary measure for inactive server (s=0) with n waiting customers
#   nu_U[n] : Unnormalized stationary measure nu_{U_n} for state (s=1, n)
#             = stationary measure for active server (s=1) with n waiting customers
#             Size is always dim_S (= T_S.shape[0])
#
# Note: In the manuscript, queue length l = |Q| (including customer in service)
#       is used. In the code below, n denotes the number of waiting customers
#       (excluding any customer in service).
#       State (s=0, n) corresponds to (0, l) in the manuscript with l = n,
#       State (s=1, n) corresponds to (1, l) in the manuscript with l = n+1.
# ==========================================
def solve_mam(M, lamb, p, q, alpha_S, T_S, beta_V, U_V, model_type, N_pol):
    dim_S = T_S.shape[0]
    dim_V = U_V.shape[0]
    t_S   = (-T_S) @ np.ones(dim_S)   # Service completion rate vector t = (-T) 1
    u_V   = (-U_V) @ np.ones(dim_V)   # Vacation completion rate vector u = (-U) 1

    nu_L = [None] * (M + 1)
    nu_U = [None] * (M + 1)

    # --------------------------------------------------
    # Denominator matrix for the active (service) state (common structure for all levels)
    #   A_U(eff_q) = T_S - eff_q*lamb*I + eff_q*lamb * 1 * alpha_S^T
    # Corresponds to -(Q_{U_n U_n} + Q_{U_n U_{n+1}} 1 alpha_S^T) in the manuscript.
    # Used as inv(-A_U).
    # --------------------------------------------------
    def make_A_U(eff_q):
        return (T_S - eff_q * lamb * np.eye(dim_S)
                + eff_q * lamb * np.outer(np.ones(dim_S), alpha_S))

    # ==================================================
    # Computation of nu_L[n] for each model
    # (corresponds to Proposition, Steps 1--2 in the manuscript)
    # ==================================================

    if model_type == 'MV':
        # Step 1: nu_L[0] (manuscript Eq. (2.1), inactive state at level 0)
        # A_{L0} = Q_{L0L0} + Q_{L0L1} 1 beta^T + Q_{L0U0} 1 beta^T
        #        = (U + u beta^T - lamb p0 I) + lamb p0 * 1 * beta^T + O
        A_L0 = (U_V + np.outer(u_V, beta_V)
                - lamb * p[0] * np.eye(dim_V)
                + lamb * p[0] * np.outer(np.ones(dim_V), beta_V))
        vals, vecs = np.linalg.eig(A_L0.T)
        nu_L[0] = np.real(vecs[:, np.argmin(np.abs(vals))])
        nu_L[0] /= np.sum(nu_L[0])

        # Step 2: nu_L[n] (n >= 1) (manuscript Eq. (2.2))
        # nu_L[n] = nu_L[n-1] @ Q_{L_{n-1}L_n} @ inv(-Q_{L_n L_n})
        #         = nu_L[n-1] * (lamb p_{n-1}) @ inv(-(U - lamb p_n I))
        for n in range(1, M + 1):
            nu_L[n] = (nu_L[n-1] * (lamb * p[n-1])
                       @ inv(-(U_V - lamb * p[n] * np.eye(dim_V))))

    elif model_type == 'SV':
        # L_0 = {Delta} ∪ Y, size = 1 + dim_V
        # Step 1: nu_L[0] (inactive state at level 0, including ready state Delta)
        Q_L0L0 = np.block([
            [-lamb * p[0],                   np.zeros((1, dim_V))          ],
            [u_V.reshape(-1, 1),             U_V - lamb*p[0]*np.eye(dim_V) ]
        ])
        sigma_L0    = np.concatenate(([0.0], beta_V))                        # sigma_{L_0} = (0, beta^T)
        Q_L0L1_1    = np.concatenate(([0.0], lamb * p[0] * np.ones(dim_V))) # Q_{L0L1} 1
        Q_L0U0_1    = np.concatenate(([lamb * p[0]], np.zeros(dim_V)))       # Q_{L0U0} 1
        A_L0 = (Q_L0L0
                + np.outer(Q_L0L1_1, sigma_L0)
                + np.outer(Q_L0U0_1, sigma_L0))
        vals, vecs = np.linalg.eig(A_L0.T)
        nu_L[0] = np.real(vecs[:, np.argmin(np.abs(vals))])
        nu_L[0] /= np.sum(nu_L[0])

        # Step 2: nu_L[n] (n >= 1)  --- L_n = Y (dimension dim_V)
        # n=1: Q_{L0L1} = [[0^T], [lamb p0 I]]  (size (1+dim_V) x dim_V)
        # n>=2: Q_{L_{n-1}L_n} = lamb p_{n-1} I
        for n in range(1, M + 1):
            if n == 1:
                Q_prev = np.vstack([np.zeros((1, dim_V)),
                                    lamb * p[0] * np.eye(dim_V)])
            else:
                Q_prev = lamb * p[n-1] * np.eye(dim_V)
            nu_L[n] = nu_L[n-1] @ Q_prev @ inv(-(U_V - lamb * p[n] * np.eye(dim_V)))

    elif model_type == 'ST':
        # L_0 = {Delta} (scalar, idle state), L_n = Y (n>=1, setup phase)
        nu_L[0] = np.array([1.0])   # Idle state: stationary measure normalized to 1
        for n in range(1, M + 1):
            if n == 1:
                # Q_{L0L1} = lamb p0 * beta^T  (1 x dim_V)
                # Setup starts upon arrival of the first customer, with initial distribution beta^T
                Q_prev = lamb * p[0] * beta_V.reshape(1, -1)
            else:
                Q_prev = lamb * p[n-1] * np.eye(dim_V)
            nu_L[n] = nu_L[n-1] @ Q_prev @ inv(-(U_V - lamb * p[n] * np.eye(dim_V)))

    elif model_type == 'NT':
        # L_n = {Delta} (scalar), n = 0, ..., N (idle state)
        # Cut balance: nu_L[n-1] * lamb * p[n-1] = nu_L[n] * lamb * p[n]
        # (inflow from level n-1 to n = outflow from level n to n+1)
        nu_L[0] = np.array([1.0])
        for n in range(1, N_pol + 1):
            denom    = lamb * p[n] if p[n] > 0 else 1e-10
            nu_L[n]  = nu_L[n-1] * (lamb * p[n-1]) / denom
        # For N < n <= M, there is no inactive state (L_n is empty for n > N in NT model)
        for n in range(N_pol + 1, M + 1):
            nu_L[n] = np.array([0.0])

    # ==================================================
    # Computation of nu_U[n] (common to all models,
    # corresponds to Proposition, Step 3 in the manuscript)
    #
    # Since service times are independent, U_n = X (dimension dim_S)
    #
    # Manuscript Eq. (2.3):
    #   nu_U[n] = { nu_U[n-1] Q_{U_{n-1}U_n}
    #             + nu_L[n]   Q_{L_n U_n}
    #             + nu_L[n+1] (Q_{L_{n+1}U_n} + Q_{L_{n+1}L_{n+2}} 1 alpha_S^T)
    #             } @ inv(-(Q_{U_n U_n} + Q_{U_n U_{n+1}} 1 alpha_S^T))
    # ==================================================
    for n in range(M + 1):
        eff_q = q[n] if n < M else 0.0
        A_U   = make_A_U(eff_q)

        # Term 1: nu_U[n-1] Q_{U_{n-1}U_n}
        # Q_{U_{n-1}U_n} = lamb * q[n-1] * I (transition due to arrival)
        if n == 0:
            term_U_prev = np.zeros(dim_S)
        else:
            term_U_prev = nu_U[n-1] * (lamb * q[n-1])

        # Term 2: nu_L[n] Q_{L_n U_n}
        #
        # Structure of Q_{L_n U_n} in the manuscript (varies by model):
        #   MV          : Q_{L_n U_n} = O (level decreases when vacation ends)
        #   SV (n=0)    : Only the Delta row of Q_{L0 U0} is nonzero
        #                 Service starts upon arrival at ready state Delta
        #                 => nu_L[0][0] * lamb * p[0] * alpha_S
        #   ST          : Q_{L0 U0} = 0^T (no direct transition from idle to U_0)
        #   NT (n=N)    : Q_{L_N U_N} = lamb p_N * (0^T, alpha_S^T)
        #                 Service starts when the N-th customer arrives
        #                 => nu_L[N] * lamb * p[N] * alpha_S
        if model_type == 'SV' and n == 0:
            term_L_n = nu_L[0][0] * lamb * p[0] * alpha_S
        elif model_type == 'NT' and n == N_pol:
            term_L_n = float(nu_L[N_pol]) * lamb * p[N_pol] * alpha_S
        else:
            term_L_n = np.zeros(dim_S)

        # Term 3: nu_L[n+1] (Q_{L_{n+1}U_n} + Q_{L_{n+1}L_{n+2}} 1 alpha_S^T)
        #
        # MV/SV/ST:
        #   Q_{L_{n+1}U_n} = u_V alpha_S^T (service starts when vacation ends)
        #     => (nu_L[n+1] @ u_V) * alpha_S
        #   Q_{L_{n+1}L_{n+2}} 1 alpha_S^T = lamb p_{n+1} * 1 * alpha_S^T
        #     => sum(nu_L[n+1]) * lamb * p_{n+1} * alpha_S
        #
        # NT:
        #   Q_{L_{n+1}U_n} = 0 (service does not start until threshold N is exceeded)
        #   Q_{L_{n+1}L_{n+2}} 1 alpha_S^T = lamb p_{n+1} (when n+1 < N)
        #     => float(nu_L[n+1]) * lamb * p[n+1] * alpha_S
        if n >= M or nu_L[n+1] is None or np.all(nu_L[n+1] == 0):
            term_L_next = np.zeros(dim_S)
        elif model_type == 'NT':
            if n + 1 < N_pol:
                term_L_next = float(nu_L[n+1]) * lamb * p[n+1] * alpha_S
            else:
                term_L_next = np.zeros(dim_S)
        else:
            # MV/SV/ST
            jump_vac    = (nu_L[n+1] @ u_V) * alpha_S       # Vacation (or setup) completion
            jump_arr    = (np.sum(nu_L[n+1]) * lamb * p[n+1] * alpha_S
                           if n + 1 < M else np.zeros(dim_S))  # Transition due to arrival
            term_L_next = jump_vac + jump_arr

        nu_U[n] = (term_U_prev + term_L_n + term_L_next) @ inv(-A_U)

    # Normalize and return (corresponding to stationary distributions pi_{L_n}, pi_{U_n})
    pi_L = [nu / np.sum(nu) if (nu is not None and np.sum(nu) > 1e-12)
            else None for nu in nu_L]
    pi_U = [nu / np.sum(nu) if (nu is not None and np.sum(nu) > 1e-12)
            else None for nu in nu_U]
    return pi_L, pi_U


# ==========================================
# Expected utility computation
#
# Correspondence to manuscript notation:
#   n (code) = number of waiting customers (excluding customer in service)
#   State (s=0, n) corresponds to (0, l) in the manuscript with l = n
#   State (s=1, n) corresponds to (1, l) in the manuscript with l = n+1
#
#   u_Ln: Expected utility U_{(0,l)} (l=n) for an arriving customer
#         who observes state (s=0, n)
#   u_Un: Expected utility U_{(1,l)} (l=n+1) for an arriving customer
#         who observes state (s=1, n)
# ==========================================
def compute_utility(n, M, lamb, R, C, alpha_S, T_S, beta_V, U_V,
                    model_type, N_pol, p, q):
    pi_L, pi_U = solve_mam(M, lamb, p, q, alpha_S, T_S,
                            beta_V, U_V, model_type, N_pol)

    dim_S  = T_S.shape[0]
    E_S    = inv(-T_S) @ np.ones(dim_S)   # (-T)^{-1} 1: mean service time vector
    mu_inv = float(alpha_S @ E_S)          # mu^{-1}: mean service time (scalar)

    # --------------------------------------------------
    # w_{(1,l)} (l = n+1): Conditional expected sojourn time for an arriving
    # customer who observes state (s=1, n)
    # Manuscript Eq. (4.2):
    #   w_{(1,l)} = pi_{U_{l-1}}^T (-T)^{-1} 1 + l * mu^{-1}
    # Code correspondence: pi_U[n] = tilde{pi}_{U_n}, l = n+1
    #   w_{(1,n+1)} = pi_U[n]^T (-T)^{-1} 1 + (n+1) * mu^{-1}
    # --------------------------------------------------
    if pi_U[n] is not None:
        w_Un = float(pi_U[n] @ E_S) + (n + 1) * mu_inv
        u_Un = R - C * w_Un   # U_{(1,l)} (l = n+1)
    else:
        u_Un = None

    # --------------------------------------------------
    # w_{(0,l)} (l = n): Conditional expected sojourn time for an arriving
    # customer who observes state (s=0, n)
    # Manuscript Eq. (4.1):
    #   w_{(0,l)} = pi_{L_l}^T (-U)^{-1} 1 + (l+1) * mu^{-1}
    # Code correspondence: pi_L[n] = tilde{pi}_{L_n}, l = n
    #   w_{(0,n)} = pi_L[n]^T (-U)^{-1} 1 + (n+1) * mu^{-1}
    # --------------------------------------------------
    E_V = inv(-U_V) @ np.ones(U_V.shape[0])   # (-U)^{-1} 1: mean vacation time vector

    if model_type == 'NT':
        if n <= N_pol:
            # NT model, manuscript Eq. (4.5):
            # w_{(0,l)} = sum_{k=l+1}^{N} 1/(lambda p_k) + (l+1) * mu^{-1}
            # Code correspondence: n = l
            w_Ln = (sum(1.0 / (lamb * p[k]) if p[k] > 0 else 1e10
                        for k in range(n + 1, N_pol + 1))
                    + (n + 1) * mu_inv)
            u_Ln = R - C * w_Ln   # U_{(0,l)} (l = n)
        else:
            u_Ln = None

    elif model_type == 'ST':
        if n == 0:
            # ST model, manuscript Eq. (4.3):
            # w_{(0,0)} = beta^T (-U)^{-1} 1 + mu^{-1} (constant, independent of strategy)
            w_Ln = float(beta_V @ E_V) + mu_inv
        else:
            # ST model, n >= 1 (during setup):
            # w_{(0,l)} = pi_{L_l}^T (-U)^{-1} 1 + (l+1) * mu^{-1} (l = n)
            if pi_L[n] is None:
                return None, u_Un
            w_Ln = float(pi_L[n] @ E_V) + (n + 1) * mu_inv
        u_Ln = R - C * w_Ln   # U_{(0,l)} (l = n)

    elif model_type == 'SV':
        if n == 0:
            # SV model, manuscript Eq. (4.3):
            # w_{(0,0)} = pi_{L_0}(Y)^T (-U)^{-1} 1 + mu^{-1}
            # pi_L[0] = (pi_Delta, pi_Y^T), Y component is [1:]
            if pi_L[0] is None:
                return None, u_Un
            w_Ln = float(pi_L[0][1:] @ E_V) + mu_inv
        else:
            # SV model, n >= 1 (during vacation):
            # w_{(0,l)} = pi_{L_l}^T (-U)^{-1} 1 + (l+1) * mu^{-1} (l = n)
            if pi_L[n] is None:
                return None, u_Un
            w_Ln = float(pi_L[n] @ E_V) + (n + 1) * mu_inv
        u_Ln = R - C * w_Ln   # U_{(0,l)} (l = n)

    else:  # MV
        # MV model, manuscript Eq. (4.1):
        # w_{(0,l)} = pi_{L_l}^T (-U)^{-1} 1 + (l+1) * mu^{-1} (l = n)
        if pi_L[n] is None:
            return None, u_Un
        w_Ln = float(pi_L[n] @ E_V) + (n + 1) * mu_inv
        u_Ln = R - C * w_Ln   # U_{(0,l)} (l = n)

    return u_Ln, u_Un

########################################
#### Compute equilibrium thresholds ####
########################################
def run_equilibrium_analysis():
    alpha_S, T_S = get_ph_dist(SERVICE_DISTR,  MEAN_SERVICE,  CV_SERVICE)
    beta_V,  U_V = get_ph_dist(VACATION_DISTR, MEAN_VACATION, CV_VACATION)

    # Initial values: all customers join
    # (evaluation starts with p_k=1, q_k=1 following the procedure in the manuscript)
    p_level = [1.0] * (M_LIMIT + 1)
    q_level = [1.0] * (M_LIMIT + 1)

    if MODEL_TYPE == 'NT':
        # NT model: determination of tau_0^*
        # (corresponds to Corollary 4.5, 4.6 in the manuscript)
        # U_{(0,l)} is a constant independent of tau_0
        # (manuscript Corollary 4.4)
        # Evaluate p_level[n] for all 0 <= n <= N, then determine tau_0^*
        for n in range(N_POLICY + 1):
            u_Ln, _ = compute_utility(n, M_LIMIT, Lambda, R, C,
                                      alpha_S, T_S, beta_V, U_V,
                                      MODEL_TYPE, N_POLICY, p_level, q_level)
            p_level[n] = 1.0 if (u_Ln is not None and u_Ln >= 0) else 0.0
        # No inactive state for n > N in the NT model
        for n in range(N_POLICY + 1, M_LIMIT + 1):
            p_level[n] = 0.0
        # tau_0^* = maximum n satisfying U_{(0,l)} >= 0 (l = n)
        tau_0 = max((n for n in range(N_POLICY + 1) if p_level[n] == 1.0),
                    default=-1)
        print(f"tau_0^* = {tau_0}")

        # Determination of tau_1^*
        # (corresponds to Corollary 4.6 in the manuscript)
        # Evaluate U_{(1,l)} with q_k=1 (0 <= k <= n) at each step
        # tau_1^* is determined when U_{(1,l)} < 0 first occurs
        # tau_1^* = 1 + (maximum n satisfying u_Un >= 0)
        tau_1 = 0
        for n in range(M_LIMIT + 1):
            _, u_Un = compute_utility(n, M_LIMIT, Lambda, R, C,
                                      alpha_S, T_S, beta_V, U_V,
                                      MODEL_TYPE, N_POLICY, p_level, q_level)
            if u_Un is not None and u_Un >= 0:
                tau_1 = n + 1   # tau_1^* = l = n+1 (l = n+1 when s=1)
            else:
                # Determined at the first n where U_{(1,l)} < 0
                for k in range(n, M_LIMIT + 1):
                    q_level[k] = 0.0
                break
        print(f"tau_1^* = {tau_1}")

    else:
        # MV/SV/ST: determination of tau_0^*
        # (corresponds to Corollary 4.1 in the manuscript)
        # Evaluate U_{(0,l)} with p_k=1 (0 <= k <= n) at each step
        # If U_{(0,l)} >= 0, then tau_0^* >= l = n is confirmed
        # tau_0^* = l-1 = n-1 is determined when U_{(0,l)} < 0 first occurs
        tau_0 = -1
        for n in range(M_LIMIT + 1):
            u_Ln, _ = compute_utility(n, M_LIMIT, Lambda, R, C,
                                      alpha_S, T_S, beta_V, U_V,
                                      MODEL_TYPE, N_POLICY, p_level, q_level)
            if u_Ln is not None and u_Ln >= 0:
                tau_0 = n   # tau_0^* = l = n (l = n when s=0)
            else:
                # Determined at the first n where U_{(0,l)} < 0
                for k in range(n, M_LIMIT + 1):
                    p_level[k] = 0.0
                break
        print(f"tau_0^* = {tau_0}")

        # Determination of tau_1^*
        # (corresponds to Corollary 4.1 in the manuscript)
        # Evaluate U_{(1,l)} with q_k=1 (0 <= k <= n) at each step
        # tau_1^* is determined when U_{(1,l)} < 0 first occurs
        # tau_1^* = 1 + (maximum n satisfying u_Un >= 0)
        tau_1 = 0
        for n in range(M_LIMIT + 1):
            _, u_Un = compute_utility(n, M_LIMIT, Lambda, R, C,
                                      alpha_S, T_S, beta_V, U_V,
                                      MODEL_TYPE, N_POLICY, p_level, q_level)
            if u_Un is not None and u_Un >= 0:
                tau_1 = n + 1   # tau_1^* = l = n+1 (l = n+1 when s=1)
            else:
                # Determined at the first n where U_{(1,l)} < 0
                for k in range(n, M_LIMIT + 1):
                    q_level[k] = 0.0
                break
        print(f"tau_1^* = {tau_1}")

    # --------------------------------------------------
    # Plotting
    # Horizontal axis: queue length l = |Q| observed by arriving customer
    #   (including customer in service)
    #   s=0 (inactive): l = n (number of waiting customers n = l)
    #   s=1 (active):   l = n+1 (n waiting customers + 1 in service)
    # --------------------------------------------------
    # Horizontal axis for s=0: l = 0, 1, ..., M_LIMIT (corresponding to l = n)
    l_axis_L   = np.arange(M_LIMIT + 1)
    # Horizontal axis for s=1: l = 1, 2, ..., M_LIMIT+1 (corresponding to l = n+1)
    l_axis_U   = np.arange(1, M_LIMIT + 2)
    p_strategy = [int(p_level[n]) for n in range(M_LIMIT + 1)]
    q_strategy = [int(q_level[n]) for n in range(M_LIMIT + 1)]

    model_label = (f"{MODEL_TYPE} (N={N_POLICY})"
                   if MODEL_TYPE == 'NT' else MODEL_TYPE)

    plt.rcParams.update({
        'font.size':       21,
        'axes.titlesize':  21,
        'axes.labelsize':  21,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 21,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    def draw_steps(ax, x_vals, y_vals, title, color, marker, x_limit=None):
        if x_limit is not None:
            idx    = [i for i in range(len(x_vals)) if x_vals[i] <= x_limit]
            x_vals = [x_vals[i] for i in idx]
            y_vals = [y_vals[i] for i in idx]
        ax.plot(x_vals, y_vals, marker=marker, linestyle='',
                color=color, markersize=10, zorder=5)
        for i in range(len(x_vals)):
            x0 = max(x_vals[i] - 0.5, -0.3)
            ax.hlines(y_vals[i], x0, x_vals[i] + 0.5,
                      color=color, linewidth=3, alpha=0.7)
            if i < len(y_vals) - 1 and y_vals[i] != y_vals[i+1]:
                ax.vlines(x_vals[i] + 0.5, 0, 1,
                          color=color, linestyles=':', linewidth=2)
        ax.set_title(title, fontsize=21)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Balk (0)', 'Join (1)'])
        ax.grid(True, axis='x', alpha=0.2)

    # s=0 (inactive): horizontal axis is queue length l (= number of waiting customers n)
    l_limit = N_POLICY if MODEL_TYPE == 'NT' else M_LIMIT
    draw_steps(ax1, list(l_axis_L), p_strategy,
               f"Nash Equilibrium: Server inactive ($s=0$), {model_label}, "
               f"$\\tau^{{*}}_0 = {tau_0}$",
               "deepskyblue", "o", x_limit=l_limit)

    # s=1 (active): horizontal axis is queue length l (= n waiting customers + 1 in service)
    draw_steps(ax2, list(l_axis_U), q_strategy,
               f"Nash Equilibrium: Server active ($s=1$), {model_label}, "
               f"$\\tau^{{*}}_1 = {tau_1}$",
               "hotpink", "s")

    ax2.set_xlabel(
        "Queue length $l = |Q|$ (incl. customer in service) \n observed upon arrival",
        fontsize=21)
    ax2.set_xlim(-0.5, M_LIMIT + 0.5)
    plt.tight_layout()

    distr_label = SERVICE_DISTR.replace('-2', '')
    filename = f"LDQBD_{MODEL_TYPE}_{distr_label}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")

    plt.show()

run_equilibrium_analysis()
