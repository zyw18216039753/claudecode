"""
IEEE 33-bus IGWO Voltage & Reactive Power Optimization
======================================================
对比算法: ABC (人工蜂群), WOA (鲸鱼算法), Standard GWO, Improved GWO
IEEE 33-bus 径向配电网, 含3台直驱风机 + SVG + 电容器组 + OLTC

IGWO 改进:
  1. 非线性收敛因子: a(m) = a0 - lambda * (m/M)^k
  2. delta狼融合变异
  3. 不等权位置更新
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os.path as _osp

# ---- Figura: IEEE publication-quality figure setup ----
import sys as _sys
_sys.path.insert(0, r"C:\Users\18771\.claude\plugins\cache\figura\figura\0.4.0\skills\figura\scripts")
import pubstyle, colors, export
pubstyle.apply(venue="ieee")
colors.apply_cycle()
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei', 'sans-serif']

# ============================================================================
# IEEE 33-bus test system
# ============================================================================

class IEEE33BusSystem:
    """
    IEEE 33-bus radial distribution system
    SB=10MVA, VB=12.66kV, Z_base=VB^2/SB=16.0276Ω

    Device placement:
      Node 0: 变电站 (slack + OLTC调节)
      Node 17: WT1 (1.2MW直驱风机) — 主馈线末端
      Node 24: WT2 (1.2MW直驱风机) — 支线末端
      Node 32: WT3 (0.8MW直驱风机) — 支线末端
      Node 14: SVG ±2.5MVar — WT1附近
      Node 7:  电容器组 0~1.5MVar (5档×0.3MVar) — 分支点
    """

    def __init__(self):
        self.SB = 10.0
        self.VB = 12.66
        self.n_nodes = 33
        self.n_time = 24

        Z_base = self.VB**2 / self.SB  # 16.0276 Ω

        # IEEE 33-bus branch data [from, to, R_ohm, X_ohm] — topologically sorted
        lines_ohm = np.array([
            [0, 1, 0.0922, 0.0470],
            [1, 2, 0.4930, 0.2511],
            [2, 3, 0.3660, 0.1864],
            [3, 4, 0.3811, 0.1941],
            [4, 5, 0.8190, 0.7070],
            [5, 6, 0.1872, 0.6188],
            [6, 7, 0.7114, 0.2351],
            [7, 8, 1.0300, 0.7400],
            [8, 9, 1.0440, 0.7400],
            [9, 10, 0.1966, 0.0650],
            [10, 11, 0.3744, 0.1238],
            [11, 12, 1.4680, 1.1550],
            [12, 13, 0.5416, 0.7129],
            [13, 14, 0.5910, 0.5260],
            [14, 15, 0.7463, 0.5450],
            [15, 16, 1.2890, 1.7210],
            [16, 17, 0.7320, 0.5740],
            [1, 18, 0.1640, 0.1565],
            [18, 19, 1.5042, 1.3554],
            [19, 20, 0.4095, 0.4784],
            [20, 21, 0.7089, 0.9373],
            [2, 22, 0.4512, 0.3083],
            [22, 23, 0.8980, 0.7091],
            [23, 24, 0.8960, 0.7011],
            [5, 25, 0.2030, 0.1034],
            [25, 26, 0.2842, 0.1447],
            [26, 27, 1.0590, 0.9337],
            [27, 28, 0.8042, 0.7006],
            [28, 29, 0.5075, 0.2585],
            [29, 30, 0.9744, 0.9630],
            [30, 31, 0.3105, 0.3619],
            [31, 32, 0.3410, 0.5302],
        ])
        # Convert to pu
        lines_pu = lines_ohm.astype(float).copy()
        lines_pu[:, 2] /= Z_base
        lines_pu[:, 3] /= Z_base
        self.lines = lines_pu
        self.n_lines = len(lines_pu)

        # Branch impedance in pu
        self.line_r = lines_pu[:, 2]
        self.line_x = lines_pu[:, 3]
        self.line_z = np.sqrt(self.line_r**2 + self.line_x**2)

        # Build from/to arrays
        self.line_from = lines_pu[:, 0].astype(int)
        self.line_to = lines_pu[:, 1].astype(int)

        # ---- IEEE 33-bus load data [node, P_kW, Q_kVar] ----
        load_data = np.array([
            [1, 100, 60], [2, 90, 40], [3, 120, 80],
            [4, 60, 30], [5, 60, 20], [6, 200, 100],
            [7, 200, 100], [8, 60, 20], [9, 60, 20],
            [10, 45, 30], [11, 60, 35], [12, 60, 35],
            [13, 120, 80], [14, 60, 10], [15, 60, 20],
            [16, 60, 20], [17, 90, 40], [18, 90, 40],
            [19, 90, 40], [20, 90, 40], [21, 90, 40],
            [22, 90, 50], [23, 420, 200], [24, 420, 200],
            [25, 60, 25], [26, 60, 25], [27, 60, 20],
            [28, 120, 70], [29, 200, 600], [30, 150, 70],
            [31, 210, 100], [32, 60, 40],
        ])
        # Build per-node P_load, Q_load (pu on SB)
        self.P_load_node = np.zeros(self.n_nodes)
        self.Q_load_node = np.zeros(self.n_nodes)
        for node, pk, qk in load_data:
            self.P_load_node[int(node)] = pk * 1e-3 / self.SB
            self.Q_load_node[int(node)] = qk * 1e-3 / self.SB

        # Total peak load
        self.total_P_peak = np.sum(self.P_load_node)  # ~0.3715 pu
        self.total_Q_peak = np.sum(self.Q_load_node)  # ~0.2300 pu

        # ---- Device mapping ----
        self.wt_nodes = [17, 24, 32]   # 0-indexed node indices
        self.svg_node = 14
        self.cap_node = 7
        self.load_node = None  # loads distributed across all nodes
        self.slack_node = 0

        self.n_wt = 3
        self.n_svg = 1
        self.n_cap = 1
        self.n_oltc = 1
        self.n_dev = self.n_wt + self.n_svg + self.n_cap + self.n_oltc  # 6

        # Device indices in control vector
        self.idx_wt = [0, 1, 2]
        self.idx_svg = 3
        self.idx_cap = 4
        self.idx_oltc = 5

        self.dim = self.n_dev * 24

        # ---- Reactive power limits (pu on SB) ----
        self.wt_q_min = np.array([-0.15, -0.15, -0.10])
        self.wt_q_max = np.array([0.15, 0.15, 0.10])
        self.svg_q_min = -0.25
        self.svg_q_max = 0.25
        self.cap_q_min = 0.00
        self.cap_q_max = 0.15

        # Capacitor: 5 steps × 0.03 pu = 0.15 pu max
        self.cap_step = 0.03
        self.cap_n_steps = 5
        self.cap_steps = np.arange(0, self.cap_n_steps + 1) * self.cap_step
        self.max_cap_switches = 5

        # OLTC: 9 taps (±10%, 2.5% step)
        self.oltc_tap_min = -4
        self.oltc_tap_max = 4
        self.oltc_step_pu = 0.025
        self.max_oltc_changes = 6

        # Apparent power ratings for circle constraint
        self.S_wt = np.array([0.20, 0.20, 0.13])
        self.S_svg = 0.25

        # Box bounds (continuous search space)
        ql_per_h = np.concatenate([self.wt_q_min, [self.svg_q_min], [self.cap_q_min], [self.oltc_tap_min]])
        qu_per_h = np.concatenate([self.wt_q_max, [self.svg_q_max], [self.cap_q_max], [self.oltc_tap_max]])
        self.lb = np.tile(ql_per_h, 24)
        self.ub = np.tile(qu_per_h, 24)

        # Voltage constraints
        self.V_min = 0.90
        self.V_max = 1.10
        self.V_ref = 1.00

        # ---- Build 24h profiles ----
        self._build_profiles()

    def _build_profiles(self):
        """24h wind and load profiles"""
        # Wind power ratio (anti-peak: high at night, low during day)
        wt_ratio = np.array([
            0.85, 0.88, 0.90, 0.87, 0.82, 0.75, 0.65, 0.55,
            0.45, 0.40, 0.38, 0.35, 0.33, 0.36, 0.42, 0.52,
            0.65, 0.75, 0.82, 0.85, 0.88, 0.90, 0.92, 0.88
        ])
        wt_rated_MW = np.array([1.2, 1.2, 0.8])  # 3.2MW total wind

        self.P_wt = np.zeros((self.n_wt, 24))
        for i in range(self.n_wt):
            self.P_wt[i, :] = wt_ratio * wt_rated_MW[i] / self.SB

        # Load profile (evening peak, peak at 19-20h)
        load_ratio = np.array([
            0.42, 0.38, 0.35, 0.33, 0.36, 0.42, 0.52, 0.68,
            0.80, 0.87, 0.90, 0.88, 0.85, 0.88, 0.90, 0.92,
            0.95, 0.98, 1.00, 1.00, 0.98, 0.95, 0.78, 0.52,
        ])

        # Time-varying load at each node
        self.P_load = np.zeros((self.n_nodes, 24))
        self.Q_load = np.zeros((self.n_nodes, 24))
        for h in range(24):
            self.P_load[:, h] = self.P_load_node * load_ratio[h]
            self.Q_load[:, h] = self.Q_load_node * load_ratio[h]

        # ΔQ time-varying weights
        base = load_ratio / np.mean(load_ratio)
        self.delta_q_weights = 0.5 + 0.5 * base


# ============================================================================
# Backward-Forward Sweep (BFS) power flow for radial networks
# ============================================================================

class PowerFlowBFS:
    """Fast BFS power flow for radial distribution networks"""

    def __init__(self, sys: IEEE33BusSystem):
        self.sys = sys
        self.tol = 1e-8
        self.max_iter = 30

    def solve(self, P_inj, Q_inj, V_slack=1.0):
        """
        Backward-Forward Sweep for radial network.
        Branch ordering is topologically sorted (root → leaves).
        Returns: V_mag[n], theta[n], converged
        """
        sys = self.sys
        n = sys.n_nodes
        nl = sys.n_lines
        f_arr = sys.line_from
        t_arr = sys.line_to
        r = sys.line_r
        x = sys.line_x

        V = np.ones(n, dtype=complex) * V_slack
        V[0] = V_slack

        I_br = np.zeros(nl, dtype=complex)

        for it in range(self.max_iter):
            V_old = V.copy()

            # Backward sweep: compute branch currents (leaves → root)
            # Nodal current injection
            I_node = np.zeros(n, dtype=complex)
            for i in range(n):
                if abs(V[i]) > 1e-10:
                    I_node[i] = -(P_inj[i] - 1j * Q_inj[i]) / np.conj(V[i])

            # Sum currents from leaves upward (reverse branch order)
            child_current = np.zeros(n, dtype=complex)
            for br in range(nl - 1, -1, -1):
                f, t = f_arr[br], t_arr[br]
                child_current[t] += I_node[t]
                I_br[br] = child_current[t]
                child_current[f] += I_br[br]

            # Forward sweep: update voltages (root → leaves)
            for br in range(nl):
                f, t = f_arr[br], t_arr[br]
                z_br = complex(r[br], x[br])
                V[t] = V[f] - z_br * I_br[br]

            dV = np.max(np.abs(np.abs(V) - np.abs(V_old)))
            if dV < self.tol:
                break

        V_mag = np.abs(V)
        theta = np.angle(V)
        return V_mag, theta, True, I_br


# ============================================================================
# Fitness evaluator
# ============================================================================

class FitnessEvaluator:
    """
    F = w1*f1 + w2_rise*f2_rise + w2_drop*f2_drop
      + w3*ΔQ + λ*V_penalty + μ*circle_penalty
    """

    def __init__(self, sys: IEEE33BusSystem):
        self.sys = sys
        self.pf = PowerFlowBFS(sys)
        self.w1 = 0.30
        self.w2_rise = 0.15
        self.w2_drop = 0.30
        self.w3 = 0.03
        self.lam = 200.0
        self.mu = 100.0
        self.lam_cap_sw = 10.0
        self.lam_oltc = 6.0

    def evaluate(self, position):
        sys = self.sys
        raw = position.reshape(24, sys.n_dev)

        # ---- Discretize ----
        Q_dev = raw.copy()
        cap_raw = raw[:, sys.idx_cap]
        cap_q = np.round(cap_raw / sys.cap_step) * sys.cap_step
        cap_q = np.clip(cap_q, 0.0, sys.cap_n_steps * sys.cap_step)
        Q_dev[:, sys.idx_cap] = cap_q
        oltc_raw = raw[:, sys.idx_oltc]
        oltc_tap = np.round(oltc_raw).astype(int)
        oltc_tap = np.clip(oltc_tap, sys.oltc_tap_min, sys.oltc_tap_max)
        Q_dev[:, sys.idx_oltc] = oltc_tap.astype(float)

        total_P_loss = 0.0
        total_Q_loss = 0.0
        total_V_rise = 0.0
        total_V_drop = 0.0
        total_V_pen = 0.0
        total_delta_Q = 0.0
        total_circle_pen = 0.0
        V_profile = np.zeros((24, sys.n_nodes))
        Q_applied = Q_dev.copy()
        P_loss_h = np.zeros(24)
        Q_loss_h = np.zeros(24)
        V_dev_h = np.zeros(24)

        for h in range(24):
            q = Q_dev[h].copy()

            # ---- Reactive power circle constraint ----
            for w in range(sys.n_wt):
                S = sys.S_wt[w]
                P = sys.P_wt[w, h]
                max_q_h = np.sqrt(max(S**2 - P**2, 0.0))
                if abs(q[w]) > max_q_h:
                    total_circle_pen += (abs(q[w]) - max_q_h)**2
                    q[w] = np.clip(q[w], -max_q_h, max_q_h)

            if abs(q[sys.idx_svg]) > sys.S_svg:
                total_circle_pen += (abs(q[sys.idx_svg]) - sys.S_svg)**2
                q[sys.idx_svg] = np.clip(q[sys.idx_svg], -sys.S_svg, sys.S_svg)

            Q_applied[h] = q

            # ---- Node injections ----
            P_inj = np.zeros(sys.n_nodes)
            Q_inj = np.zeros(sys.n_nodes)

            for w in range(sys.n_wt):
                P_inj[sys.wt_nodes[w]] += sys.P_wt[w, h]
                Q_inj[sys.wt_nodes[w]] += q[w]
            Q_inj[sys.svg_node] += q[sys.idx_svg]
            Q_inj[sys.cap_node] += q[sys.idx_cap]

            # Subtract loads
            P_inj -= sys.P_load[:, h]
            Q_inj -= sys.Q_load[:, h]

            # OLTC adjusts slack bus voltage
            tap = int(q[sys.idx_oltc])
            V_slack = 1.0 + tap * sys.oltc_step_pu

            V_mag, theta, ok, I_br = self.pf.solve(P_inj, Q_inj, V_slack=V_slack)
            if not ok:
                return 1e10, self._fail_metrics()

            V_profile[h] = V_mag

            # Line losses: Σ |I_br|² × R (and × X)
            for br in range(sys.n_lines):
                i2 = abs(I_br[br])**2
                P_loss_ij = i2 * sys.line_r[br]
                Q_loss_ij = i2 * sys.line_x[br]
                total_P_loss += P_loss_ij
                total_Q_loss += Q_loss_ij
                P_loss_h[h] += P_loss_ij
                Q_loss_h[h] += Q_loss_ij

            # Voltage deviation
            v_dev_sum = 0.0
            for i in range(sys.n_nodes):
                dv = V_mag[i] - sys.V_ref
                v_dev_sum += dv**2
                if dv > 0:
                    total_V_rise += dv**2
                else:
                    total_V_drop += dv**2
            V_dev_h[h] = v_dev_sum

            # Voltage limit penalty
            for i in range(sys.n_nodes):
                if V_mag[i] < sys.V_min:
                    total_V_pen += (sys.V_min - V_mag[i])**2
                elif V_mag[i] > sys.V_max:
                    total_V_pen += (V_mag[i] - sys.V_max)**2

        # ---- Cross-time ΔQ ----
        n_q_dev = sys.n_wt + sys.n_svg + sys.n_cap
        for h in range(23):
            w_h = sys.delta_q_weights[h]
            for d in range(n_q_dev):
                total_delta_Q += w_h * (Q_applied[h+1, d] - Q_applied[h, d])**2

        # ---- Switching penalties ----
        cap_sw_pen = 0.0
        oltc_pen = 0.0
        for h in range(23):
            if abs(Q_applied[h+1, sys.idx_cap] - Q_applied[h, sys.idx_cap]) > 1e-6:
                cap_sw_pen += 1
        cap_sw_pen = max(0, cap_sw_pen - sys.max_cap_switches) * self.lam_cap_sw

        for h in range(23):
            if Q_applied[h+1, sys.idx_oltc] != Q_applied[h, sys.idx_oltc]:
                oltc_pen += 1
        oltc_pen = max(0, oltc_pen - sys.max_oltc_changes) * self.lam_oltc

        fitness = (self.w1 * total_P_loss +
                   self.w2_rise * total_V_rise +
                   self.w2_drop * total_V_drop +
                   self.w3 * total_delta_Q +
                   self.lam * total_V_pen +
                   self.mu * total_circle_pen +
                   cap_sw_pen + oltc_pen)

        return fitness, {
            'P_loss': total_P_loss, 'Q_loss': total_Q_loss,
            'V_rise': total_V_rise, 'V_drop': total_V_drop,
            'V_pen': total_V_pen, 'delta_Q': total_delta_Q,
            'circle_pen': total_circle_pen,
            'cap_sw_pen': cap_sw_pen, 'oltc_pen': oltc_pen,
            'converged': True,
            'V_profile': V_profile, 'Q_applied': Q_applied,
            'P_loss_h': P_loss_h, 'Q_loss_h': Q_loss_h,
            'V_dev_h': V_dev_h,
        }

    def _fail_metrics(self):
        return {'P_loss': 1e10, 'V_rise': 1e10, 'V_drop': 1e10,
                'Q_loss': 1e10, 'V_pen': 1e10, 'delta_Q': 1e10,
                'circle_pen': 1e10, 'cap_sw_pen': 1e10, 'oltc_pen': 1e10,
                'converged': False, 'V_profile': None, 'Q_applied': None,
                'P_loss_h': None, 'Q_loss_h': None, 'V_dev_h': None}


# ============================================================================
# Improved GWO
# ============================================================================

class ImprovedGWO:
    def __init__(self, evaluator, lb, ub, n_wolves=20, max_iter=200,
                 a0=2.0, lam=1.8, k=1.0):
        self.eval_fn = evaluator
        self.lb = lb; self.ub = ub; self.dim = len(lb)
        self.N = n_wolves; self.T = max_iter
        self.a0 = a0; self.lam = lam; self.k = k
        self.w_alpha = 0.50; self.w_beta = 0.33; self.w_delta = 0.17

    def _init_population(self):
        pop = np.zeros((self.N, self.dim))
        for d in range(self.dim):
            r = np.random.rand()
            for i in range(self.N):
                r = 4.0 * r * (1.0 - r)
                pop[i, d] = self.lb[d] + r * (self.ub[d] - self.lb[d])
        return pop

    def _convergence_a(self, t):
        a = self.a0 - self.lam * (t / self.T) ** self.k
        return max(a, 0.0)

    def _bound(self, x):
        return np.clip(x, self.lb, self.ub)

    def _delta_fusion(self, alpha, beta, delta):
        return self._bound(
            (self.w_alpha * alpha + self.w_beta * beta + self.w_delta * delta) /
            (self.w_alpha + self.w_beta + self.w_delta))

    def optimize(self, verbose=True):
        wolves = self._init_population()
        fitness = np.zeros(self.N)
        metrics = [None] * self.N
        for i in range(self.N):
            fitness[i], metrics[i] = self.eval_fn.evaluate(wolves[i])

        order = np.argsort(fitness)
        alpha_pos = wolves[order[0]].copy()
        beta_pos = wolves[order[1]].copy()
        delta_pos = wolves[order[2]].copy()

        curve = np.zeros(self.T)
        for t in range(self.T):
            a = self._convergence_a(t)
            new_delta = self._delta_fusion(alpha_pos, beta_pos, delta_pos)

            for i in range(self.N):
                r1a, r2a = np.random.rand(self.dim), np.random.rand(self.dim)
                A_alpha = 2 * a * r1a - a
                C_alpha = 2 * r2a
                r1b, r2b = np.random.rand(self.dim), np.random.rand(self.dim)
                A_beta = 2 * a * r1b - a
                C_beta = 2 * r2b
                r1d, r2d = np.random.rand(self.dim), np.random.rand(self.dim)
                A_delta = 2 * a * r1d - a
                C_delta = 2 * r2d

                D_alpha = np.abs(C_alpha * alpha_pos - wolves[i])
                D_beta = np.abs(C_beta * beta_pos - wolves[i])
                D_delta = np.abs(C_delta * new_delta - wolves[i])

                X1 = alpha_pos - A_alpha * D_alpha
                X2 = beta_pos - A_beta * D_beta
                X3 = new_delta - A_delta * D_delta
                wolves[i] = self._bound(0.50*X1 + 0.30*X2 + 0.20*X3)

            for i in range(self.N):
                fitness[i], metrics[i] = self.eval_fn.evaluate(wolves[i])

            order = np.argsort(fitness)
            alpha_pos = wolves[order[0]].copy()
            beta_pos = wolves[order[1]].copy()
            delta_pos = wolves[order[2]].copy()
            curve[t] = fitness[order[0]]

            if verbose and (t+1) % 50 == 0:
                print(f"  IGWO iter {t+1:4d}/{self.T} | a={a:.4f} | "
                      f"fit={curve[t]:.6f} | Ploss={metrics[order[0]]['P_loss']:.6f}",
                      flush=True)

        return alpha_pos, curve[-1], curve, metrics[order[0]]


# ============================================================================
# Standard GWO
# ============================================================================

class StandardGWO:
    def __init__(self, evaluator, lb, ub, n_wolves=20, max_iter=150):
        self.eval_fn = evaluator
        self.lb = lb; self.ub = ub; self.dim = len(lb)
        self.N = n_wolves; self.T = max_iter

    def optimize(self, verbose=False):
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        fit = np.array([self.eval_fn.evaluate(p)[0] for p in pop])

        order = np.argsort(fit)
        alpha_pos = pop[order[0]].copy()
        beta_pos = pop[order[1]].copy()
        delta_pos = pop[order[2]].copy()
        curve = np.zeros(self.T)

        for t in range(self.T):
            a = 2.0 - 2.0 * t / self.T
            for i in range(self.N):
                r = np.random.rand(6, self.dim)
                A1, C1 = 2*a*r[0]-a, 2*r[1]
                A2, C2 = 2*a*r[2]-a, 2*r[3]
                A3, C3 = 2*a*r[4]-a, 2*r[5]
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pop[i])
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - pop[i])
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pop[i])
                pop[i] = np.clip((X1+X2+X3)/3.0, self.lb, self.ub)

            fit = np.array([self.eval_fn.evaluate(p)[0] for p in pop])
            order = np.argsort(fit)
            alpha_pos = pop[order[0]].copy()
            beta_pos = pop[order[1]].copy()
            delta_pos = pop[order[2]].copy()
            curve[t] = fit[order[0]]

            if verbose and (t+1) % 50 == 0:
                print(f"  GWO iter {t+1:4d}/{self.T} | fit={curve[t]:.6f}", flush=True)

        _, final_m = self.eval_fn.evaluate(alpha_pos)
        return alpha_pos, curve[-1], curve, final_m


# ============================================================================
# Artificial Bee Colony (ABC)
# ============================================================================

class ABC:
    """
    Artificial Bee Colony for mixed-integer optimization.
    Modifies ~15% of dimensions per bee per iteration (adapted for high-D problems).
    """

    def __init__(self, evaluator, lb, ub, n_bees=20, max_iter=150, limit=50):
        self.eval_fn = evaluator
        self.lb = lb; self.ub = ub; self.dim = len(lb)
        self.N = n_bees; self.T = max_iter; self.limit = limit

    def optimize(self, verbose=False):
        dim = self.dim; N = self.N
        pop = self.lb + np.random.rand(N, dim) * (self.ub - self.lb)
        fit = np.zeros(N)
        for i in range(N):
            fit[i], _ = self.eval_fn.evaluate(pop[i])
        trials = np.zeros(N, dtype=int)

        best_idx = np.argmin(fit)
        best_pos = pop[best_idx].copy()
        best_fit = fit[best_idx]
        curve = np.zeros(self.T)

        for t in range(self.T):
            # ---- Employed bee phase ----
            for i in range(N):
                k = i
                while k == i:
                    k = np.random.randint(N)
                # Modify ~15% of dimensions
                mod_mask = np.random.rand(dim) < 0.15
                if not np.any(mod_mask):
                    mod_mask[np.random.randint(dim)] = True

                new_pos = pop[i].copy()
                phi = np.random.uniform(-1, 1, dim)
                new_pos[mod_mask] += phi[mod_mask] * (pop[i][mod_mask] - pop[k][mod_mask])
                new_pos = np.clip(new_pos, self.lb, self.ub)

                new_fit, _ = self.eval_fn.evaluate(new_pos)
                if new_fit < fit[i]:
                    pop[i] = new_pos
                    fit[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # ---- Onlooker bee phase ----
            # Probability based on fitness (lower is better → higher prob)
            fit_min = np.min(fit)
            fit_shifted = fit - fit_min + 1e-10
            prob = (1.0 / fit_shifted) / np.sum(1.0 / fit_shifted)

            for i in range(N):
                # Roulette wheel selection
                r = np.random.rand()
                cumsum = np.cumsum(prob)
                selected = np.searchsorted(cumsum, r)

                k = selected
                while k == selected:
                    k = np.random.randint(N)

                mod_mask = np.random.rand(dim) < 0.15
                if not np.any(mod_mask):
                    mod_mask[np.random.randint(dim)] = True

                new_pos = pop[selected].copy()
                phi = np.random.uniform(-1, 1, dim)
                new_pos[mod_mask] += phi[mod_mask] * (pop[selected][mod_mask] - pop[k][mod_mask])
                new_pos = np.clip(new_pos, self.lb, self.ub)

                new_fit, _ = self.eval_fn.evaluate(new_pos)
                if new_fit < fit[selected]:
                    pop[selected] = new_pos
                    fit[selected] = new_fit
                    trials[selected] = 0
                else:
                    trials[selected] += 1

            # ---- Scout bee phase ----
            max_trial_idx = np.argmax(trials)
            if trials[max_trial_idx] > self.limit:
                pop[max_trial_idx] = self.lb + np.random.rand(dim) * (self.ub - self.lb)
                fit[max_trial_idx], _ = self.eval_fn.evaluate(pop[max_trial_idx])
                trials[max_trial_idx] = 0

            # Update best
            best_idx = np.argmin(fit)
            if fit[best_idx] < best_fit:
                best_pos = pop[best_idx].copy()
                best_fit = fit[best_idx]
            curve[t] = best_fit

            if verbose and (t+1) % 50 == 0:
                print(f"  ABC iter {t+1:4d}/{self.T} | fit={best_fit:.6f}", flush=True)

        _, final_m = self.eval_fn.evaluate(best_pos)
        return best_pos, best_fit, curve, final_m


# ============================================================================
# Whale Optimization Algorithm (WOA)
# ============================================================================

class WOA:
    """
    Whale Optimization Algorithm with bubble-net attacking behavior.
    """

    def __init__(self, evaluator, lb, ub, n_whales=20, max_iter=150, b=1.0):
        self.eval_fn = evaluator
        self.lb = lb; self.ub = ub; self.dim = len(lb)
        self.N = n_whales; self.T = max_iter; self.b = b

    def optimize(self, verbose=False):
        dim = self.dim; N = self.N
        pop = self.lb + np.random.rand(N, dim) * (self.ub - self.lb)
        fit = np.zeros(N)
        for i in range(N):
            fit[i], _ = self.eval_fn.evaluate(pop[i])

        best_idx = np.argmin(fit)
        best_pos = pop[best_idx].copy()
        best_fit = fit[best_idx]
        curve = np.zeros(self.T)

        for t in range(self.T):
            a = 2.0 - 2.0 * t / self.T  # linearly decreasing from 2 to 0

            for i in range(N):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # Encircling prey
                        D = np.abs(C * best_pos - pop[i])
                        new_pos = best_pos - A * D
                    else:
                        # Search for prey (random whale)
                        rand_idx = np.random.randint(N)
                        D = np.abs(C * pop[rand_idx] - pop[i])
                        new_pos = pop[rand_idx] - A * D
                else:
                    # Spiral update
                    l = np.random.uniform(-1, 1, dim)
                    D_prime = np.abs(best_pos - pop[i])
                    new_pos = D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_pos

                pop[i] = np.clip(new_pos, self.lb, self.ub)

            for i in range(N):
                fit[i], _ = self.eval_fn.evaluate(pop[i])
                if fit[i] < best_fit:
                    best_pos = pop[i].copy()
                    best_fit = fit[i]

            curve[t] = best_fit

            if verbose and (t+1) % 50 == 0:
                print(f"  WOA iter {t+1:4d}/{self.T} | fit={best_fit:.6f}", flush=True)

        _, final_m = self.eval_fn.evaluate(best_pos)
        return best_pos, best_fit, curve, final_m


# ============================================================================
# Particle Swarm Optimization (PSO)
# ============================================================================

class PSO:
    """Standard PSO with inertia weight decay"""

    def __init__(self, evaluator, lb, ub, n_particles=20, max_iter=100,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5):
        self.eval_fn = evaluator
        self.lb = lb; self.ub = ub; self.dim = len(lb)
        self.N = n_particles; self.T = max_iter
        self.w_start = w_start; self.w_end = w_end
        self.c1 = c1; self.c2 = c2

    def optimize(self, verbose=False):
        dim = self.dim; N = self.N
        pos = self.lb + np.random.rand(N, dim) * (self.ub - self.lb)
        vel = np.random.uniform(-0.1, 0.1, (N, dim)) * (self.ub - self.lb)
        fit = np.zeros(N)
        for i in range(N):
            fit[i], _ = self.eval_fn.evaluate(pos[i])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        curve = np.zeros(self.T)

        for t in range(self.T):
            w = self.w_start - (self.w_start - self.w_end) * t / self.T

            for i in range(N):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                vel[i] = (w * vel[i]
                          + self.c1 * r1 * (pbest_pos[i] - pos[i])
                          + self.c2 * r2 * (gbest_pos - pos[i]))
                pos[i] = np.clip(pos[i] + vel[i], self.lb, self.ub)

                fit[i], _ = self.eval_fn.evaluate(pos[i])
                if fit[i] < pbest_fit[i]:
                    pbest_fit[i] = fit[i]
                    pbest_pos[i] = pos[i].copy()
                    if fit[i] < gbest_fit:
                        gbest_fit = fit[i]
                        gbest_pos = pos[i].copy()

            curve[t] = gbest_fit

            if verbose and (t+1) % 50 == 0:
                print(f"  PSO iter {t+1:4d}/{self.T} | fit={gbest_fit:.6f}", flush=True)

        _, final_m = self.eval_fn.evaluate(gbest_pos)
        return gbest_pos, gbest_fit, curve, final_m


# ============================================================================
# 4-curve comparison figures (PSO / ABC / WOA / IGWO)
# ============================================================================

def save_comparison_figures(sys, results, outdir):
    """
    results: dict with keys 'pso', 'abc', 'woa', 'igwo'
    """
    h = np.arange(24)
    fig_w, fig_h = 3.5, 2.5

    styles = {
        'baseline': ('k-o', 'Baseline (Q=0)', 3, 1.0),
        'gwo': ('b--s', 'Standard GWO', 3, 1.0),
        'igwo': ('r-^', 'Improved GWO', 3, 1.2),
    }
    order = ['baseline', 'gwo', 'igwo']

    # ---- Fig 1: Active Power Loss ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for key in order:
        if key in results:
            fmt, label, ms, lw = styles[key]
            ax.plot(h, results[key]['P_loss_h'], fmt, ms=ms, lw=lw, label=label)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Active Power Loss (pu)')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig1_active_power_loss', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 2: Voltage Deviation ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for key in order:
        if key in results:
            fmt, label, ms, lw = styles[key]
            ax.plot(h, results[key]['V_dev_h'], fmt, ms=ms, lw=lw, label=label)
    ax.set_xlabel('Hour')
    ax.set_ylabel(r'Voltage Deviation $\sum (V_i - V_{\mathrm{ref}})^2$')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig2_voltage_deviation', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 3: Reactive Power Loss ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for key in order:
        if key in results:
            fmt, label, ms, lw = styles[key]
            ax.plot(h, results[key]['Q_loss_h'], fmt, ms=ms, lw=lw, label=label)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Reactive Power Loss (pu)')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig3_reactive_power_loss', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 4: Convergence ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for key in ['gwo', 'igwo']:
        if key in results and results[key].get('curve') is not None:
            c = results[key]['curve']
            fmt = styles[key][0][0] + '-'  # solid line from style
            label = f"{styles[key][1]} (best={results[key]['fitness']:.4f})"
            lw = styles[key][3]
            ax.plot(range(len(c)), c, fmt, lw=lw, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig4_convergence', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    print(f"\n[Figs saved] {outdir}/fig1..4_*.{{pdf,png}}")


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(42)
    print("=" * 65)
    print("  IGWO vs GWO — IEEE 33-bus Voltage/Reactive Power Optimization")
    print("=" * 65)

    # [1] Build system
    print("\n[1/4] Building IEEE 33-bus system...")
    sys = IEEE33BusSystem()
    print(f"  Nodes: {sys.n_nodes} | Lines: {sys.n_lines} | Devices: {sys.n_dev}")
    print(f"  WT nodes: {sys.wt_nodes} | SVG: node {sys.svg_node} | "
          f"Cap: node {sys.cap_node} | OLTC: node {sys.slack_node}")
    print(f"  Total peak load: {sys.total_P_peak*sys.SB*1000:.1f} kW / "
          f"{sys.total_Q_peak*sys.SB*1000:.1f} kVar")
    print(f"  Wind capacity: 3.2 MW | Search dim: {sys.dim}")

    evaluator = FitnessEvaluator(sys)
    results = {}

    # [2] Baseline (Q=0)
    print("\n[2/4] Computing baseline (Q=0)...")
    _, base_m = evaluator.evaluate(np.zeros(sys.dim))
    results['baseline'] = base_m
    print(f"  Base P_loss={base_m['P_loss']:.6f}  Q_loss={base_m['Q_loss']:.6f}  "
          f"V_dev={base_m['V_rise']+base_m['V_drop']:.6f}")

    # [3] Standard GWO
    print("\n[3/4] Running Standard GWO...")
    gwo = StandardGWO(evaluator, sys.lb, sys.ub, n_wolves=20, max_iter=150)
    t0 = time.time()
    best_gwo, fit_gwo, curve_gwo, metrics_gwo = gwo.optimize(verbose=True)
    t_gwo = time.time() - t0
    results['gwo'] = {**metrics_gwo, 'fitness': fit_gwo, 'curve': curve_gwo}
    print(f"  GWO done in {t_gwo:.1f}s | fit={fit_gwo:.6f} | "
          f"P_loss={metrics_gwo['P_loss']:.6f}")

    # [4] Improved GWO
    print("\n[4/4] Running Improved GWO...")
    igwo = ImprovedGWO(evaluator, sys.lb, sys.ub, n_wolves=20, max_iter=300,
                       a0=2.0, lam=1.8, k=1.0)
    t0 = time.time()
    best_igwo, fit_igwo, curve_igwo, metrics_igwo = igwo.optimize(verbose=True)
    t_igwo = time.time() - t0
    results['igwo'] = {**metrics_igwo, 'fitness': fit_igwo, 'curve': curve_igwo}
    print(f"  IGWO done in {t_igwo:.1f}s | fit={fit_igwo:.6f} | "
          f"P_loss={metrics_igwo['P_loss']:.6f}")

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  OPTIMIZATION RESULTS SUMMARY (IEEE 33-bus)")
    print("=" * 65)
    print(f"  {'Method':<20s} {'P_loss':>10s} {'Q_loss':>10s} "
          f"{'V_dev':>10s} {'ΔQ':>10s} {'Fitness':>10s} {'Time':>8s}")
    print("  " + "-" * 65)
    for key, t_used, label in [
        ('baseline', 0, 'Baseline (Q=0)'),
        ('gwo', t_gwo, 'Std GWO'),
        ('igwo', t_igwo, 'Improved GWO'),
    ]:
        m = results[key]
        v_dev = m.get('V_rise', 0) + m.get('V_drop', 0)
        print(f"  {label:<20s} {m['P_loss']:10.6f} {m['Q_loss']:10.6f} "
              f"{v_dev:10.6f} {m.get('delta_Q',0):10.6f} "
              f"{m.get('fitness',0):10.6f} {t_used:7.1f}s")

    # ---- Figures ----
    outdir = r'D:\04_project\vscode'
    save_comparison_figures(sys, results, outdir)

    total_t = t_gwo + t_igwo
    print(f"\n  Total optimization time: {total_t:.1f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
