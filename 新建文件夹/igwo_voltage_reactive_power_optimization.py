"""
改进灰狼优化算法(IGWO) — 新能源发电系统全天时电压偏差与无功功率损耗优化
=====================================================================
参考文献：新能源发电系统电能质量优化方法及装置 (修订发明-163P012)

对照论文 Mahmoud 2020 框架所做改进：
  改进1: 引入跨时段耦合项 ΔQ（无功出力变化率），替代原专利的 Δu
  改进2: 引入风机/SVG 无功圆约束 Q² ≤ S² − P²
  改进3: 拆分电压偏差为电压升高(f2_rise)和电压跌落(f2_drop)两项
  改进4: 权重灵敏度分析
  改进6: 补全约束具体参数值

IGWO 算法核心改进（对比标准GWO）：
  1. 非线性收敛因子: a(m) = a_initial - lambda * (m/M)^k
  2. delta狼融合变异策略: alpha, beta, delta 加权融合生成新delta狼
  3. 调整位置更新权重系数（不等权）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，无需GUI
import matplotlib.pyplot as plt
import time, os.path as _osp

# ---- Figura: IEEE publication-quality figure setup ----
import sys as _sys
_sys.path.insert(0, r"C:\Users\18771\.claude\plugins\cache\figura\figura\0.4.0\skills\figura\scripts")
import pubstyle, colors, export
pubstyle.apply(venue="ieee")
colors.apply_cycle()
# CJK fallback: 保留IEEE主字体, 中文回退到SimHei
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei', 'sans-serif']

# ============================================================================
# 第1部分：测试系统 — 7节点新能源发电系统
# ============================================================================

class WindFarmSystem:
    """
    新能源发电测试系统 (7节点, SB=10MVA, VB=35kV)

    拓扑:
      Node 0: PCC平衡节点 (V=1.0∠0°)
      Node 1-3: 直驱风机 WT1, WT2, WT3 (PQ节点, 按Q设定值调节)
      Node 4: SVG安装点 (PQ节点)
      Node 5: 电容器组安装点 (PQ节点)
      Node 6: 负荷节点 (PQ节点)

            0 (PCC)
           /|\
          / | \
         1  2  6 (负荷)
         |  |  |
         4--3--5
         |_____|

      线路: 0-1, 0-2, 0-6, 1-4, 2-5, 3-4, 4-5, 5-6
    """

    def __init__(self):
        self.SB = 10.0       # MVA 基准
        self.VB = 35.0       # kV 基准
        self.n_nodes = 7
        self.n_time = 24

        # 线路: [from, to, R(pu), X(pu)]
        # 35kV农网长馈线 (25-40km), ACSR-120导线: R≈0.27 Ω/km, X≈0.35 Ω/km
        # Z_base = 35²/10 = 122.5 Ω, per-unit ≈ 0.0022/km
        lines = np.array([
            [0, 1, 0.088, 0.280],   # 0→WT1 主干 ~35km
            [0, 2, 0.100, 0.320],   # 0→WT2 主干 ~40km
            [0, 6, 0.130, 0.400],   # 0→负荷 主干 ~50km (远端)
            [1, 4, 0.044, 0.140],   # WT1→SVG 分支 ~18km
            [2, 5, 0.044, 0.140],   # WT2→Cap 分支 ~18km
            [3, 4, 0.066, 0.210],   # WT3→SVG 分支 ~27km
            [4, 5, 0.055, 0.175],   # SVG↔Cap 联络 ~22km
            [5, 6, 0.033, 0.105],   # Cap→负荷 末端 ~13km
        ])
        self.n_lines = len(lines)

        # 构建导纳矩阵
        self.Ybus = np.zeros((self.n_nodes, self.n_nodes), dtype=complex)
        for f, t, r, x in lines:
            f, t = int(f), int(t)
            y = 1.0 / (r + 1j * x)
            self.Ybus[f, t] -= y
            self.Ybus[t, f] -= y
            self.Ybus[f, f] += y
            self.Ybus[t, t] += y

        self.G = np.real(self.Ybus)
        self.B = np.imag(self.Ybus)

        # 存储线路数据用于损耗计算
        self.lines = np.array(lines)
        self.line_g = np.array([r/(r*r + x*x) for _, _, r, x in lines])
        self.line_b = np.array([x/(r*r + x*x) for _, _, r, x in lines])

        # 节点类型: 0=slack, 1=PQ
        self.node_type = np.array([0, 1, 1, 1, 1, 1, 1])
        self.slack_node = 0
        self.pq_nodes = np.where(self.node_type == 1)[0]
        self.pv_nodes = np.where(self.node_type == 2)[0]

        # 设备节点映射
        self.wt_nodes  = [1, 2, 3]
        self.svg_node  = 4
        self.cap_node  = 5
        self.load_node = 6

        self.n_wt  = 3
        self.n_svg = 1
        self.n_cap = 1
        self.n_oltc = 1
        self.n_dev = self.n_wt + self.n_svg + self.n_cap + self.n_oltc  # 6

        # 设备索引 (在24h展开中的offset)
        self.idx_wt   = [0, 1, 2]
        self.idx_svg  = 3
        self.idx_cap  = 4
        self.idx_oltc = 5

        # 搜索空间维度 = 设备数 × 24时段
        self.dim = self.n_dev * 24

        # ---- 设备无功出力范围 (pu on SB) — 工程实际尺度 ----
        self.wt_q_min  = np.array([-0.25, -0.25, -0.20])
        self.wt_q_max  = np.array([ 0.25,  0.25,  0.20])
        self.svg_q_min = -0.30
        self.svg_q_max =  0.30
        self.cap_q_min =  0.00
        self.cap_q_max =  0.20

        # ---- 离散电容器分组投切 ----
        # 5组电容器, 每组0.04pu, 构成6档: [0, 0.04, 0.08, 0.12, 0.16, 0.20] pu
        self.cap_step = 0.04
        self.cap_n_steps = 5
        self.cap_steps = np.arange(0, self.cap_n_steps + 1) * self.cap_step
        self.max_cap_switches = 5  # 每日最大投切次数

        # ---- 有载调压变压器 (OLTC) ----
        # 9档位: ±10%, 步长2.5%
        self.oltc_tap_min = -4
        self.oltc_tap_max = 4
        self.oltc_step_pu = 0.025
        self.max_oltc_changes = 6  # 每日最大分接头变化次数

        # ---- 无功圆约束 (视在功率额定值) ----
        self.S_wt = np.array([0.40, 0.40, 0.32])
        self.S_svg = 0.30
        self.S_svg = 0.30

        # 盒式上下限 (含OLTC档位)
        ql_per_h = np.concatenate([self.wt_q_min, [self.svg_q_min], [self.cap_q_min], [self.oltc_tap_min]])
        qu_per_h = np.concatenate([self.wt_q_max, [self.svg_q_max], [self.cap_q_max], [self.oltc_tap_max]])
        self.lb = np.tile(ql_per_h, 24)
        self.ub = np.tile(qu_per_h, 24)

        # ---- 电压约束 ----
        self.V_min = 0.90
        self.V_max = 1.10
        self.V_ref = 1.00

        # ---- 日曲线 ----
        self._build_profiles()

    def _build_profiles(self):
        """构建24h典型风电出力和负荷曲线 (工程实际尺度)"""
        # 风电出力比 (反调峰: 夜高昼低)
        wt_ratio = np.array([
            0.85, 0.88, 0.90, 0.87, 0.82, 0.75, 0.65, 0.55,
            0.45, 0.40, 0.38, 0.35, 0.33, 0.36, 0.42, 0.52,
            0.65, 0.75, 0.82, 0.85, 0.88, 0.90, 0.92, 0.88
        ])
        # 风机额定功率 (MW): WT1=3.0, WT2=3.0, WT3=2.5 → 总装机8.5MW
        wt_rated_MW = np.array([3.0, 3.0, 2.5])

        self.P_wt = np.zeros((self.n_wt, 24))
        for i in range(self.n_wt):
            self.P_wt[i, :] = wt_ratio * wt_rated_MW[i] / self.SB

        # 负荷曲线 (昼高夜低), 峰值9MW/4.0MVar (重载农网馈线)
        load_ratio = np.array([
            0.50, 0.45, 0.42, 0.40, 0.42, 0.55, 0.72, 0.85,
            0.92, 0.95, 0.97, 0.93, 0.88, 0.90, 0.95, 0.98,
            1.00, 0.95, 0.88, 0.82, 0.75, 0.70, 0.62, 0.55
        ])
        self.P_load = load_ratio * 9.0 / self.SB   # 峰值9MW (0.90 pu)
        self.Q_load = load_ratio * 4.0 / self.SB   # 峰值4.0MVar (0.40 pu)


# ============================================================================
# 第2部分：潮流计算（牛顿-拉夫逊法，解析雅可比矩阵）
# ============================================================================

class PowerFlow:
    """牛顿-拉夫逊法潮流求解器（解析雅可比矩阵，高效实现）"""

    def __init__(self, sys: WindFarmSystem):
        self.sys = sys
        self.tol = 1e-8
        self.max_iter = 30

    def solve(self, P_inj, Q_inj, V_slack=1.0):
        """
        输入: P_inj[n], Q_inj[n], V_slack (OLTC调节后的平衡节点电压)
        输出: V[n], theta[n], converged
        """
        sys = self.sys
        G, B = sys.G, sys.B
        n = sys.n_nodes
        slack = sys.slack_node
        pq = sys.pq_nodes
        pv = sys.pv_nodes

        non_slack = np.concatenate([pv, pq])
        n_ns = len(non_slack)

        V = np.ones(n)
        V[slack] = V_slack  # OLTC调节平衡节点电压
        theta = np.zeros(n)

        # 未知量索引: dθ for all non-slack, then dV for PQ only
        n_pq = len(pq)
        n_unk = n_ns + n_pq  # 6 + 3 = 9

        for it in range(self.max_iter):
            # ---- 计算计算功率和失配量 ----
            P_calc = np.zeros(n)
            Q_calc = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    ang = theta[i] - theta[j]
                    P_calc[i] += V[i] * V[j] * (G[i,j] * np.cos(ang) + B[i,j] * np.sin(ang))
                    Q_calc[i] += V[i] * V[j] * (G[i,j] * np.sin(ang) - B[i,j] * np.cos(ang))

            dP = P_inj - P_calc
            dQ = Q_inj - Q_calc

            # 失配量向量: [dP_non_slack; dQ_pq]
            mismatch = np.zeros(n_unk)
            mismatch[:n_ns] = dP[non_slack]          # dP for PV + PQ
            mismatch[n_ns:] = dQ[pq]                  # dQ for PQ only

            if np.max(np.abs(mismatch)) < self.tol:
                return V, theta, True

            # ---- 构建解析雅可比矩阵 ----
            # J = [H  N]   H = dP/dθ,  N = dP/dV * V
            #     [K  L]   K = dQ/dθ,  L = dQ/dV * V
            J = np.zeros((n_unk, n_unk))

            for row_i, i in enumerate(non_slack):
                for col_j, j in enumerate(non_slack):
                    ang = theta[i] - theta[j]
                    if i == j:
                        # H_ii = -Q_i - B_ii * V_i^2
                        J[row_i, col_j] = -Q_calc[i] - B[i,i] * V[i]**2
                    else:
                        # H_ij = V_i * V_j * (G_ij * sin(ang) - B_ij * cos(ang))
                        J[row_i, col_j] = V[i] * V[j] * (G[i,j] * np.sin(ang) - B[i,j] * np.cos(ang))

                # N = dP/dV * V
                for col_j, j in enumerate(pq):  # only PQ nodes have dV
                    col = n_ns + col_j
                    if i == j:
                        # N_ii = P_i + G_ii * V_i^2
                        J[row_i, col] = P_calc[i] + G[i,i] * V[i]**2
                    else:
                        ang = theta[i] - theta[j]
                        # N_ij = V_i * V_j * (G_ij * cos(ang) + B_ij * sin(ang))
                        J[row_i, col] = V[i] * V[j] * (G[i,j] * np.cos(ang) + B[i,j] * np.sin(ang))

            # K = dQ/dθ  (only for PQ nodes)
            for row_i, i in enumerate(pq):
                row = n_ns + row_i
                for col_j, j in enumerate(non_slack):
                    ang = theta[i] - theta[j]
                    if i == j:
                        # K_ii = P_i - G_ii * V_i^2
                        J[row, col_j] = P_calc[i] - G[i,i] * V[i]**2
                    else:
                        # K_ij = -V_i * V_j * (G_ij * cos(ang) + B_ij * sin(ang))
                        J[row, col_j] = -V[i] * V[j] * (G[i,j] * np.cos(ang) + B[i,j] * np.sin(ang))

                # L = dQ/dV * V (only for PQ nodes)
                for col_j, j in enumerate(pq):
                    col = n_ns + col_j
                    if i == j:
                        # L_ii = Q_i - B_ii * V_i^2
                        J[row, col] = Q_calc[i] - B[i,i] * V[i]**2
                    else:
                        ang = theta[i] - theta[j]
                        # L_ij = V_i * V_j * (G_ij * sin(ang) - B_ij * cos(ang))
                        J[row, col] = V[i] * V[j] * (G[i,j] * np.sin(ang) - B[i,j] * np.cos(ang))

            # 求解修正方程
            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, mismatch, rcond=None)[0]

            # 更新变量
            theta[non_slack] += dx[:n_ns]
            V[pq] += dx[n_ns:]

        # 不收敛
        return V, theta, False


# ============================================================================
# 第3部分：多目标评估函数（对照论文框架改进）
# ============================================================================

class FitnessEvaluator:
    """
    对照论文 Mahmoud 2020 的目标函数框架所做改进：

    F = w1 × f1 + w2_rise × f2_rise + w2_drop × f2_drop
      + w3 × ΔQ + λ × V_penalty + μ × circle_penalty

    其中:
      f1:         全天线路有功损耗总和              (对应专利原有)
      f2_rise:    全天电压升高偏差总和 (Vi > Vref)  (改进3: 拆分电压)
      f2_drop:    全天电压跌落偏差总和 (Vi < Vref)  (改进3: 拆分电压)
      ΔQ:         无功出力跨时段变化率              (改进1+7: 替代原专利Δu)
      V_penalty:  电压越限惩罚 (Vi 超出 [Vmin,Vmax])
      circle_pen: 无功圆约束惩罚 (改进2: Q² ≤ S²−P²)
    """

    def __init__(self, sys: WindFarmSystem):
        self.sys = sys
        self.pf = PowerFlow(sys)
        self.w1 = 0.30         # 有功损耗
        self.w2_rise = 0.15    # 电压升高偏差
        self.w2_drop = 0.30    # 电压跌落偏差
        self.w3 = 0.10         # 跨时段无功变化率 ΔQ
        self.lam = 150.0       # 电压越限惩罚
        self.mu = 80.0         # 无功圆约束惩罚
        self.lam_cap_sw = 8.0  # 电容器日投切次数超标惩罚
        self.lam_oltc = 5.0    # OLTC日分接头变化次数超标惩罚

    def evaluate(self, position):
        """
        position: (n_dev * 24,) 决策变量向量
        设备顺序: WT1, WT2, WT3, SVG, Cap(离散), OLTC(离散)
        返回: (fitness, metrics_dict)
        """
        sys = self.sys
        raw = position.reshape(24, sys.n_dev)

        # ---- 离散化 ----
        Q_dev = raw.copy()
        # 电容器量化到最近档位
        cap_raw = raw[:, sys.idx_cap]
        cap_q = np.round(cap_raw / sys.cap_step) * sys.cap_step
        cap_q = np.clip(cap_q, 0.0, sys.cap_n_steps * sys.cap_step)
        Q_dev[:, sys.idx_cap] = cap_q
        # OLTC量化到最近整数档位
        oltc_raw = raw[:, sys.idx_oltc]
        oltc_tap = np.round(oltc_raw).astype(int)
        oltc_tap = np.clip(oltc_tap, sys.oltc_tap_min, sys.oltc_tap_max)
        Q_dev[:, sys.idx_oltc] = oltc_tap.astype(float)

        total_P_loss = 0.0
        total_Q_loss = 0.0
        total_V_rise  = 0.0
        total_V_drop  = 0.0
        total_V_pen   = 0.0
        total_delta_Q = 0.0
        total_circle_pen = 0.0
        V_profile = np.zeros((24, sys.n_nodes))
        Q_applied = Q_dev.copy()
        P_loss_h = np.zeros(24)
        Q_loss_h = np.zeros(24)
        V_dev_h  = np.zeros(24)

        for h in range(24):
            q = Q_dev[h].copy()

            # ---- 无功圆约束 (WT + SVG) ----
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

            # 构建节点注入
            P_inj = np.zeros(sys.n_nodes)
            Q_inj = np.zeros(sys.n_nodes)

            for w in range(sys.n_wt):
                P_inj[sys.wt_nodes[w]] = sys.P_wt[w, h]

            P_inj[sys.load_node] = -sys.P_load[h]
            Q_inj[sys.load_node] = -sys.Q_load[h]

            for w in range(sys.n_wt):
                Q_inj[sys.wt_nodes[w]] = q[w]
            Q_inj[sys.svg_node] = q[sys.idx_svg]
            Q_inj[sys.cap_node] = q[sys.idx_cap]

            # OLTC调节平衡节点电压
            tap = int(q[sys.idx_oltc])
            V_slack = 1.0 + tap * sys.oltc_step_pu

            V, theta, ok = self.pf.solve(P_inj, Q_inj, V_slack=V_slack)
            if not ok:
                return 1e10, {
                    'P_loss': 1e10, 'V_rise': 1e10, 'V_drop': 1e10,
                    'Q_loss': 1e10, 'V_pen': 1e10, 'delta_Q': 1e10,
                    'circle_pen': 1e10, 'cap_sw_pen': 1e10, 'oltc_pen': 1e10,
                    'converged': False, 'V_profile': None, 'Q_applied': None,
                    'P_loss_h': None, 'Q_loss_h': None, 'V_dev_h': None,
                }

            V_profile[h] = V

            for line_idx, (f, t, r, x) in enumerate(sys.lines):
                f, t = int(f), int(t)
                g = sys.line_g[line_idx]
                b = sys.line_b[line_idx]
                ang = theta[f] - theta[t]
                dV2 = V[f]**2 + V[t]**2 - 2 * V[f] * V[t] * np.cos(ang)
                P_loss_ij = g * dV2
                Q_loss_ij = b * dV2
                total_P_loss += P_loss_ij
                total_Q_loss += Q_loss_ij
                P_loss_h[h] += P_loss_ij
                Q_loss_h[h] += Q_loss_ij

            v_dev_sum = 0.0
            for i in range(sys.n_nodes):
                dv = V[i] - sys.V_ref
                v_dev_sum += dv**2
                if dv > 0:
                    total_V_rise += dv**2
                else:
                    total_V_drop += dv**2
            V_dev_h[h] = v_dev_sum

            for i in range(sys.n_nodes):
                if V[i] < sys.V_min:
                    total_V_pen += (sys.V_min - V[i])**2
                elif V[i] > sys.V_max:
                    total_V_pen += (V[i] - sys.V_max)**2

        # ---- 跨时段无功变化率 (仅Q设备, 不含OLTC) ----
        n_q_dev = sys.n_wt + sys.n_svg + sys.n_cap
        for h in range(23):
            for d in range(n_q_dev):
                total_delta_Q += (Q_applied[h+1, d] - Q_applied[h, d])**2

        # ---- 日投切/调压次数约束 ----
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
            'P_loss': total_P_loss,
            'Q_loss': total_Q_loss,
            'V_rise': total_V_rise,
            'V_drop': total_V_drop,
            'V_pen': total_V_pen,
            'delta_Q': total_delta_Q,
            'circle_pen': total_circle_pen,
            'cap_sw_pen': cap_sw_pen,
            'oltc_pen': oltc_pen,
            'converged': True,
            'V_profile': V_profile,
            'Q_applied': Q_applied,
            'P_loss_h': P_loss_h,
            'Q_loss_h': Q_loss_h,
            'V_dev_h': V_dev_h,
        }


# ============================================================================
# 第4部分：【核心】改进灰狼优化算法 (IGWO)
# ============================================================================

class ImprovedGWO:
    """
    改进灰狼优化算法

    专利核心改进:
    (1) 非线性收敛因子: a(m) = a0 - lambda * (m/M)^k
        - k>1: 前期a下降慢→强全局搜索; 后期下降快→精细局部搜索
    (2) delta狼融合变异: δ_new = w_α*X_α + w_β*X_β + w_δ*X_δ
        - α,β获得更高权重, 引导群体向优质解移动
    (3) 位置更新使用不等权加权平均 (w1>w2>w3)
    """

    def __init__(self, evaluator: FitnessEvaluator, lb, ub,
                 n_wolves=30, max_iter=200,
                 a0=2.0, lam=1.8, k=1.5,
                 w_alpha=0.50, w_beta=0.33, w_delta=0.17):
        self.eval_fn = evaluator
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.N = n_wolves
        self.T = max_iter
        self.a0 = a0
        self.lam = lam
        self.k = k
        self.w_alpha = w_alpha
        self.w_beta = w_beta
        self.w_delta = w_delta

    def _init_population(self):
        """Logistic混沌映射初始化, 提高初始种群多样性"""
        pop = np.zeros((self.N, self.dim))
        for d in range(self.dim):
            r = np.random.rand()
            for i in range(self.N):
                r = 4.0 * r * (1.0 - r)   # Logistic映射
                pop[i, d] = self.lb[d] + r * (self.ub[d] - self.lb[d])
        return pop

    def _convergence_a(self, t):
        """非线性收敛因子 (专利公式)"""
        ratio = t / self.T
        a = self.a0 - self.lam * (ratio ** self.k)
        return max(a, 0.0)

    def _bound(self, x):
        return np.clip(x, self.lb, self.ub)

    def _delta_fusion(self, alpha, beta, delta):
        """delta狼融合变异 (专利核心改进)"""
        w_sum = self.w_alpha + self.w_beta + self.w_delta
        new_delta = (self.w_alpha * alpha + self.w_beta * beta +
                     self.w_delta * delta) / w_sum
        return self._bound(new_delta)

    def optimize(self, verbose=True):
        """执行IGWO优化"""
        # ---- 初始化 ----
        wolves = self._init_population()
        fitness = np.zeros(self.N)
        metrics = [None] * self.N
        for i in range(self.N):
            fitness[i], metrics[i] = self.eval_fn.evaluate(wolves[i])

        # 升序排序 (适应度越小越好)
        order = np.argsort(fitness)
        alpha_pos = wolves[order[0]].copy()
        beta_pos  = wolves[order[1]].copy()
        delta_pos = wolves[order[2]].copy()
        alpha_fit = fitness[order[0]]

        curve = np.zeros(self.T)
        a_vals = np.zeros(self.T)

        for t in range(self.T):
            a = self._convergence_a(t)
            a_vals[t] = a

            # 融合变异生成新delta
            new_delta = self._delta_fusion(alpha_pos, beta_pos, delta_pos)

            # 更新每只狼
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
                D_beta  = np.abs(C_beta  * beta_pos  - wolves[i])
                D_delta = np.abs(C_delta * new_delta - wolves[i])

                X1 = alpha_pos - A_alpha * D_alpha
                X2 = beta_pos  - A_beta  * D_beta
                X3 = new_delta - A_delta * D_delta

                # 不等权加权 (专利公式13: 调整权重系数)
                wolves[i] = self._bound((0.50*X1 + 0.30*X2 + 0.20*X3) / 1.0)

            # 重新评估
            for i in range(self.N):
                fitness[i], metrics[i] = self.eval_fn.evaluate(wolves[i])

            # 更新alpha/beta/delta
            order = np.argsort(fitness)
            alpha_pos = wolves[order[0]].copy()
            beta_pos  = wolves[order[1]].copy()
            delta_pos = wolves[order[2]].copy()
            alpha_fit = fitness[order[0]]

            curve[t] = alpha_fit

            if verbose and (t+1) % 50 == 0:
                print(f"  Iter {t+1:4d}/{self.T} | a={a:.4f} | "
                      f"fitness={alpha_fit:.8f} | "
                      f"P_loss={metrics[order[0]]['P_loss']:.6f} | "
                      f"V_rise={metrics[order[0]]['V_rise']:.6f} | "
                      f"V_drop={metrics[order[0]]['V_drop']:.6f}",
                      flush=True)

        best_metrics = metrics[order[0]]
        return alpha_pos, alpha_fit, curve, a_vals, best_metrics


# ============================================================================
# 第5部分：标准GWO（对比用）
# ============================================================================

class StandardGWO:
    """标准GWO — 线性收敛因子，等权位置更新，无融合变异"""

    def __init__(self, evaluator, lb, ub, n_wolves=30, max_iter=200):
        self.eval_fn = evaluator
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.N = n_wolves
        self.T = max_iter

    def optimize(self, verbose=False):
        pop = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        fit = np.array([self.eval_fn.evaluate(p)[0] for p in pop])

        order = np.argsort(fit)
        alpha_pos = pop[order[0]].copy()
        beta_pos  = pop[order[1]].copy()
        delta_pos = pop[order[2]].copy()
        alpha_fit = fit[order[0]]
        curve = np.zeros(self.T)

        for t in range(self.T):
            a = 2.0 - 2.0 * t / self.T  # 线性递减

            for i in range(self.N):
                r = np.random.rand(6, self.dim)
                A1, C1 = 2*a*r[0]-a, 2*r[1]
                A2, C2 = 2*a*r[2]-a, 2*r[3]
                A3, C3 = 2*a*r[4]-a, 2*r[5]

                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pop[i])
                X2 = beta_pos  - A2 * np.abs(C2 * beta_pos  - pop[i])
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pop[i])

                pop[i] = np.clip((X1 + X2 + X3) / 3.0, self.lb, self.ub)

            fit = np.array([self.eval_fn.evaluate(p)[0] for p in pop])

            order = np.argsort(fit)
            alpha_pos = pop[order[0]].copy()
            beta_pos  = pop[order[1]].copy()
            delta_pos = pop[order[2]].copy()
            alpha_fit = fit[order[0]]
            curve[t] = alpha_fit

        _, final_m = self.eval_fn.evaluate(alpha_pos)
        return alpha_pos, alpha_fit, curve, final_m


# ============================================================================
# 第6部分：IGWO vs GWO 对比图 (4张独立, 适用于LaTeX单栏)
# ============================================================================

def save_comparison_figures(sys, igwo_best, metrics_igwo, curve_igwo, fit_igwo,
                            gwo_best, metrics_gwo, curve_gwo, fit_gwo, outdir):
    r"""保存4张独立对比图，IEEE单栏宽度，适用于LaTeX \includegraphics"""
    eval_fn = FitnessEvaluator(sys)
    _, baseline = eval_fn.evaluate(np.zeros_like(igwo_best))
    h = np.arange(24)

    fig_w, fig_h = 3.5, 2.5  # IEEE single-column

    # ---- Fig 1: Active Power Loss ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(h, baseline['P_loss_h'], 'k-o', ms=3, lw=1.0,
            label='Baseline (Q=0)')
    ax.plot(h, metrics_gwo['P_loss_h'], 'b--s', ms=3, lw=1.0,
            label='Standard GWO')
    ax.plot(h, metrics_igwo['P_loss_h'], 'r-^', ms=3, lw=1.2,
            label='Improved GWO')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Active Power Loss (pu)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig1_active_power_loss', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 2: Voltage Deviation ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(h, baseline['V_dev_h'], 'k-o', ms=3, lw=1.0,
            label='Baseline (Q=0)')
    ax.plot(h, metrics_gwo['V_dev_h'], 'b--s', ms=3, lw=1.0,
            label='Standard GWO')
    ax.plot(h, metrics_igwo['V_dev_h'], 'r-^', ms=3, lw=1.2,
            label='Improved GWO')
    ax.set_xlabel('Hour')
    ax.set_ylabel(r'Voltage Deviation $\sum (V_i - V_{\mathrm{ref}})^2$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig2_voltage_deviation', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 3: Reactive Power Loss ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(h, baseline['Q_loss_h'], 'k-o', ms=3, lw=1.0,
            label='Baseline (Q=0)')
    ax.plot(h, metrics_gwo['Q_loss_h'], 'b--s', ms=3, lw=1.0,
            label='Standard GWO')
    ax.plot(h, metrics_igwo['Q_loss_h'], 'r-^', ms=3, lw=1.2,
            label='Improved GWO')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Reactive Power Loss (pu)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig3_reactive_power_loss', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    # ---- Fig 4: Convergence Curve ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(curve_igwo, 'r-', lw=1.2,
            label=f'IGWO (best={fit_igwo:.4f})')
    ax.plot(curve_gwo, 'b--', lw=1.2,
            label=f'Standard GWO (best={fit_gwo:.4f})')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.3)
    export.save(fig, 'fig4_convergence', outdir=outdir, formats=("pdf", "png"))
    plt.close()

    print(f"\n[Figs saved] {outdir}/fig1..4_*.{{pdf,png}}")


# ============================================================================
# 第7部分：主程序
# ============================================================================

def main():
    np.random.seed(42)
    print("=" * 65)
    print("  Improved GWO for All-Day Voltage & Reactive Power Optimization")
    print("  New Energy Power System — 7-bus Test Case")
    print("=" * 65)

    # [1] 初始化系统
    print("\n[1/5] Building test system...")
    sys = WindFarmSystem()
    print(f"  Nodes: {sys.n_nodes} | Devices: {sys.n_dev} "
          f"(WT:{sys.n_wt} SVG:1 Cap:1 OLTC:1)")
    print(f"  Search dimension: {sys.dim} (= {sys.n_dev} devices x 24h)")
    print(f"  Cap: discrete {sys.cap_n_steps+1} steps, OLTC: 9 taps, "
          f"max_sw={sys.max_cap_switches}/{sys.max_oltc_changes}")

    # [2] 初始化评估器
    print("\n[2/5] Initializing evaluator (Newton-Raphson PF)...")
    evaluator = FitnessEvaluator(sys)

    # [3] 运行IGWO
    print("\n[3/5] Running Improved GWO...")
    igwo = ImprovedGWO(evaluator, sys.lb, sys.ub,
                       n_wolves=30, max_iter=200,
                       a0=2.0, lam=1.8, k=1.5)
    t0 = time.time()
    best_igwo, fit_igwo, curve_igwo, a_vals, metrics_igwo = igwo.optimize(verbose=True)
    t_igwo = time.time() - t0
    print(f"\n  IGWO done in {t_igwo:.1f}s | Best fitness = {fit_igwo:.8f}")
    print(f"  P_loss = {metrics_igwo['P_loss']:.6f} pu")
    print(f"  Q_loss = {metrics_igwo['Q_loss']:.6f} pu")
    print(f"  V_rise = {metrics_igwo['V_rise']:.6f}")
    print(f"  V_drop = {metrics_igwo['V_drop']:.6f}")
    print(f"  Cap_sw_pen = {metrics_igwo.get('cap_sw_pen',0):.4f} | "
          f"OLTC_pen = {metrics_igwo.get('oltc_pen',0):.4f}")

    # [4] 运行标准GWO对比
    print("\n[4/5] Running Standard GWO for comparison...")
    gwo = StandardGWO(evaluator, sys.lb, sys.ub, n_wolves=30, max_iter=200)
    t1 = time.time()
    best_gwo, fit_gwo, curve_gwo, metrics_gwo = gwo.optimize(verbose=False)
    t_gwo = time.time() - t1
    print(f"  GWO done in {t_gwo:.1f}s | Best fitness = {fit_gwo:.8f}")

    # [5] 输出结果
    Q_opt = best_igwo.reshape(24, sys.n_dev)

    print("\n[5/5] Optimal dispatch (Q in MVar, OLTC as tap position):")
    hdr = f"  {'Hour':>6s}"
    for d in ['WT1', 'WT2', 'WT3', 'SVG', 'Cap(MVar)', 'OLTC(tap)']:
        hdr += f" {d:>10s}"
    print(hdr)
    print("  " + "-" * 70)
    n_q = sys.n_wt + sys.n_svg + sys.n_cap
    for h in [0, 6, 12, 18, 23]:
        q_vals = Q_opt[h, :n_q] * sys.SB    # MVar
        oltc = int(Q_opt[h, sys.idx_oltc])  # tap
        line = f"  {h:02d}:00  "
        for v in q_vals:
            line += f" {v:8.4f} "
        line += f" {oltc:10d}"
        print(line)

    # 优化前基准
    _, base_m = evaluator.evaluate(np.zeros_like(best_igwo))
    print(f"\n  --- Before vs IGWO vs GWO ---")
    print(f"  P_loss:  {base_m['P_loss']:.6f} -> IGWO={metrics_igwo['P_loss']:.6f} | "
          f"GWO={metrics_gwo['P_loss']:.6f} pu")
    print(f"  Q_loss:  {base_m['Q_loss']:.6f} -> IGWO={metrics_igwo['Q_loss']:.6f} | "
          f"GWO={metrics_gwo['Q_loss']:.6f} pu")
    print(f"  V_dev:   {base_m['V_rise']+base_m['V_drop']:.6f} -> "
          f"IGWO={metrics_igwo['V_rise']+metrics_igwo['V_drop']:.6f} | "
          f"GWO={metrics_gwo['V_rise']+metrics_gwo['V_drop']:.6f}")

    # 画图: 4张独立图
    outdir = r'D:\04_project\vscode'
    save_comparison_figures(sys, best_igwo, metrics_igwo, curve_igwo, fit_igwo,
                            best_gwo, metrics_gwo, curve_gwo, fit_gwo, outdir)

    print("\n" + "=" * 65)
    print(f"  Optimization complete! ({t_igwo:.1f}s IGWO | {t_gwo:.1f}s GWO)")
    print("=" * 65)


if __name__ == '__main__':
    main()
