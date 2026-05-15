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

        # 线路: [from, to, R(pu), X(pu)]  — ×6增强, 模拟更长馈线
        lines = np.array([
            [0, 1, 0.018, 0.072],
            [0, 2, 0.030, 0.108],
            [0, 6, 0.012, 0.048],
            [1, 4, 0.006, 0.024],
            [2, 5, 0.006, 0.024],
            [3, 4, 0.012, 0.048],
            [4, 5, 0.006, 0.018],
            [5, 6, 0.006, 0.030],
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
        # 线路电导 g_ij = R/(R²+X²) > 0 (用于计算P_loss, 区别于Ybus非对角元G_ij<0)
        self.line_g = np.array([r/(r*r + x*x) for _, _, r, x in lines])

        # 节点类型: 0=slack, 1=PQ (风机改为PQ, Q调度直接影响电压)
        self.node_type = np.array([0, 1, 1, 1, 1, 1, 1])
        self.slack_node = 0
        self.pq_nodes = np.where(self.node_type == 1)[0]   # [4, 5, 6]
        self.pv_nodes = np.where(self.node_type == 2)[0]   # [1, 2, 3]

        # 设备节点映射
        self.wt_nodes  = [1, 2, 3]    # 风机 (PV节点)
        self.svg_node  = 4            # SVG
        self.cap_node  = 5            # 电容器
        self.load_node = 6            # 负荷

        self.n_wt  = 3
        self.n_svg = 1
        self.n_cap = 1
        self.n_dev = self.n_wt + self.n_svg + self.n_cap  # 5

        # 搜索空间维度 = 设备数 × 24时段
        self.dim = self.n_dev * 24

        # ---- 设备无功出力范围 (pu on SB) — 扩大范围 ----
        self.wt_q_min  = np.array([-0.15, -0.15, -0.12])
        self.wt_q_max  = np.array([ 0.15,  0.15,  0.12])
        self.svg_q_min = -0.20
        self.svg_q_max =  0.20
        self.cap_q_min =  0.00
        self.cap_q_max =  0.10

        # ---- 改进2: 无功圆约束参数 (视在功率额定值) ----
        # 风机: S_wt = √(P_max²+Q_max²), 保证满发时仍有无功裕度
        #  WT1/WT2: P_max=2MW→0.20pu, Q_max=0.15pu → S≈√(0.20²+0.15²)=0.250pu (2.50MVA)
        #  WT3:     P_max=1.5MW→0.15pu, Q_max=0.12pu → S≈√(0.15²+0.12²)=0.192pu (1.92MVA)
        self.S_wt = np.array([0.250, 0.250, 0.192])  # pu
        self.S_svg = 0.20  # pu, SVG无有功出力, S=Q_max

        # 每时段静态盒式上下限（用于初始化和基础边界）
        ql_per_h = np.concatenate([self.wt_q_min, [self.svg_q_min], [self.cap_q_min]])
        qu_per_h = np.concatenate([self.wt_q_max, [self.svg_q_max], [self.cap_q_max]])
        self.lb = np.tile(ql_per_h, 24)
        self.ub = np.tile(qu_per_h, 24)

        # ---- 改进6: 电压约束具体参数 ----
        # V_min=0.90pu, V_max=1.10pu (IEC 60038允许±10%范围)
        # V_ref=1.00pu (标称电压)
        self.V_min = 0.90
        self.V_max = 1.10
        self.V_ref = 1.00

        # ---- 日曲线 ----
        self._build_profiles()

    def _build_profiles(self):
        """构建24h典型风电出力和负荷曲线"""
        t = np.arange(24)

        # 风电出力比（夜晚高/白天低的反调峰特性）
        wt_ratio = np.array([
            0.85, 0.88, 0.90, 0.87, 0.82, 0.75, 0.65, 0.55,
            0.45, 0.40, 0.38, 0.35, 0.33, 0.36, 0.42, 0.52,
            0.65, 0.75, 0.82, 0.85, 0.88, 0.90, 0.92, 0.88
        ])
        wt_rated_MW = np.array([2.0, 2.0, 1.5])

        self.P_wt = np.zeros((self.n_wt, 24))
        for i in range(self.n_wt):
            self.P_wt[i, :] = wt_ratio * wt_rated_MW[i] / self.SB

        # 负荷曲线（白天高/夜晚低）
        load_ratio = np.array([
            0.50, 0.45, 0.42, 0.40, 0.42, 0.55, 0.72, 0.85,
            0.92, 0.95, 0.97, 0.93, 0.88, 0.90, 0.95, 0.98,
            1.00, 0.95, 0.88, 0.82, 0.75, 0.70, 0.62, 0.55
        ])
        self.P_load = load_ratio * 5.0 / self.SB   # 峰值5MW (增大负荷应力)
        self.Q_load = load_ratio * 2.0 / self.SB   # 峰值2.0MVar


# ============================================================================
# 第2部分：潮流计算（牛顿-拉夫逊法，解析雅可比矩阵）
# ============================================================================

class PowerFlow:
    """牛顿-拉夫逊法潮流求解器（解析雅可比矩阵，高效实现）"""

    def __init__(self, sys: WindFarmSystem):
        self.sys = sys
        self.tol = 1e-8
        self.max_iter = 30

    def solve(self, P_inj, Q_inj):
        """
        输入: P_inj[n], Q_inj[n]  各节点注入功率
        输出: V[n], theta[n], converged
        """
        sys = self.sys
        G, B = sys.G, sys.B
        n = sys.n_nodes
        slack = sys.slack_node
        pq = sys.pq_nodes   # [4,5,6]
        pv = sys.pv_nodes   # [1,2,3]

        # 非平衡节点编号
        non_slack = np.concatenate([pv, pq])  # [1,2,3,4,5,6]
        n_ns = len(non_slack)

        # 初始电压
        V = np.ones(n)
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
        # 权重系数（改进4: 灵敏度分析中会改变这些值）
        self.w1 = 0.30         # 有功损耗
        self.w2_rise = 0.15    # 电压升高偏差
        self.w2_drop = 0.30    # 电压跌落偏差 (增大权重)
        self.w3 = 0.05         # 跨时段无功变化率 ΔQ (降低, 避免压制Q调度)
        self.lam = 100.0       # 电压越限惩罚系数
        self.mu = 50.0         # 无功圆约束惩罚系数

    def evaluate(self, position):
        """
        position: (n_dev * 24,) 决策变量向量
        返回: (fitness, metrics_dict)
        """
        sys = self.sys
        Q_dev = position.reshape(24, sys.n_dev)  # (24, 5)

        total_P_loss = 0.0
        total_V_rise  = 0.0
        total_V_drop  = 0.0
        total_V_pen   = 0.0
        total_delta_Q = 0.0
        total_circle_pen = 0.0
        V_profile = np.zeros((24, sys.n_nodes))
        Q_applied = np.zeros_like(Q_dev)  # 记录圆约束裁剪后的实际Q

        for h in range(24):
            q = Q_dev[h].copy()

            # ---- 改进2: 无功圆约束 ----
            # 风机无功: Q² ≤ S_wt² − P_wt²
            for w in range(sys.n_wt):
                S = sys.S_wt[w]
                P = sys.P_wt[w, h]
                max_q_h = np.sqrt(max(S**2 - P**2, 0.0))
                if abs(q[w]) > max_q_h:
                    total_circle_pen += (abs(q[w]) - max_q_h)**2
                    q[w] = np.clip(q[w], -max_q_h, max_q_h)

            # SVG无功: Q² ≤ S_svg² (SVG无有功出力)
            max_q_svg = sys.S_svg
            if abs(q[sys.n_wt]) > max_q_svg:
                total_circle_pen += (abs(q[sys.n_wt]) - max_q_svg)**2
                q[sys.n_wt] = np.clip(q[sys.n_wt], -max_q_svg, max_q_svg)

            Q_applied[h] = q

            # 构建节点注入
            P_inj = np.zeros(sys.n_nodes)
            Q_inj = np.zeros(sys.n_nodes)

            # 风机有功注入
            for w in range(sys.n_wt):
                P_inj[sys.wt_nodes[w]] = sys.P_wt[w, h]

            # 负荷 (负注入)
            P_inj[sys.load_node] = -sys.P_load[h]
            Q_inj[sys.load_node] = -sys.Q_load[h]

            # 设备无功注入 (使用圆约束裁剪后的值)
            for w in range(sys.n_wt):
                Q_inj[sys.wt_nodes[w]] = q[w]
            Q_inj[sys.svg_node] = q[sys.n_wt]
            Q_inj[sys.cap_node] = q[sys.n_wt + 1]

            # 潮流计算
            V, theta, ok = self.pf.solve(P_inj, Q_inj)
            if not ok:
                return 1e10, {
                    'P_loss': 1e10, 'V_rise': 1e10, 'V_drop': 1e10,
                    'V_pen': 1e10, 'delta_Q': 1e10, 'circle_pen': 1e10,
                    'converged': False, 'V_profile': None, 'Q_applied': None
                }

            V_profile[h] = V

            # 线路有功损耗
            for line_idx, (f, t, r, x) in enumerate(sys.lines):
                f, t = int(f), int(t)
                g = sys.line_g[line_idx]
                ang = theta[f] - theta[t]
                P_loss_ij = g * (V[f]**2 + V[t]**2 -
                                 2 * V[f] * V[t] * np.cos(ang))
                total_P_loss += P_loss_ij

            # ---- 改进3: 拆分电压升高和跌落 ----
            for i in range(sys.n_nodes):
                dv = V[i] - sys.V_ref
                if dv > 0:
                    total_V_rise += dv**2
                else:
                    total_V_drop += dv**2

            # 电压越限惩罚
            for i in range(sys.n_nodes):
                if V[i] < sys.V_min:
                    total_V_pen += (sys.V_min - V[i])**2
                elif V[i] > sys.V_max:
                    total_V_pen += (V[i] - sys.V_max)**2

        # ---- 改进1+7: 跨时段无功变化率 (替代原专利Δu) ----
        # 罚相邻时段设备无功出力的跳变，建立真正的全天时耦合
        for h in range(23):
            for d in range(sys.n_dev):
                total_delta_Q += (Q_applied[h+1, d] - Q_applied[h, d])**2

        fitness = (self.w1 * total_P_loss +
                   self.w2_rise * total_V_rise +
                   self.w2_drop * total_V_drop +
                   self.w3 * total_delta_Q +
                   self.lam * total_V_pen +
                   self.mu * total_circle_pen)

        return fitness, {
            'P_loss': total_P_loss,
            'V_rise': total_V_rise,
            'V_drop': total_V_drop,
            'V_pen': total_V_pen,
            'delta_Q': total_delta_Q,
            'circle_pen': total_circle_pen,
            'converged': True,
            'V_profile': V_profile,
            'Q_applied': Q_applied,
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

        return alpha_pos, alpha_fit, curve


# ============================================================================
# 第6部分：权重灵敏度分析（改进4）
# ============================================================================

def sensitivity_analysis(sys, base_evaluator, save_to=None):
    """
    对照论文 Fig.7, 改变各子目标的权重系数, 观察对应子目标值的变化趋势
    验证优化结果的鲁棒性, 确定合理的权重取值范围
    """
    import sys as _sys
    print("\n" + "=" * 65)
    print("  改进4: 权重灵敏度分析")
    print("=" * 65)
    _sys.stdout.flush()

    # 轻量参数: 20 wolves, 50 iterations (灵敏度分析不需收敛到极致)
    n_wolves = 20
    max_iter = 50

    # ---- 测试1: w1(有功损耗) 从0.10→0.70 变化 (5点) ----
    print("\n  [Test 1] Varying w1 (P_loss weight)...")
    _sys.stdout.flush()
    w1_values = np.arange(0.10, 0.75, 0.15)  # 5点 (原7点→减半)
    results_w1 = {'w1': [], 'P_loss': [], 'V_rise': [], 'V_drop': [],
                  'delta_Q': [], 'V_pen': [], 'circle_pen': []}

    for w1_val in w1_values:
        remain = 1.0 - w1_val
        w_rise = remain * 0.25
        w_drop = remain * 0.30
        w_delta = remain * 0.25

        evaluator = FitnessEvaluator(sys)
        evaluator.w1 = w1_val
        evaluator.w2_rise = w_rise
        evaluator.w2_drop = w_drop
        evaluator.w3 = w_delta

        igwo = ImprovedGWO(evaluator, sys.lb, sys.ub,
                          n_wolves=n_wolves, max_iter=max_iter)
        _, _, _, _, metrics = igwo.optimize(verbose=False)
        results_w1['w1'].append(w1_val)
        results_w1['P_loss'].append(metrics['P_loss'])
        results_w1['V_rise'].append(metrics['V_rise'])
        results_w1['V_drop'].append(metrics['V_drop'])
        results_w1['delta_Q'].append(metrics['delta_Q'])
        results_w1['V_pen'].append(metrics['V_pen'])
        results_w1['circle_pen'].append(metrics['circle_pen'])
        print(f"    w1={w1_val:.2f} | P_loss={metrics['P_loss']:.6f} | "
              f"V_rise={metrics['V_rise']:.6f} | V_drop={metrics['V_drop']:.6f} | "
              f"ΔQ={metrics['delta_Q']:.6f}")
        _sys.stdout.flush()

    # ---- 测试2: w3(ΔQ) 从0.05→0.50 变化 (5点) ----
    print("\n  [Test 2] Varying w3 (ΔQ weight)...")
    _sys.stdout.flush()
    w3_values = np.arange(0.05, 0.55, 0.10)  # 5点
    results_w3 = {'w3': [], 'P_loss': [], 'V_rise': [], 'V_drop': [],
                  'delta_Q': [], 'V_pen': [], 'circle_pen': []}

    for w3_val in w3_values:
        remain = 1.0 - w3_val
        w1 = remain * 0.35
        w_rise = remain * 0.25
        w_drop = remain * 0.35
        w_delta = w3_val

        evaluator = FitnessEvaluator(sys)
        evaluator.w1 = w1
        evaluator.w2_rise = w_rise
        evaluator.w2_drop = w_drop
        evaluator.w3 = w3_val

        igwo = ImprovedGWO(evaluator, sys.lb, sys.ub,
                          n_wolves=n_wolves, max_iter=max_iter)
        _, _, _, _, metrics = igwo.optimize(verbose=False)
        results_w3['w3'].append(w3_val)
        results_w3['P_loss'].append(metrics['P_loss'])
        results_w3['V_rise'].append(metrics['V_rise'])
        results_w3['V_drop'].append(metrics['V_drop'])
        results_w3['delta_Q'].append(metrics['delta_Q'])
        results_w3['V_pen'].append(metrics['V_pen'])
        results_w3['circle_pen'].append(metrics['circle_pen'])
        print(f"    w3={w3_val:.2f} | P_loss={metrics['P_loss']:.6f} | "
              f"V_rise={metrics['V_rise']:.6f} | V_drop={metrics['V_drop']:.6f} | "
              f"ΔQ={metrics['delta_Q']:.6f}")
        _sys.stdout.flush()

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensitivity Analysis — Effect of Weight Factors on Sub-Objectives\n'
                 '(对照论文 Mahmoud 2020 Fig.7)', fontsize=13, fontweight='bold')

    # Row 1: w1 variation
    ax = axes[0, 0]
    ax.plot(results_w1['w1'], results_w1['P_loss'], 'b-o', lw=1.5, ms=6)
    ax.set_xlabel('w1 (P_loss weight)')
    ax.set_ylabel('P_loss (pu)')
    ax.set_title('P_loss vs w1')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results_w1['w1'], results_w1['V_rise'], 'r-s', lw=1.5, ms=6, label='V_rise')
    ax.plot(results_w1['w1'], results_w1['V_drop'], 'b-^', lw=1.5, ms=6, label='V_drop')
    ax.set_xlabel('w1 (P_loss weight)')
    ax.set_ylabel('Voltage Deviation')
    ax.set_title('V_rise / V_drop vs w1')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(results_w1['w1'], results_w1['delta_Q'], 'g-D', lw=1.5, ms=6)
    ax.set_xlabel('w1 (P_loss weight)')
    ax.set_ylabel('ΔQ')
    ax.set_title('ΔQ vs w1')
    ax.grid(True, alpha=0.3)

    # Row 2: w3 variation
    ax = axes[1, 0]
    ax.plot(results_w3['w3'], results_w3['P_loss'], 'b-o', lw=1.5, ms=6)
    ax.set_xlabel('w3 (ΔQ weight)')
    ax.set_ylabel('P_loss (pu)')
    ax.set_title('P_loss vs w3')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(results_w3['w3'], results_w3['V_rise'], 'r-s', lw=1.5, ms=6, label='V_rise')
    ax.plot(results_w3['w3'], results_w3['V_drop'], 'b-^', lw=1.5, ms=6, label='V_drop')
    ax.set_xlabel('w3 (ΔQ weight)')
    ax.set_ylabel('Voltage Deviation')
    ax.set_title('V_rise / V_drop vs w3')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(results_w3['w3'], results_w3['delta_Q'], 'g-D', lw=1.5, ms=6)
    ax.set_xlabel('w3 (ΔQ weight)')
    ax.set_ylabel('ΔQ')
    ax.set_title('ΔQ vs w3')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_to:
        _stem = _osp.splitext(_osp.basename(save_to))[0]
        _dir = _osp.dirname(save_to) or "."
        export.save(fig, _stem, outdir=_dir, formats=("pdf", "png"))
        print(f"\n  [Sensitivity fig saved] {_dir}/{_stem}.{{pdf,png}}")
    plt.close()
    return results_w1, results_w3

def plot_results(sys, best_pos, best_fit, curve_igwo, a_vals, metrics,
                 curve_gwo=None, fit_gwo=None, save_to=None):
    """综合结果可视化（含改进后的新指标）"""
    Q_opt = metrics.get('Q_applied')
    if Q_opt is None:
        Q_opt = best_pos.reshape(24, sys.n_dev)
    V_opt = metrics['V_profile']

    # 优化前基准：无功出力=0
    zero_eval = FitnessEvaluator(sys)
    _, base_m = zero_eval.evaluate(np.zeros_like(best_pos))
    V_base = base_m.get('V_profile')
    if V_base is None:
        V_base = np.ones((24, sys.n_nodes))

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    fig.suptitle('IGWO 改进灰狼算法 — 全天时电压/无功优化结果 (对照论文框架)', fontsize=13, fontweight='bold')

    h = np.arange(24)

    # [0,0] 收敛曲线对比
    ax = axes[0, 0]
    ax.plot(curve_igwo, 'b-', lw=1.2, label=f'IGWO best={best_fit:.4f}')
    if curve_gwo is not None and fit_gwo is not None:
        ax.plot(curve_gwo, 'r--', lw=1.2, label=f'GWO best={fit_gwo:.4f}')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.set_title('Convergence Curve (IGWO vs GWO)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] 非线性收敛因子 a
    ax = axes[0, 1]
    ax.plot(a_vals, 'orange', lw=1.8)
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('a')
    ax.set_title('Nonlinear Convergence Factor')
    ax.grid(True, alpha=0.3)

    # [0,2] 电压优化前后对比 + 升高/跌落分区
    ax = axes[0, 2]
    n_show = 1  # WT1节点
    ax.plot(h, V_base[:, n_show], 'b-o', ms=3, label='Before', alpha=0.7)
    ax.plot(h, V_opt[:, n_show], 'r-s', ms=3, label='After', alpha=0.7)
    ax.axhline(sys.V_min, color='gray', ls='--', label=f'Vmin={sys.V_min}')
    ax.axhline(sys.V_max, color='gray', ls='--', label=f'Vmax={sys.V_max}')
    ax.axhline(sys.V_ref, color='green', ls=':', label='Vref=1.0')
    # 填充升高/跌落区域 (改进3)
    ax.fill_between(h, sys.V_ref, sys.V_max, alpha=0.08, color='red', label='Rise zone')
    ax.fill_between(h, sys.V_min, sys.V_ref, alpha=0.08, color='blue', label='Drop zone')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Voltage (pu)')
    ax.set_title(f'Node {n_show} (WT1) Voltage')
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    # [0,3] V_rise / V_drop per hour (改进3)
    ax = axes[0, 3]
    V_rise_h = np.sum(np.maximum(V_opt - sys.V_ref, 0)**2, axis=1)
    V_drop_h = np.sum(np.maximum(sys.V_ref - V_opt, 0)**2, axis=1)
    ax.bar(h - 0.15, V_rise_h, 0.3, color='red', alpha=0.7, label='V_rise')
    ax.bar(h + 0.15, V_drop_h, 0.3, color='blue', alpha=0.7, label='V_drop')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Deviation')
    ax.set_title('V_rise / V_drop per Hour (改进3)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [1,0] 无功出力热力图 (含圆约束边界)
    ax = axes[1, 0]
    labels = ['WT1', 'WT2', 'WT3', 'SVG', 'Cap']
    im = ax.imshow(Q_opt.T, aspect='auto', cmap='RdBu_r',
                   vmin=-0.10, vmax=0.10, interpolation='nearest')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Device')
    ax.set_yticks(range(sys.n_dev))
    ax.set_yticklabels(labels)
    ax.set_title('Optimal Q Dispatch (pu) [with circle constraint]')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # [1,1] 无功出力变化率 ΔQ per transition (改进1)
    ax = axes[1, 1]
    dQ_h = np.zeros(23)
    for hh in range(23):
        dQ_h[hh] = np.sum((Q_opt[hh+1] - Q_opt[hh])**2)
    ax.bar(np.arange(23), dQ_h, 0.6, color='purple', alpha=0.7)
    ax.set_xlabel('Hour transition')
    ax.set_ylabel('ΔQ')
    ax.set_title('ΔQ per Transition (改进1: cross-time coupling)')
    ax.grid(True, alpha=0.3)

    # [1,2] 风机无功圆约束可视化 (改进2)
    ax = axes[1, 2]
    for w in range(sys.n_wt):
        P_h = sys.P_wt[w]
        S = sys.S_wt[w]
        q_max_h = np.sqrt(np.maximum(S**2 - P_h**2, 0))
        ax.fill_between(h, -q_max_h, q_max_h, alpha=0.15,
                        label=f'WT{w+1} Q-bound')
        ax.plot(h, Q_opt[:, w], 'o-', ms=3, lw=1.2, label=f'WT{w+1} Q')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Q (pu)')
    ax.set_title('WT Q Limits (circle constraint, 改进2)')
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    # [1,3] 数值总结
    ax = axes[1, 3]
    ax.axis('off')
    loss_reduction = (1 - metrics['P_loss']/max(base_m['P_loss'], 1e-10)) * 100
    v_rise_reduction = (1 - metrics['V_rise']/max(base_m['V_rise'], 1e-10)) * 100
    v_drop_reduction = (1 - metrics['V_drop']/max(base_m['V_drop'], 1e-10)) * 100
    summary = (
        "==== Optimization Summary ====\n\n"
        f"Best fitness:       {best_fit:.6f}\n"
        f"Total P_loss:       {metrics['P_loss']:.6f} pu\n"
        f"Loss reduction:     {loss_reduction:.2f}%\n"
        f"V_rise:             {metrics['V_rise']:.6f} ({v_rise_reduction:.1f}%)\n"
        f"V_drop:             {metrics['V_drop']:.6f} ({v_drop_reduction:.1f}%)\n"
        f"ΔQ (cross-time):    {metrics['delta_Q']:.6f}\n"
        f"V_penalty:          {metrics['V_pen']:.6f}\n"
        f"Circle_pen:         {metrics['circle_pen']:.6f}\n\n"
        f"--- Core Improvements ---\n"
        f"1. Cross-time ΔQ\n"
        f"2. Circle constraint\n"
        f"3. Split V_rise/V_drop\n"
        f"4. Nonlinear a(t)\n"
        f"5. Delta fusion\n"
        f"6. Weighted position update"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    if save_to:
        _stem = _osp.splitext(_osp.basename(save_to))[0]
        _dir = _osp.dirname(save_to) or "."
        export.save(fig, _stem, outdir=_dir, formats=("pdf", "png"))
        print(f"[Fig saved] {_dir}/{_stem}.{{pdf,png}}")
    plt.close()
    return fig


# ============================================================================
# 第7部分：主程序
# ============================================================================

def main():
    print("=" * 65)
    print("  Improved GWO for All-Day Voltage & Reactive Power Optimization")
    print("  New Energy Power System — 7-bus Test Case")
    print("=" * 65)

    # [1] 初始化系统
    print("\n[1/5] Building test system...")
    sys = WindFarmSystem()
    print(f"  Nodes: {sys.n_nodes} | Devices: {sys.n_dev} "
          f"(WT:{sys.n_wt} SVG:1 Cap:1)")
    print(f"  Search dimension: {sys.dim} (= {sys.n_dev} devices x 24h)")

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
    print(f"  V_rise = {metrics_igwo['V_rise']:.6f}")
    print(f"  V_drop = {metrics_igwo['V_drop']:.6f}")
    print(f"  ΔQ     = {metrics_igwo['delta_Q']:.6f}")
    print(f"  Circle_pen = {metrics_igwo['circle_pen']:.6f}")

    # [4] 运行标准GWO对比
    print("\n[4/5] Running Standard GWO for comparison...")
    gwo = StandardGWO(evaluator, sys.lb, sys.ub, n_wolves=30, max_iter=200)
    t1 = time.time()
    best_gwo, fit_gwo, curve_gwo = gwo.optimize(verbose=False)
    t_gwo = time.time() - t1
    print(f"  GWO done in {t_gwo:.1f}s | Best fitness = {fit_gwo:.8f}")

    # [5] 输出结果
    Q_opt = best_igwo.reshape(24, sys.n_dev)

    print("\n[5/5] Optimal dispatch (selected hours, in MVar):")
    hdr = f"  {'Hour':>6s}"
    for d in ['WT1', 'WT2', 'WT3', 'SVG', 'Cap']:
        hdr += f" {d:>10s}"
    print(hdr)
    print("  " + "-" * 58)
    for h in [0, 6, 12, 18, 23]:
        vals = Q_opt[h] * sys.SB
        line = f"  {h:02d}:00  "
        for v in vals:
            line += f" {v:8.4f} "
        print(line)

    # 优化前基准
    _, base_m = evaluator.evaluate(np.zeros_like(best_igwo))
    print(f"\n  --- Before vs After ---")
    print(f"  P_loss:  {base_m['P_loss']:.6f} -> {metrics_igwo['P_loss']:.6f} pu "
          f"({(1-metrics_igwo['P_loss']/max(base_m['P_loss'],1e-10))*100:.1f}%)")
    print(f"  V_rise:  {base_m['V_rise']:.6f} -> {metrics_igwo['V_rise']:.6f} "
          f"({(1-metrics_igwo['V_rise']/max(base_m['V_rise'],1e-10))*100:.1f}%)")
    print(f"  V_drop:  {base_m['V_drop']:.6f} -> {metrics_igwo['V_drop']:.6f} "
          f"({(1-metrics_igwo['V_drop']/max(base_m['V_drop'],1e-10))*100:.1f}%)")
    print(f"  ΔQ:      {base_m['delta_Q']:.6f} -> {metrics_igwo['delta_Q']:.6f}")
    print(f"  V_pen:   {base_m['V_pen']:.6f} -> {metrics_igwo['V_pen']:.6f}")
    print(f"  Circle_pen: {base_m['circle_pen']:.6f} -> {metrics_igwo['circle_pen']:.6f}")

    # 画图
    plot_results(sys, best_igwo, fit_igwo, curve_igwo, a_vals, metrics_igwo,
                 curve_gwo=curve_gwo, fit_gwo=fit_gwo,
                 save_to=r'D:\04_project\vscode\igwo_optimization_results.png')
    print("\n  Results saved to: D:/04_project/vscode/igwo_optimization_results.png")

    # ---- 改进4: 权重灵敏度分析 ----
    sensitivity_analysis(sys, evaluator,
                        save_to=r'D:\04_project\vscode\sensitivity_analysis.png')

    print("\n" + "=" * 65)
    print(f"  Optimization complete! ({t_igwo:.1f}s IGWO | {t_gwo:.1f}s GWO)")
    print("=" * 65)


if __name__ == '__main__':
    main()
