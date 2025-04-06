import numpy as np


class UCEDAdjuster:
    def __init__(self, constraints, config):
        """
        第二阶段调整器

        参数：
            constraints (dict): 包含以下键：
                - P_min (np.ndarray): [N] 各机组最小出力
                - P_max (np.ndarray): [N] 各机组最大出力
                - ramp_up (np.ndarray): [N] 向上爬坡率
                - ramp_down (np.ndarray): [N] 向下爬坡率
                - min_on (np.ndarray): [N] 最小开机时间
                - min_off (np.ndarray): [N] 最小停机时间
                - cost_a (np.ndarray): [N] 二次成本函数系数 (二次项)
                - cost_b (np.ndarray): [N] 二次成本函数系数 (一次项)
                - cost_c (np.ndarray): [N] 二次成本函数系数 (常数项)
            config (dict): 算法超参数
        """
        self.constraints = constraints
        self.config = {
            'max_iter_power_balance': 100,
            'power_tolerance': 1e-3,
            'ramp_adjust_steps': 50,
            **config
        }

    def __call__(self, nn_output, system_demand):
        """
        主调整流程

        参数：
            nn_output (np.ndarray): [T, N] 神经网络输出
            system_demand (np.ndarray): [T] 系统总需求

        返回：
            adjusted (np.ndarray): [T, N] 调整后的可行解
        """
        # Step 1: 硬约束初步修正（使用sigmoid函数）
        clamped = self._sigmoid_clamp_generation(nn_output)

        # Step 2: 启停时间约束处理
        on_off_schedule = self._min_uptime_downtime(clamped)

        # Step 3: 功率平衡迭代
        balanced = self._iterative_power_balance(on_off_schedule, system_demand)

        # Step 4: 爬坡率调整
        ramp_fixed = self._adjust_ramp_rates(balanced)

        return ramp_fixed

    def _sigmoid(self, x):
        """Sigmoid函数实现"""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_clamp_generation(self, x):
        """使用sigmoid函数进行出力上下限硬约束修正"""
        T, N = x.shape
        clamped = x.copy()
        for i in range(N):
            for t in range(T):
                if x[t, i] < self.constraints['P_min'][i]:
                    p_on = self._sigmoid(x[t, i] - self.constraints['P_min'][i])
                    if p_on > 0.5:
                        clamped[t, i] = self.constraints['P_min'][i]
                    else:
                        clamped[t, i] = 0
        return np.clip(clamped, self.constraints['P_min'], self.constraints['P_max'])

    def _min_uptime_downtime(self, x):
        """处理最小启停时间约束（动态规划方法）"""
        T, N = x.shape
        adjusted = x.copy()
        on_off = (x > 0).astype(int)

        for i in range(N):
            state = on_off[0, i]
            counter = 0

            for t in range(1, T):
                if on_off[t, i] == state:
                    counter += 1
                else:
                    min_time = self.constraints['min_on'][i] if state == 1 else self.constraints['min_off'][i]
                    if counter < min_time:
                        # 回退到前一个有效状态
                        adjusted[t - min_time:t, i] = x[t - min_time, i] if state == 1 else 0
                        on_off[t - min_time:t, i] = state
                        t -= min_time  # 重置计数器
                    state = on_off[t, i]
                    counter = 0
        return adjusted

    def _iterative_power_balance(self, x, demand):
        """按照成本优化的功率平衡迭代（并行权重分配法）"""
        T, N = x.shape
        adjusted = x.copy()

        # 迭代直到满足功率平衡或达到最大迭代次数
        for _ in range(self.config['max_iter_power_balance']):
            # 计算功率不平衡
            imbalance = demand - np.sum(adjusted, axis=1)

            # 如果所有时段平衡，退出循环
            if np.all(np.abs(imbalance) < self.config['power_tolerance']):
                break

            # 对每个时段进行基于成本的功率平衡
            for t in range(T):
                # 如果该时段已平衡，跳过
                if abs(imbalance[t]) < self.config['power_tolerance']:
                    continue

                # 计算边际成本
                marginal_costs = self._calculate_marginal_costs(adjusted[t])

                # 基于成本选择可调节机组
                selected_units = self._cost_guided_select(adjusted[t], imbalance[t], marginal_costs)

                # 如果没有可调机组，跳到下一个时段
                if len(selected_units) == 0:
                    continue

                # 计算并行权重分配
                for i in selected_units:
                    # 计算调节空间
                    if imbalance[t] > 0:  # 需要增加发电
                        margin = self.constraints['P_max'][i] - adjusted[t, i]
                    else:  # 需要减少发电
                        margin = adjusted[t, i] - self.constraints['P_min'][i]

                    # 如果没有调节空间，跳过该机组
                    if margin <= 0:
                        continue

                    # 计算机组权重（与边际成本成反比）
                    weights = {j: self._calculate_weight(adjusted[t, j], marginal_costs[j],
                                                         imbalance[t] > 0) for j in selected_units
                               if (imbalance[t] > 0 and self.constraints['P_max'][j] > adjusted[t, j]) or
                               (imbalance[t] < 0 and adjusted[t, j] > self.constraints['P_min'][j])}

                    # 如果没有有效权重，跳过
                    if not weights:
                        continue

                    # 权重归一化
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        normalized_weights = {j: w / total_weight for j, w in weights.items()}
                    else:
                        continue

                    # 分配功率调整
                    remaining_imbalance = imbalance[t]
                    for j, weight in normalized_weights.items():
                        # 计算分配到该机组的调整量
                        delta = weight * remaining_imbalance

                        # 应用调整并确保在约束范围内
                        old_value = adjusted[t, j]
                        adjusted[t, j] = np.clip(
                            old_value + delta,
                            self.constraints['P_min'][j],
                            self.constraints['P_max'][j]
                        )

                        # 更新剩余不平衡量
                        actual_delta = adjusted[t, j] - old_value
                        remaining_imbalance -= actual_delta

                        # 如果已达到平衡，退出循环
                        if abs(remaining_imbalance) < self.config['power_tolerance']:
                            break

        return adjusted

    def _calculate_marginal_costs(self, power):
        """计算当前功率下的边际成本"""
        # 边际成本 = 2*a*P + b
        return 2 * self.constraints['cost_a'] * power + self.constraints['cost_b']

    def _cost_guided_select(self, power, imbalance, marginal_costs):
        """基于成本和调节方向选择可调节机组"""
        N = len(power)
        unit_status = power > 0  # 机组开/关状态

        if imbalance > 0:  # 需要增加发电
            # 筛选出已开机且未达最大出力的机组
            candidates = np.where(
                (unit_status) &
                (power < self.constraints['P_max'])
            )[0]

            # 按边际成本从低到高排序
            return candidates[np.argsort(marginal_costs[candidates])]

        else:  # 需要减少发电
            # 筛选出已开机且高于最小出力的机组
            candidates = np.where(
                (unit_status) &
                (power > self.constraints['P_min'])
            )[0]

            # 按边际成本从高到低排序
            return candidates[np.argsort(-marginal_costs[candidates])]

    def _calculate_weight(self, power, marginal_cost, is_increase):
        """计算机组调整权重（与边际成本成反比）"""
        # 避免除零
        if marginal_cost <= 0:
            return 0

        #权重与边际成本成反比
        return 1.0 / marginal_cost

    def _adjust_ramp_rates(self, x):
        """爬坡率约束修正（前向扫描法）"""
        T, N = x.shape
        adjusted = x.copy()

        for i in range(N):
            for t in range(1, T):
                max_increase = self.constraints['ramp_up'][i]
                max_decrease = self.constraints['ramp_down'][i]
                delta = adjusted[t, i] - adjusted[t - 1, i]

                if delta > max_increase:
                    adjusted[t, i] = adjusted[t - 1, i] + max_increase
                elif delta < -max_decrease:
                    adjusted[t, i] = adjusted[t - 1, i] - max_decrease

        return adjusted
