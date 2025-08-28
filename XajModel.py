import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Xaj:
    def __init__(self,
                 KI = 0.6622097738172629,
                 CS = 0.9277029194349287,
                 CI = 0.5099229728929486,
                 CG = 0.9969022040280667,
                 Kc = 0.5003898723919551,
                 B  = 0.11893920020326171,
                 IM = 0.02,
                 EX = 1.6700462559883593,
                 SM = 73.01742324756884,
                 L  = 0.08001649614952838):
        self.KI = KI
        self.KG = 0.7 - KI
        self.CS = CS
        self.CI = CI
        self.CG = CG
        self.Kc = Kc
        self.B = B
        self.IM = IM
        self.EX = EX
        self.SM = SM
        self.L = L

        # 固定参数
        self.WUM = 15
        self.WLM = 60
        self.WDM = 50
        self.WM  = 125
        self.C   = 0.0686525983064172
        self.F   = 2087

        self.WMM = self.WM * (1 + self.B) / (1 - self.IM)

    def water_division(self, S1, FR0, PE, R):
        if R == 0 or PE == 0:
            RS = 0
            RG = S1 * self.KG * FR0
            RI = S1 * self.KI * FR0
            S2 = S1 * 0.2
        else:
            FR = R / PE
            Smm = (1 + self.EX) * self.SM
            AU = Smm * (1 - (1 - S1 * FR0 / FR / self.SM) ** (1 / (1 + self.EX)))
            if PE + AU < Smm:
                RS = FR*(PE + S1*FR0/FR - self.SM + self.SM*(1 - (PE + AU)/Smm)**(self.EX + 1))
            else:
                RS = FR*(PE + S1*FR0/FR - self.SM)
            S = S1*FR0/FR + (R - RS)/FR
            RI = self.KI * S * FR
            RG = self.KG * S * FR
            S2 = S * 0.2

        RS = max(RS, 0)
        RI = max(RI, 0)
        RG = max(RG, 0)
        S2 = max(min(S2, self.SM), 0)

        return RS, RI, RG, S2

    def qs_calculation(self, FR, QS0, RS):
        return self.CS * QS0 + (1 - self.CS) * RS * self.F * FR / (24 * 3.6)

    def qi_calculation(self, FR, QI0, RI):
        return self.CI * QI0 + (1 - self.CI) * RI * self.F * FR / (24 * 3.6)

    def qg_calculation(self, FR, QG0, RG):
        return self.CG * QG0 + (1 - self.CG) * RG * self.F * FR / (24 * 3.6)

    def q_calculation(self, Q0, QT):
        return self.CS * Q0 + (1 - self.CS) * QT * (1 - self.L)

    def e_calculation(self, P, WU, EP, WL):
        EU = WU + P
        if EU >= EP:
            EU = EP
            EL = 0
            ED = 0
        else:
            if WL >= self.C * self.WLM:
                EL = (EP - EU) * WL / self.WLM
                ED = 0
            else:
                if WL >= self.C * (EP - EU):
                    EL = self.C * (EP - EU)
                    ED = 0
                else:
                    EL = WL
                    ED = self.C * (EP - EU) - EL
        E = EU + EL + ED
        return EU, EL, ED, E

    def w_calculation(self, WU, WL, WD, EU, EL, ED, P, R):
        WU_new = WU + P - EU - R
        WL_new = WL - EL
        WD_new = WD - ED

        if WU_new > self.WUM:
            WL_new += WU_new - self.WUM
            WU_new = self.WUM
        if WL_new > self.WLM:
            WD_new += WL_new - self.WLM
            WL_new = self.WLM
        if WD_new > self.WDM:
            WD_new = self.WDM

        W = WU_new + WL_new + WD_new
        return WU_new, WL_new, WD_new, W

    def r_calculation(self, W, P, E):
        x = 1 - W / self.WM
        y = 1 / (1 + self.B)
        a = self.WMM * (1 - x**y)
        if P == 0:
            R = 0
        else:
            if (a + P - E) <= self.WMM:
                m = 1 - (P - E + a) / self.WMM
                n = 1 + self.B
                R = P - E + W - self.WM + self.WM * m**n
            else:
                R = P - E + W - self.WM
        RR = (1 - self.IM) * R + self.IM * (P - E)
        return max(RR, 0)


def run_model(filepath, output_path=None):
    df = pd.read_excel(filepath, header=None)
    arr = df.values

    # 使用所有数据行
    P = arr[:, 2]
    E0 = arr[:, 3]
    Q_measurement = arr[:, 1]
    l = len(P)

    xaj = Xaj()

    EP = xaj.Kc * E0

    # 初始化变量
    EU = np.zeros(l)
    EL = np.zeros(l)
    ED = np.zeros(l)
    E = np.zeros(l)
    WU = np.zeros(l)
    WL = np.zeros(l)
    WD = np.zeros(l)
    W = np.zeros(l)
    R = np.zeros(l)
    PE = np.zeros(l)

    # 初始化初值
    WU[0] = xaj.WUM
    WL[0] = xaj.WLM
    WD[0] = xaj.WDM
    W[0] = WU[0] + WL[0] + WD[0]

    # 蒸散发和水量计算循环
    for i in range(1, l):
        WU[i], WL[i], WD[i], W[i] = xaj.w_calculation(WU[i-1], WL[i-1], WD[i-1], EU[i-1], EL[i-1], ED[i-1], P[i-1], R[i-1])
        EU[i], EL[i], ED[i], E[i] = xaj.e_calculation(P[i], WU[i], EP[i], WL[i])
        R[i] = xaj.r_calculation(W[i-1], P[i], E[i])
        PE[i] = P[i] - E[i]

    # 汇流计算初始化
    RS = np.zeros(l)
    RI = np.zeros(l)
    RG = np.zeros(l)
    S = np.zeros(l)
    FR = np.zeros(l)
    QS = np.zeros(l)
    QI = np.zeros(l)
    QG = np.zeros(l)
    QT = np.zeros(l)
    Q = np.zeros(l)

    # 初值设置
    Q[0] = 1.90
    QS[0] = 0.9
    QI[0] = 0.5
    QG[0] = 0.5
    QT[0] = QS[0] + QI[0] + QG[0]
    FR[0] = 0.4

    for i in range(1, l):
        FR[i] = R[i]/PE[i] if PE[i] > 0 else 0
        RS[i], RI[i], RG[i], S[i] = xaj.water_division(S[i-1], FR[i-1], PE[i], R[i])
        QS[i] = xaj.qs_calculation(FR[i], QS[i-1], RS[i])
        QI[i] = xaj.qi_calculation(FR[i], QI[i-1], RI[i])
        QG[i] = xaj.qg_calculation(FR[i], QG[i-1], RG[i])
        QT[i] = QS[i] + QI[i] + QG[i]
        Q[i] = xaj.q_calculation(Q[i-1], QT[i])

    # 评价指标
    peak_err = (np.max(Q) - np.max(Q_measurement)) / np.max(Q_measurement)
    quantity_err = (np.sum(Q) - np.sum(Q_measurement)) / np.sum(Q_measurement)
    NSE = 1 - np.sum((Q - Q_measurement)**2) / np.sum((Q_measurement - np.mean(Q_measurement))**2)

    print(f"洪峰误差 = {peak_err:.4f}")
    print(f"水量误差 = {quantity_err:.4f}")
    print(f"NSE = {NSE:.4f}")

    # 将计算结果写入第四列
    df[4] = Q
    if output_path:
        df.to_excel(output_path, index=False)
        print(f"结果已保存到 {output_path}")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(Q_measurement, label='OBS')
    plt.plot(Q, label='SIM')
    plt.xlabel('day')
    plt.ylabel('runoff')
    plt.title('OBS VS SIM')
    plt.legend()
    plt.show()

    return Q, peak_err, quantity_err, NSE

if __name__ == "__main__":
    # 运行模型，替换为你的数据路径和结果保存路径
    input_file = r"D:\Desktop\data.xlsx"
    output_file = r"D:\Desktop\data_with_model_results.xlsx"
    run_model(input_file, output_file)


