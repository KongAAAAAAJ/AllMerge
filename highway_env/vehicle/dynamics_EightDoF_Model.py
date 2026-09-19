from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json


def dynamic_check(
        t,  # 时间序列 单位：s
        delta_f,  # 前轮转角序列 单位：rad
        vx,  # 纵向车速序列 单位：m/s
        ax,  # 纵向加速度序列 单位：m/s^2
):

    # =====================整车动力学校核==============
    # 动力学计算步长 (s) 步长越短计算越准确，计算速度越慢 +-+- 0.01 +-+-
    sample_time = 0.01

    # 初值
    d_vy1 = 0
    d_d_phi1 = 0
    v0 = vx[0]
    Rw = 0.51 / 1.01
    s0 = 0.01
    omega0 = v0 / ((1 - s0) * Rw)
    x = np.array([0, 0, 0, 0, omega0, omega0, omega0, omega0])
    u = np.zeros(3)

    # +-+- 防止车速为0时的动力学计算错误 +-+-
    # vx = np.where(vx == 0, 1e-4, x)
    vx = np.where(vx <= 0.005, 0.005, vx)

    LTR_values = []
    Yaw_rate = []
    Roll_rate = []
    Roll = []

    # ------------ 迭代计算整车动力学模型 ----------------
    for i in range(len(t)):
        # 插值计算输入
        u[0] = delta_f[i]
        u[1] = ax[i]
        u[2] = vx[i]

        # 计算微分
        dx, LTR = Devs(x, u, d_vy1, d_d_phi1)  # dx = [d_vy, d_r, d_d_phi, d_phi, d_Wfl, d_Wfr, d_Wrl, d_Wrr]
        LTR_values.append(LTR)

        # 更新中间量
        d_vy1 = dx[0]
        d_d_phi1 = dx[2]

        # 更新状态
        x = x + dx * sample_time  # x = [vy, r, d_phi, phi, Wfl, Wfr, Wrl, Wrr]

        # 计算输出
        beta = np.arctan(x[0] / u[2])
        Yaw_rate.append(x[1])
        Roll_rate.append(x[2])
        Roll.append(x[3])

    # debug: plot
    # plt.figure()
    # plt.plot(LTR_values)
    # plt.title('LTR')
    #
    # plt.figure()
    # plt.plot(Yaw_rate)
    # plt.title('Yaw Rate')
    #
    # plt.figure()
    # plt.plot(Roll_rate)
    # plt.title('Roll Rate')
    #
    # plt.figure()
    # plt.plot(Roll)
    # plt.title('Yaw Angle')
    #
    # plt.show()

    # 判断
    LTR_max = 0.85
    if all(x < LTR_max for x in LTR_values):
        return True, LTR_values
    else:
        return False, LTR_values


def Devs(x, u, d_vy1, d_d_phi1):
    # ==============整车参数================

    # B级车参数
    with open(Path(__file__).resolve().parent / "parameters" / "truck.json", "r", encoding="utf-8") as file:
        parameter = json.load(file)
    m = parameter["m"]  # 整车质量
    mu = parameter["mu"]  # 非簧载质量
    ms = parameter["ms"]  # 簧载质量

    Iz = parameter["Iz"]  # 车辆横摆转动惯量
    Ix = parameter["Ix"]  # 车辆侧倾转动惯量
    Ixz = parameter["Ixz"]  # 车辆绕xz轴的转动惯量积
    lf = parameter["lf"]  # 车辆质心到前轴的距离
    L = parameter["L"]  # 轴距
    lr = parameter["lr"]  # 车辆质心到后轴的距离
    B = parameter["B"]  # 轮距
    hcg = parameter["hcg"]  # 车辆质心离地高度
    h = parameter["h"]  # 簧载质量质心到侧倾轴垂直距离

    kf = parameter["kf"]  # 前轮侧偏刚度 N/rad
    kr = parameter["kr"]  # 后轮侧偏刚度
    Cf = parameter["Cf"]  # 前轮纵向刚度 N
    Cr = parameter["Cr"]  # 后轮纵向刚度

    kphif = parameter["kphif"]  # 车辆前悬架侧倾角刚度
    kphir = parameter["kphir"]  # 车辆前后架侧倾角刚度
    bphif = parameter["bphif"]  # 车辆前悬架侧倾角阻尼
    bphir = parameter["bphir"]  # 车辆前后架侧倾角阻尼

    f = parameter["f"]  # 车轮滚动阻力系数

    CD = parameter["CD"]  # 空气阻力系数
    ru = parameter["ru"]  # 空气密度
    A = parameter["A"]  # 汽车正面迎风面积

    Iw = parameter["Iw"]  # 车轮转动惯量
    Rw = parameter["Rw"]  # 车轮滚动半径

    lfs = parameter["lfs"]  # 簧载质量质心到前轴距离
    lrs = parameter["lrs"]  # 簧载质量质心到后轴距离

    hrf = parameter["hrf"]  # 前轴侧倾中心离地距离
    hrr = parameter["hrr"]  # 后轴侧倾中心离地距离
    muf = parameter["muf"]  # 非簧载质量在前轴分配值
    mur = parameter["mur"]  # 非簧载质量在后轴分配值

    huf = parameter["huf"]  # 前轴非簧载质量中心离地高度
    hur = parameter["hur"]  # 后轴非簧载质量中心离地高度

    g = parameter["g"]  # 重力加速度

    # ==============状态给定==================
    vy = x[0]
    r = x[1]
    d_phi = x[2]  # 侧倾角速度
    phi = x[3]  # 侧倾角
    Wfl = x[4]
    Wfr = x[5]
    Wrl = x[6]
    Wrr = x[7]

    deltaf_out = u[0]
    ax_out = u[1]
    vx_out = u[2]
    vx = vx_out

    # 不变参数按照常数定义，减少无用输入
    deltarl = 0
    deltarr = 0
    Tbfl = 0
    Tbfr = 0
    Tbrl = 0
    Tbrr = 0
    miu = 0.85  # 路面附着系数

    # 转角换算
    deltafl = np.arctan(np.tan(deltaf_out) / (1 - B / L / 2 * np.tan(deltaf_out)))
    deltafr = np.arctan(np.tan(deltaf_out) / (1 + B / L / 2 * np.tan(deltaf_out)))

    # 转矩换算
    Fx = m * ax_out + f * m * g + CD * A * ru * vx_out ** 2 / 2
    Tall = Fx * Rw
    Delta_T = Tall / 2 * (1 - np.cos(deltaf_out))
    Tdfl = (Tall + Delta_T) / 4
    Tdfr = Tdfl
    Tdrl = Tdfl
    Tdrr = Tdfl

    # ==============车轮纵向速度==================
    ufl = (vx - 0.5 * B * r) * np.cos(deltafl) + (vy + lf * r) * np.sin(deltafl)
    ufr = (vx + 0.5 * B * r) * np.cos(deltafr) + (vy + lf * r) * np.sin(deltafr)
    url = (vx - 0.5 * B * r) * np.cos(deltarl) + (vy - lr * r) * np.sin(deltarl)
    urr = (vx + 0.5 * B * r) * np.cos(deltarr) + (vy - lr * r) * np.sin(deltarr)
    # ==============车轮侧偏角==================
    alfafl = deltafl - np.arctan((vy + lf * r) / (vx - 0.5 * B * r))
    alfafr = deltafr - np.arctan((vy + lf * r) / (vx + 0.5 * B * r))
    alfarl = deltarl - np.arctan((vy - lr * r) / (vx - 0.5 * B * r))
    alfarr = deltarr - np.arctan((vy - lr * r) / (vx + 0.5 * B * r))
    # ==============车轮滑移率==================
    if ufl < Rw * Wfl:
        sfl = 1 - ufl / (Rw * Wfl)
    else:
        sfl = (Rw * Wfl) / ufl - 1

    if ufr < Rw * Wfr:
        sfr = 1 - ufr / (Rw * Wfr)
    else:
        sfr = (Rw * Wfr) / ufr - 1

    if url < Rw * Wrl:
        srl = 1 - url / (Rw * Wrl)
    else:
        srl = (Rw * Wrl) / url - 1

    if urr < Rw * Wrr:
        srr = 1 - urr / (Rw * Wrr)
    else:
        srr = (Rw * Wrr) / urr - 1
    # ==============中间变量==================
    ax = ax_out - vy * r
    ay = d_vy1 + vx * r
    # ==============车轮垂直载荷==================
    Fzfl = m * g * lr / 2 / L - m * ax * hcg / 2 / L - ay / B * (ms * hrf * lrs / L + muf * huf) - 1 / B * (
            kphif * phi + bphif * d_phi)
    Fzfr = m * g * lr / 2 / L - m * ax * hcg / 2 / L + ay / B * (ms * hrf * lrs / L + muf * huf) + 1 / B * (
            kphif * phi + bphif * d_phi)
    Fzrl = m * g * lf / 2 / L + m * ax * hcg / 2 / L - ay / B * (ms * hrr * lfs / L + mur * hur) - 1 / B * (
            kphir * phi + bphir * d_phi)
    Fzrr = m * g * lf / 2 / L + m * ax * hcg / 2 / L + ay / B * (ms * hrr * lfs / L + mur * hur) + 1 / B * (
            kphir * phi + bphir * d_phi)
    # ==============车轮纵向力侧向力(Dugoff轮胎模型)==================
    lambdafl = miu * Fzfl * (1 - sfl) / 2 / np.sqrt(Cf ** 2 * sfl ** 2 + kf ** 2 * np.tan(alfafl) ** 2)
    if lambdafl <= 1:
        flambdafl = (2 - lambdafl) * lambdafl
    else:
        flambdafl = 1
    Fxfl = Cf * sfl * flambdafl / (1 - sfl)
    Fyfl = kf * np.tan(alfafl) * flambdafl / (1 - sfl)

    lambdafr = miu * Fzfr * (1 - sfr) / 2 / np.sqrt(Cf ** 2 * sfr ** 2 + kf ** 2 * np.tan(alfafr) ** 2)
    if lambdafr <= 1:
        flambdafr = (2 - lambdafr) * lambdafr
    else:
        flambdafr = 1
    Fxfr = Cf * sfr * flambdafr / (1 - sfr)
    Fyfr = kf * np.tan(alfafr) * flambdafr / (1 - sfr)

    lambdarl = miu * Fzrl * (1 - srl) / 2 / np.sqrt(Cr ** 2 * srl ** 2 + kr ** 2 * np.tan(alfarl) ** 2)
    if lambdarl <= 1:
        flambdarl = (2 - lambdarl) * lambdarl
    else:
        flambdarl = 1
    Fxrl = Cr * srl * flambdarl / (1 - srl)
    Fyrl = kr * np.tan(alfarl) * flambdarl / (1 - srl)

    lambdarr = miu * Fzrr * (1 - srr) / 2 / np.sqrt(Cr ** 2 * srr ** 2 + kr ** 2 * np.tan(alfarr) ** 2)
    if lambdarr <= 1:
        flambdarr = (2 - lambdarr) * lambdarr
    else:
        flambdarr = 1
    Fxrr = Cr * srr * flambdarr / (1 - srr)
    Fyrr = kr * np.tan(alfarr) * flambdarr / (1 - srr)

    # ==============车辆八自由度运动微分方程==================
    d_vy = ms / m * h * d_d_phi1 - vx * r + 1 / m * (
            Fxfl * np.sin(deltafl) + Fxfr * np.sin(deltafr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(
        deltarr) + Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fyrl * np.cos(deltarl) + Fyrr * np.cos(
        deltarr))
    d_r = 1 / Iz * Ixz * d_d_phi1 + 1 / Iz * (
            lf * (Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fxfl * np.sin(deltafl) + Fxfr * np.sin(
        deltafr)) - lr * (Fyrl * np.cos(deltarl) + Fyrr * np.cos(deltarr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(
        deltarr)) + B / 2 * (Fyfl * np.sin(deltafl) + Fyrl * np.sin(deltarl) - Fyfr * np.sin(deltafr) - Fyrr * np.sin(
        deltarr)) + B / 2 * (-Fxfl * np.cos(deltafl) + Fxfr * np.cos(deltafr) - Fxrl * np.cos(deltarl) + Fxrr * np.cos(
        deltarr)))
    d_d_phi = 1 / Ix * (
            ms * g * h * phi - (bphif + bphir) * d_phi - (kphif + kphir) * phi + ms * h * (
            d_vy1 + vx * r) + Ixz * d_r)
    d_Wfl = 1 / Iw * (Tdfl - Fxfl * Rw - Tbfl)
    d_Wfr = 1 / Iw * (Tdfr - Fxfr * Rw - Tbfr)
    d_Wrl = 1 / Iw * (Tdrl - Fxrl * Rw - Tbrl)
    d_Wrr = 1 / Iw * (Tdrr - Fxrr * Rw - Tbrr)

    # ==============更新状态==================
    dx = np.array([d_vy, d_r, d_d_phi, d_phi, d_Wfl, d_Wfr, d_Wrl, d_Wrr])

    # 侧倾指标
    LTR = abs(((Fzfl + Fzrl) - (Fzfr + Fzrr)) / ((Fzfl + Fzrl) + (Fzfr + Fzrr)))

    return dx, LTR
