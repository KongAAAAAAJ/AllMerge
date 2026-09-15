import copy
import importlib
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import random
import os, csv, datetime
import json
import seaborn as sns

from matplotlib.pyplot import contour
from numpy.ma.extras import average
from sphinxcontrib.bibtex import visit_raw_latex
from sympy import false
from torch.autograd import is_view_replay_enabled

# Useful types
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


# Kong add
def lmap_plus(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (x[1] - v) * (y[1] - y[0]) / (x[1] - x[0]) if np.logical_and(v >= x[0], v <= x[-1]) else 0


def get_class_path(cls: Callable) -> str:
    return cls.__module__ + "." + cls.__qualname__


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return (
        rect_min[0] <= point[0] <= rect_max[0]
        and rect_min[1] <= point[1] <= rect_max[1]
    )


def point_in_rotated_rectangle(
    point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float
) -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def point_in_ellipse(
    point: Vector, center: Vector, angle: float, length: float, width: float
) -> bool:
    """
    Check if a point is inside an ellipse

    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    :return: is the point inside the ellipse
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(
    rect1: Tuple[Vector, float, float, float], rect2: Tuple[Vector, float, float, float]
) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def rect_corners(
    center: np.ndarray,
    length: float,
    width: float,
    angle: float,
    include_midpoints: bool = False,
    include_center: bool = False,
) -> List[np.ndarray]:
    """
    Returns the positions of the corners of a rectangle.
    :param center: the rectangle center
    :param length: the rectangle length
    :param width: the rectangle width
    :param angle: the rectangle angle
    :param include_midpoints: include middle of edges
    :param include_center: include the center of the rect
    :return: a list of positions
    """
    center = np.array(center)
    half_l = np.array([length / 2, 0])
    half_w = np.array([0, width / 2])
    corners = [-half_l - half_w, -half_l + half_w, +half_l + half_w, +half_l - half_w]
    if include_center:
        corners += [[0, 0]]
    if include_midpoints:
        corners += [-half_l, half_l, -half_w, half_w]

    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ np.array(corners).T).T + np.tile(center, (len(corners), 1))


def has_corner_inside(
    rect1: Tuple[Vector, float, float, float], rect2: Tuple[Vector, float, float, float]
) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return any(
        [
            point_in_rotated_rectangle(p1, *rect2)
            for p1 in rect_corners(*rect1, include_midpoints=True, include_center=True)
        ]
    )


def project_polygon(polygon: Vector, axis: Vector) -> Tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """
    Calculate the distance between [minA, maxA] and [minB, maxB]
    The distance will be negative if the intervals overlap
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(
    a: Vector, b: Vector, displacement_a: Vector, displacement_b: Vector
) -> Tuple[bool, bool, Optional[np.ndarray]]:
    """
    Checks if the two polygons are intersecting.

    See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: polygon A, as a list of [x, y] points
    :param b: polygon B, as a list of [x, y] points
    :param displacement_a: velocity of the polygon A
    :param displacement_b: velocity of the polygon B
    :return: are intersecting, will intersect, translation vector
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def confidence_ellipsoid(
    data: Dict[str, np.ndarray],
    lambda_: float = 1e-5,
    delta: float = 0.1,
    sigma: float = 0.1,
    param_bound: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a confidence ellipsoid over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 regularization parameter
    :param delta: confidence level
    :param sigma: noise covariance
    :param param_bound: an upper-bound on the parameter norm
    :return: estimated theta, Gramian matrix G_N_lambda, radius beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1 / sigma * np.transpose(phi) @ phi + lambda_ * np.identity(
        phi.shape[-1]
    )
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = (
        np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_**d) / delta))
        + np.sqrt(lambda_ * d) * param_bound
    )
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(
    data: dict, parameter_box: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a confidence polytope over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: estimated theta, polytope vertices, Gramian matrix G_N_lambda, radius beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(
        data, param_bound=param_bound
    )

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(
            d_theta[k],
            parameter_box[0] - theta_n_lambda,
            parameter_box[1] - theta_n_lambda,
        )
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    gramian: np.ndarray,
    beta: float,
    sigma: float = 0.1,
) -> bool:
    """
    Check if a new observation (phi, y) is valid according to a confidence ellipsoid on theta.

    :param y: observation
    :param phi: feature
    :param theta: estimated parameter
    :param gramian: Gramian matrix
    :param beta: ellipsoid radius
    :param sigma: noise covariance
    :return: validity of the observation
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    Check whether a dataset {phi_n, y_n} is consistent

    The last observation should be in the confidence ellipsoid obtained by the N-1 first observations.

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: consistency of the dataset
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(
            train_set, parameter_box=parameter_box
        )
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True


def near_split(x, num_bins=None, size_bins=None):
    """
    Split a number into several bins with near-even distribution.

    You can either set the number of bins, or their size.
    The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def distance_to_circle(center, radius, direction):
    scaling = radius * np.ones((2, 1))
    a = np.linalg.norm(direction / scaling) ** 2
    b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
    c = np.linalg.norm(center / scaling) ** 2 - 1
    root_inf, root_sup = solve_trinom(a, b, c)
    if root_inf and root_inf > 0:
        distance = root_inf
    elif root_sup and root_sup > 0:
        distance = 0
    else:
        distance = np.infty
    return distance


def distance_to_rect(line: Tuple[np.ndarray, np.ndarray], rect: List[np.ndarray]):
    """
    Compute the intersection between a line segment and a rectangle.

    See https://math.stackexchange.com/a/2788041.
    :param line: a line segment [R, Q]
    :param rect: a rectangle [A, B, C, D]
    :return: the distance between R and the intersection of the segment RQ with the rectangle ABCD
    """
    r, q = line
    a, b, c, d = rect
    u = b - a
    v = d - a
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    rqu = (q - r) @ u
    rqv = (q - r) @ v
    interval_1 = [(a - r) @ u / rqu, (b - r) @ u / rqu]
    interval_2 = [(a - r) @ v / rqv, (d - r) @ v / rqv]
    interval_1 = interval_1 if rqu >= 0 else list(reversed(interval_1))
    interval_2 = interval_2 if rqv >= 0 else list(reversed(interval_2))
    if (
        interval_distance(*interval_1, *interval_2) <= 0
        and interval_distance(0, 1, *interval_1) <= 0
        and interval_distance(0, 1, *interval_2) <= 0
    ):
        return max(interval_1[0], interval_2[0]) * np.linalg.norm(q - r)
    else:
        return np.inf


def solve_trinom(a, b, c):
    delta = b**2 - 4 * a * c
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
    else:
        return None, None


# Kong add
def list_flatten(nested_list) -> list:
    """
    Form the nested list to flatten list
    param: nested_list
    return flatten_list
    """
    return np.array(nested_list).flatten().tolist()


def find_keys_by_value(d, value):
    return [k for k, v in d.items() if v == value]


def inside_list(v, l, reverse: bool = False) -> bool:
    i = 0
    j = 1
    for _ in l[1:]:
        if not reverse:
            if l[i] <= v <= l[j]:
                return True
        else:
            if l[j] <= v <= l[i]:
                return True
        i += 1
        j += 1
    return False


def diff(l) -> list:
    return [a - b for a, b in zip(l, l[1:])]


def ttc(front_vehicle, rear_vehicle) -> float:
    if front_vehicle.speed >= rear_vehicle.speed:
        ttc = float(np.inf)
    else:
        ttc = (front_vehicle.position[0] - rear_vehicle.position[0]) / (rear_vehicle.speed - front_vehicle.speed)
    return ttc

def ttc_y(vehicle1, vehicle2) -> float:
    v1_y = vehicle1.velocity[1]
    v2_y = vehicle2.velocity[1]
    y1 = vehicle1.position[1]
    y2 = vehicle1.position[1]
    if y1 >= y2:
        ttc_y = (y1 - y2) / (v2_y - v1_y) if v2_y > v1_y else np.inf
    else:
        ttc_y = (y2 - y1) / (v1_y - v2_y) if v1_y > v2_y else np.inf
    return ttc_y



def normalization(limit, value) -> float:
    return (value - limit[0]) / (limit[1] - limit[0])

"""在单次测试后，绘制x, y, v, a曲线"""
def results(controlled_vehicles):
    colors = ['#9B59B6', '#3498DB', '#1ABC9C']
    labels = ['Car-1', 'Car-2', 'Car-3']
    t_end = 30
    # 字体设置
    axis_font = FontProperties(family='Arial', size=12, style='normal', weight='normal')
    title_font = FontProperties(family='Arial', size=14, style='normal', weight='normal')

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    fig = plt.figure()
    ax = fig.add_subplot()
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["x"]))
        x = v.info["x"]
        ax.plot(t, x, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title("Longitudinal Position", font_properties=title_font)
        ax.set_xlabel("$t$ ($s$)", font_properties=axis_font)
        ax.set_ylabel("$x$ ($m$)", font_properties=axis_font)
        ax.set_xlim(0, 30)

    fig = plt.figure()
    ax = fig.add_subplot()
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["y"]))
        y = v.info["y"]
        ax.plot(t, y, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title("Lateral Position", font_properties=title_font)
        ax.set_xlabel(r"$t$ ($s$)", font_properties=axis_font)
        ax.set_ylabel(r"$y$ ($m$)", font_properties=axis_font)
        ax.set_xlim(0, 30)
        ax.set_ylim(-2, 10)

    fig = plt.figure()
    ax = fig.add_subplot()
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["v"]))
        speed = v.info["v"]
        ax.plot(t, speed, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title("Speed", font_properties=title_font)
        ax.set_xlabel(r"$t$ ($s$)", font_properties=axis_font)
        ax.set_ylabel(r"$v$ ($m/s$)", font_properties=axis_font)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 35)

    fig = plt.figure()
    ax = fig.add_subplot()
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["a"]))
        a = v.info["a"]
        ax.plot(t, a, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title("Acceleration", font_properties=title_font)
        ax.set_xlabel(r"$t$ ($s$)", font_properties=axis_font)
        ax.set_ylabel(r"$a$ ($m/s^2$)", font_properties=axis_font)
        ax.set_xlim(0, 30)
        ax.set_ylim(-8, 6)

    fig = plt.figure()
    ax = fig.add_subplot()
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["delta"]))
        delta = v.info["delta"]
        ax.plot(t, delta, color=colors[i], label=labels[i], linewidth=2)
        ax.set_title("Steering Angle", font_properties=title_font)
        ax.set_xlabel(r"$t$ ($s$)", font_properties=axis_font)
        ax.set_ylabel(r"$delta$ ($rad/s$)", font_properties=axis_font)
        ax.set_xlim(0, 30)
        ax.set_ylim(-np.pi / 6, np.pi / 6)

    # x-y-t三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        t = np.linspace(0, t_end, len(v.info["x"]))
        x = v.info["x"]
        y = v.info["y"]
        z = list(t)

        ax.plot(
            y, x, z, color=colors[i], linewidth=2, label=labels[i], marker='.', markersize=5
        )
        # 垂线
        for j in range(len(t)):
            ax.plot([y[j], y[j]], [x[j], x[j]], [0, z[j]], color=colors[i], linestyle='--', linewidth=0.3, alpha=0.5)

        # 标题和标签
        ax.set_title('3D Trajectory of Platoon', font_properties=title_font)
        ax.set_ylabel('$x$ ($m$)', font_properties=axis_font)
        ax.set_xlabel('$y$ ($m$)', font_properties=axis_font)
        ax.set_zlabel('$t$ ($s$)', font_properties=axis_font)
        # 调整x, y, z轴标签的位置 (通过labelpad调整标签与轴的距离)
        ax.xaxis.labelpad = 5  # 增加x轴标签与x轴的距离
        ax.yaxis.labelpad = 10  # 增加y轴标签与y轴的距离
        ax.zaxis.labelpad = 5  # 增加z轴标签与z轴的距离

        ax.set_box_aspect([4, 10, 3])  # 显示比例

        ax.view_init(elev=30, azim=-15)   # 调整视角
        # ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_facecolor('whitesmoke')
        ax.legend()
    plt.tight_layout()
    plt.show()

def result_log(controlled_vehicles, average_speed, vehicle_infos, crashed) -> tuple:
    for v, i in zip(controlled_vehicles, range(len(controlled_vehicles))):
        if len(v.info["v"]) > 0:
            average_speed[i].append(sum(v.info["v"]) / len(v.info["v"]))
    if 1 in vehicle_infos["crashed"]:
        crashed.append(1)
    else:
        crashed.append(0)
    return average_speed, crashed

def results_show(average_speed, crashed, ave_follow_distance, ave_follow_ttc):
    with open('average_speed.json', 'w') as file:
        json.dump(average_speed, file)
    with open('crashed_rate.json', 'w') as file:
        json.dump(crashed, file)
    with open('average_follow_distance.json', 'w') as file:
        json.dump(ave_follow_distance, file)
    with open('average_follow_ttc.json', 'w') as file:
        json.dump(ave_follow_ttc, file)
    # # 画箱形图
    # # 创建图形和轴
    # plt.figure()
    # # 使用Seaborn画箱形图
    # sns.set_palette('colorblind')
    # sns.boxplot(data=average_speed, palette="Set3")
    # # 美化图形
    # plt.title('Distribution of Average speed', fontsize=14)
    # plt.xlabel('Car', fontsize=12)
    # plt.ylabel(r'$v_ave$ ($m$)', fontsize=12)
    # plt.xticks([0, 1, 2], ['Car-1', 'Car-2', 'Car-3'], fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # # 添加背景
    # sns.set(style="whitegrid")
    # # 显示图形
    # plt.show()


def generate_random_numbers(start, end, count, interval, deviation):
    """
    生成一组按顺序排列的随机数

    :param start: 起始值
    :param end: 结束值
    :param count: 生成随机数的数量
    :param interval: 间隔
    :param deviation: 每个间隔的随机偏差范围
    :return: 随机数列表
    """
    numbers = []
    current = start
    step = (end - start) / (count - 1)

    for _ in range(count):
        # 对每个间隔添加一个小的随机偏差
        numbers.append(current + random.uniform(-deviation, deviation))
        current += step

    return numbers

def vehicle_info_save(vehicle_infos, info, done, file_dir, filename) -> dict:
    # 写入标题行
    headers = ["ave_speed-car_1", "ave_speed-car_2", "ave_speed-car_3", "crash_rate",
               "average_follow_distance_1", "average_follow_distance_2", "average_follow_distance_3",
               "average_follow_ttc_1", "average_follow_ttc_2", "average_follow_ttc_3"]
    csv_path = os.path.join(file_dir, f'{filename}.csv')
    if os.path.exists(csv_path) == 0:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([header for i, header in enumerate(headers) if i in range(0, len(headers))])

    if not done:
        vehicle_infos["speed_car_1"].append(info[0]["speed"][0])
        vehicle_infos["speed_car_2"].append(info[0]["speed"][1])
        vehicle_infos["speed_car_3"].append(info[0]["speed"][2])
        vehicle_infos["crashed"].append(int(info[0]["crashed"]))
        vehicle_infos["follow_distance_1"].append(info[0]["follow_distance"][0])
        vehicle_infos["follow_distance_2"].append(info[0]["follow_distance"][1])
        vehicle_infos["follow_distance_3"].append(info[0]["follow_distance"][2])
        vehicle_infos["follow_ttc_1"].append(info[0]["follow_ttc"][0])
        vehicle_infos["follow_ttc_2"].append(info[0]["follow_ttc"][1])
        vehicle_infos["follow_ttc_3"].append(info[0]["follow_ttc"][2])
    else:
        ave_speed_1 = np.mean(vehicle_infos["speed_car_1"])
        ave_speed_2 = np.mean(vehicle_infos["speed_car_2"])
        ave_speed_3 = np.mean(vehicle_infos["speed_car_3"])
        un_crashed_rate = vehicle_infos["crashed"].count(0) / len(vehicle_infos["crashed"])
        ave_follow_distance_1 = np.mean(vehicle_infos["follow_distance_1"])
        ave_follow_distance_2 = np.mean(vehicle_infos["follow_distance_2"])
        ave_follow_distance_3 = np.mean(vehicle_infos["follow_distance_3"])
        ave_follow_ttc_1 = np.mean(vehicle_infos["follow_ttc_1"])
        ave_follow_ttc_2 = np.mean(vehicle_infos["follow_ttc_2"])
        ave_follow_ttc_3 = np.mean(vehicle_infos["follow_ttc_3"])
        row = [ave_speed_1, ave_speed_2, ave_speed_3, un_crashed_rate,
               ave_follow_distance_1, ave_follow_distance_2, ave_follow_distance_3,
               ave_follow_ttc_1, ave_follow_ttc_2, ave_follow_ttc_3]
        # 写入数据
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([item for i, item in enumerate(row) if i in range(0, len(headers))])

        # update vehicle_infos
        vehicle_infos = {
            "speed_car_1": [],
            "speed_car_2": [],
            "speed_car_3": [],
            "crashed": [],
            "follow_distance_1": [],
            "follow_distance_2": [],
            "follow_distance_3": [],
            "follow_ttc_1": [],
            "follow_ttc_2": [],
            "follow_ttc_3": [],
        }

    return vehicle_infos

def data_record_each_step(env, action):
    MAX_DISTANCE = 100
    MAX_TTC = 30
    # save ttc, speeds, actions of a test
    for vehicle, k in zip(env.controlled_vehicles, range(len(env.controlled_vehicles))):
        front_v, _ = env.road.neighbour_vehicles(vehicle=vehicle, lane_index=vehicle.lane_index)
        if front_v is not None:
            dis = front_v.position[0] - vehicle.position[0]
            env.record_data["distances"][k].append(min(MAX_DISTANCE, dis))
            env.record_data["ttcs"][k].append(
                min(ttc(front_vehicle=front_v, rear_vehicle=vehicle), MAX_TTC)
            )
        else:
            env.record_data["distances"][k].append(MAX_DISTANCE)
            env.record_data["ttcs"][k].append(MAX_TTC)
        env.record_data["speeds"][k].append(vehicle.speed)
    env.record_data["actions"].append(action)

