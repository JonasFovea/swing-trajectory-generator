import time

import matplotlib.pyplot as plt
import numpy as np


class Spline:
    def __init__(self, coefficient_lists, domain_lists):
        self.coeff_lists = [clist[::-1] for clist in coefficient_lists]
        self.domains_lists = domain_lists
        # print(f"Generated spline with coeficcients (a0->an): {self.coeff_lists} over the domains: {self.domains_lists} ")

    def __call__(self, val):
        for dom, coeffs in zip(self.domains_lists, self.coeff_lists):
            if dom[0] <= val <= dom[1]:
                # print(f"Evaluating spline for {val} in domain between {dom[0]} and {dom[1]}")
                return self.__eval(val, coeffs)
        return None

    def get_range(self):
        return np.min(self.domains_lists), np.max(self.domains_lists)

    @staticmethod
    def __eval(val, coeffs):
        return sum([c * val ** i for i, c in enumerate(coeffs)])


# =================== END CLASSES     ===============================================
# =================== START FUNCTIONS ===============================================

def gen_spline_5(point_params: list, verbose=False) -> list:
    if len(point_params) < 2:
        raise AttributeError("Not enough points provided")
    if verbose:
        print(f"Generating splines for points {point_params}")

    # tstart = time.perf_counter()
    spline_coefficients = []
    spline_domains = []
    for i, _ in enumerate(point_params[:-1]):
        t = [point_params[i + j]["t"] for j in (0, 1)]

        pos = [point_params[i + j]["pos"] for j in (0, 1)]
        vel = [point_params[i + j]["vel"] for j in (0, 1)]
        acc = [point_params[i + j]["acc"] for j in (0, 1)]

        # print(f"Using values: pos: {pos}, vel: {vel}, acc: {acc}")

        A = np.array([
            [t[0] ** 5, t[0] ** 4, t[0] ** 3, t[0] ** 2, t[0], 1],
            [t[1] ** 5, t[1] ** 4, t[1] ** 3, t[1] ** 2, t[1], 1],
            [5 * t[0] ** 4, 4 * t[0] ** 3, 3 * t[0] ** 2, 2 * t[0], 1, 0],
            [5 * t[1] ** 4, 4 * t[1] ** 3, 3 * t[1] ** 2, 2 * t[1], 1, 0],
            [20 * t[0] ** 3, 12 * t[0] ** 2, 6 * t[0], 2, 0, 0],
            [20 * t[1] ** 3, 12 * t[1] ** 2, 6 * t[1], 2, 0, 0]])

        b = np.array([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]])
        coeffs = np.linalg.solve(A, b)
        if verbose:
            print(f"Solving the following System A*x=b : \nA: {A}\nb: {b}")
            print(f"Coefficients: {coeffs}")
        # spline = lambda t, koeffs=koeffs: koeffs[0] * t ** 5 + koeffs[1] * t ** 4 + koeffs[2] * t ** 3 + koeffs[
        #     3] * t ** 2 + koeffs[
        #                                       4] * t + koeffs[5]
        spline_coefficients.append(coeffs)
        spline_domains.append(t)
    # tend = time.perf_counter()
    # print(f"Time passed while calculating splines for one dimension: {tend-tstart}")
    return Spline(spline_coefficients, spline_domains)


def gen_spline_3(point_params: list, verbose=False) -> Spline:
    if len(point_params) < 2:
        raise AttributeError("Not enough points provided")
    if verbose:
        print(f"Generating spline_coefficients for points {point_params}")

    # tstart = time.perf_counter()
    spline_coefficients = []
    spline_domains = []
    for i, _ in enumerate(point_params[:-1]):
        t = [point_params[i + j]["t"] for j in (0, 1)]

        pos = [point_params[i + j]["pos"] for j in (0, 1)]
        vel = [point_params[i + j]["vel"] for j in (0, 1)]

        # print(f"Using values: pos: {pos}, vel: {vel}, acc: {acc}")

        A = np.array([
            [t[0] ** 3, t[0] ** 2, t[0], 1],
            [t[1] ** 3, t[1] ** 2, t[1], 1],
            [3 * t[0] ** 2, 2 * t[0], 1, 0],
            [3 * t[1] ** 2, 2 * t[1], 1, 0]])

        b = np.array([pos[0], pos[1], vel[0], vel[1]])
        coeffs = np.linalg.solve(A, b)
        if verbose:
            print(f"Solving the following System A*x=b : \nA: {A}\nb: {b}")
            print(f"Coefficients: {coeffs}")
        # spline = lambda t, koeffs=koeffs: koeffs[0] * t ** 3 + koeffs[1] * t ** 2 + koeffs[2] * t + koeffs[3]
        # spline_coefficients.append(spline)
        spline_coefficients.append(coeffs)
        spline_domains.append(t)

    # tend = time.perf_counter()
    # print(f"Time passed while calculating spline_coefficients for one dimension: {tend-tstart}")
    return Spline(spline_coefficients, spline_domains)


def gen_splines_xy(point_param_lists: list, spline_generator=gen_spline_3) -> list:
    # tstart = time.perf_counter_ns()
    points_x = [p[0] for p in point_param_lists]
    points_y = [p[1] for p in point_param_lists]
    # tend = time.perf_counter_ns()
    # print(f"Time Passed for calculationd 2D splines: {tend-tstart}ns")

    return [spline_generator(points_x), spline_generator(points_y)]


def plot_splines(point_param_lists: list, splines_xy_list=None, step=0.01, pyplot_axs=None):
    splines_xy = splines_xy_list
    if not splines_xy_list:
        splines_xy = gen_splines_xy(point_param_lists)

    t = [np.arange(point_param_lists[i][0]["t"], point_param_lists[i + 1][0]["t"] + step, step) for i in
         range(len(point_param_lists) - 1)]
    x_pts = [list(map(splines_xy[0], t_)) for t_ in t]
    y_pts = [list(map(splines_xy[1], t_)) for t_ in t]

    if pyplot_axs is None or pyplot_axs.shape != (1, 3):
        fig, axs = plt.subplots(1, 3)
    else:
        axs = pyplot_axs

    axs[0].set_aspect(aspect="equal", adjustable='datalim')

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[0].set_title("Splines in x-direction over time")
    # splines x
    for ts, s in zip(t, x_pts):
        axs[0].plot(ts, s)
    # points x
    for point in point_param_lists:
        axs[0].plot(point[0]["t"], point[0]["pos"], "xk")

    axs[1].set_aspect(aspect="equal", adjustable='datalim')

    axs[1].set_xlabel("t")
    axs[1].set_ylabel("y")
    axs[1].set_title("Splines in y-direction over time")
    # splines x
    for ts, s in zip(t, y_pts):
        axs[1].plot(ts, s)
    # points x
    for point in point_param_lists:
        axs[1].plot(point[1]["t"], point[1]["pos"], "xk")

    axs[2].set_aspect(aspect="equal", adjustable='datalim')

    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Combined Splines in xy-plane")
    for xs, ys in zip(x_pts, y_pts):
        axs[2].plot(xs, ys)
    for point in point_param_lists:
        axs[2].plot(point[0]["pos"], point[1]["pos"], "xk")

    plt.show()


def get_example_point_data() -> list:
    return [[{"t": 0, "pos": -1, "vel": -5, "acc": 10}, {"t": 0, "pos": 0, "vel": 0.1, "acc": 0.1}],
            [{"t": 0.5, "pos": 0, "vel": 20, "acc": 0}, {"t": 0.5, "pos": 1, "vel": 0, "acc": -2}],
            [{"t": 1.0, "pos": 1, "vel": -5, "acc": -0.25}, {"t": 1, "pos": 0, "vel": -0.25, "acc": 0.1}]]


if __name__ == "__main__":
    ex_points = get_example_point_data()
    plot_splines(ex_points)
    plot_splines(ex_points, gen_splines_xy(ex_points, spline_generator=gen_spline_5))

# TODO
# Make scaling on axes the same
