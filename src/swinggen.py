import matplotlib.pyplot as plt
import numpy as np


def gen_splines(point_params: list, verbose=False) -> list:
    if len(point_params) < 2:
        raise AttributeError("Not enough points provided")
    if verbose:
        print(f"Generating splines for points {point_params}")

    splines = []
    for i, _ in enumerate(point_params[:-1]):
        # print(f"Loop i: {i}")
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
        koeffs = np.linalg.solve(A, b)
        if verbose:
            print(f"Solving the following System A*x=b : \nA: {A}\nb: {b}")
            print(f"Koeffs: {koeffs}")
        spline = lambda t, koeffs=koeffs: koeffs[0] * t ** 5 + koeffs[1] * t ** 4 + koeffs[2] * t ** 3 + koeffs[
            3] * t ** 2 + koeffs[
                                              4] * t + koeffs[5]
        splines.append(spline)

    return splines


def gen_splines_xy(point_param_lists: list) -> list:
    points_x = [p[0] for p in point_param_lists]
    points_y = [p[1] for p in point_param_lists]

    return [gen_splines(points_x), gen_splines(points_y)]


def plot_splines(point_param_lists: list, splines_xy_list=None, step=0.01):
    splines_xy = splines_xy_list
    if not splines_xy_list:
        splines_xy = gen_splines_xy(point_param_lists)

    t = [np.arange(point_param_lists[i][0]["t"], point_param_lists[i + 1][0]["t"] + step, step) for i in
         range(len(point_param_lists) - 1)]
    x_pts = [list(map(s, t)) for s, t in zip(splines_xy[0], t)]
    y_pts = [list(map(s, t)) for s, t in zip(splines_xy[1], t)]


    fig, axs = plt.subplots(1, 3)

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    # splines x
    for ts, s in zip(t, x_pts):
        axs[0].plot(ts, s)
    # points x
    for point in point_param_lists:
        axs[0].plot(point[0]["t"], point[0]["pos"], "xk")

    axs[1].set_xlabel("t")
    axs[1].set_ylabel("y")
    # splines x
    for ts, s in zip(t, y_pts):
        axs[1].plot(ts, s)
    # points x
    for point in point_param_lists:
        axs[1].plot(point[1]["t"], point[1]["pos"], "xk")

    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    for xs, ys in zip(x_pts, y_pts):
        axs[2].plot(xs, ys)
    for point in point_param_lists:
        axs[2].plot(point[0]["pos"], point[1]["pos"], "xk")

    plt.show()


if __name__ == "__main__":
    P0 = [{"t": 0, "pos": 0, "vel": -5, "acc": 0.25}, {"t": 0, "pos": 0, "vel": 0.5, "acc": 0.1}]
    P1 = [{"t": 0.5, "pos": 1, "vel": 20, "acc": 0}, {"t": 0.5, "pos": 1, "vel": 0, "acc": 0}]
    P2 = [{"t": 1.0, "pos": 2, "vel": -5, "acc": -0.25}, {"t": 1, "pos": 0, "vel": -0.25, "acc": 0.1}]

    plot_splines([P0, P1, P2])

    # splines_xy = gen_splines_xy([P0, P1, P2])
    #
    # step = 0.01
    # t0 = np.arange(0, 0.5 + step, step)
    # t1 = np.arange(0.5, 1 + step, step)
    #
    # x0_pts = list(map(splines_xy[0][0], t0))
    # y0_pts = list(map(splines_xy[1][0], t0))
    #
    # x1_pts = list(map(splines_xy[0][1], t1))
    # y1_pts = list(map(splines_xy[1][1], t1))
    #
    # fig, axs = plt.subplots(1, 3)
    #
    # axs[0].set_xlabel("t")
    # axs[0].set_ylabel("x")
    # axs[0].plot(t0, x0_pts, "r")
    # axs[0].plot(t1, x1_pts, "b")
    # axs[0].plot(P0[0]["t"], P0[0]["pos"], "xk", P1[0]["t"], P1[0]["pos"], "xk", P2[0]["t"], P2[0]["pos"], "xk")
    #
    # axs[1].set_xlabel("t")
    # axs[1].set_ylabel("y")
    # axs[1].plot(t0, y0_pts, "r")
    # axs[1].plot(t1, y1_pts, "b")
    # axs[1].plot(P0[1]["t"], P0[1]["pos"], "xk", P1[1]["t"], P1[1]["pos"], "xk", P2[1]["t"], P2[1]["pos"], "xk")
    #
    # axs[2].set_xlabel("x")
    # axs[2].set_ylabel("y")
    #
    # axs[2].plot(x0_pts, y0_pts, "r")
    # axs[2].plot(x1_pts, y1_pts, "b")
    # axs[2].plot(P0[0]["pos"], P0[1]["pos"], "xk")
    # axs[2].plot(P1[0]["pos"], P1[1]["pos"], "xk")
    # axs[2].plot(P2[0]["pos"], P2[1]["pos"], "xk")
    # plt.show()
