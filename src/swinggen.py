import time
import copy

import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk


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

    def __repr__(self):
        return f"Spline object over domain(s): {self.domains_lists} with coefficients: {self.coeff_lists}"

    def get_range(self):
        return np.min(self.domains_lists), np.max(self.domains_lists)

    @staticmethod
    def __eval(val, coeffs):
        return sum([c * val ** i for i, c in enumerate(coeffs)])


class AdjusterGui:
    def __init__(self, point_param_lists: list = None):

        # Main Window configuration
        self.root = tk.Tk()
        self.root.configure(bg="white")
        self.root.title("Swinggen Adjuster")
        self.root.geometry("450x700")
        self.root.resizable(width=False, height=False)

        self.__max_pos = 5
        self.__min_pos = -self.__max_pos
        self.__pos_step_size = 0.2

        self.__max_vel = 100
        self.__min_vel = -self.__max_vel
        self.__vel_step_size = 0.2

        self.__max_acc = 100
        self.__min_acc = -self.__max_acc
        self.__acc_step_size = 0.2

        self.show_collision = False
        self.__max_collision_offset = 2
        self.__min_collision_offset = 0
        self.__collision_step_size = 0.2
        self.__collision_offset = 1

        self.plot_step_size = 0.01

        self.points = get_example_point_data()
        self.collision_points = copy.deepcopy(self.points)
        self.collision_points[1][1]["pos"] += 1

        self.generator_func = gen_spline_5

        plt.ion()
        self.fig, self.axs = plt.subplots(1, 3)
        self.plot()
        # plot_splines(self.points, gen_splines_xy(self.points, spline_generator=self.generator_func),
        #              pyplot_axs=self.axs,
        #              update=True, step=self.plot_step_size)
        # self.fig.show()

        self.__init_widgets()

        # Start Tkinter mainloop
        self.root.mainloop()

    def __init_widgets(self):
        # ==== Content configuration ====

        self.order_bool = tk.BooleanVar()
        self.order_sel_check = tk.Checkbutton(self.root, text="5th order\nsplines", bg="white",
                                              variable=self.order_bool, command=self.update_order, bd=0,
                                              highlightthickness=0)
        self.order_sel_check.grid(column=0, row=1)
        self.order_bool.set(True)

        point_off = 0 * 5
        self.p1_title = tk.Label(self.root, text="Adjustments for Point 1", bg="orange")
        self.p1_title.grid(columnspan=3, column=1, row=point_off + 0, padx=5, pady=5)
        self.p1_x_dir = tk.Label(self.root, text="x-Direction", bg="white")
        self.p1_x_dir.grid(column=1, row=point_off + 1, padx=5, pady=5)
        self.p1_y_dir = tk.Label(self.root, text="y-Direction", bg="white")
        self.p1_y_dir.grid(column=2, row=point_off + 1, padx=5, pady=5)

        self.p1_pos_label = tk.Label(self.root, text="Position", bg="white")
        self.p1_pos_label.grid(column=0, row=point_off + 2, sticky=tk.SW, padx=5, pady=5)
        self.p1_pos_x_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_pos_x_scale.set(self.points[0][0]["pos"])
        self.p1_pos_x_scale.grid(column=1, row=point_off + 2, padx=5, pady=5)
        self.p1_pos_y_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_pos_y_scale.set(self.points[0][1]["pos"])
        self.p1_pos_y_scale.grid(column=2, row=point_off + 2, padx=5, pady=5)

        self.p1_vel_label = tk.Label(self.root, text="Velocity", bg="white")
        self.p1_vel_label.grid(column=0, row=point_off + 3, sticky=tk.SW, padx=5, pady=5)
        self.p1_vel_x_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_vel_x_scale.set(self.points[0][0]["vel"])
        self.p1_vel_x_scale.grid(column=1, row=point_off + 3, padx=5, pady=5)
        self.p1_vel_y_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_vel_y_scale.set(self.points[0][1]["vel"])
        self.p1_vel_y_scale.grid(column=2, row=point_off + 3, padx=5, pady=5)

        self.p1_acc_label = tk.Label(self.root, text="Acceleration", bg="white")
        self.p1_acc_label.grid(column=0, row=point_off + 4, sticky=tk.SW, padx=5, pady=5)
        self.p1_acc_x_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_acc_x_scale.set(self.points[0][0]["acc"])
        self.p1_acc_x_scale.grid(column=1, row=point_off + 4, padx=5, pady=5)
        self.p1_acc_y_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p1_acc_y_scale.set(self.points[0][1]["acc"])
        self.p1_acc_y_scale.grid(column=2, row=point_off + 4, padx=5, pady=5)

        point_off = 1 * 5
        self.p2_title = tk.Label(self.root, text="Adjustments for Point 2", bg="lightblue")
        self.p2_title.grid(columnspan=3, column=1, row=point_off + 0, padx=5, pady=5)
        self.p2_x_dir = tk.Label(self.root, text="x-Direction", bg="white")
        self.p2_x_dir.grid(column=1, row=point_off + 1, padx=5, pady=5)
        self.p2_y_dir = tk.Label(self.root, text="y-Direction", bg="white")
        self.p2_y_dir.grid(column=2, row=point_off + 1, padx=5, pady=5)

        self.p2_pos_label = tk.Label(self.root, text="Position", bg="white")
        self.p2_pos_label.grid(column=0, row=point_off + 2, sticky=tk.SW, padx=5, pady=5)
        self.p2_pos_x_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_pos_x_scale.set(self.points[1][0]["pos"])
        self.p2_pos_x_scale.grid(column=1, row=point_off + 2, padx=5, pady=5)
        self.p2_pos_y_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_pos_y_scale.set(self.points[1][1]["pos"])
        self.p2_pos_y_scale.grid(column=2, row=point_off + 2, padx=5, pady=5)
        self.p2_vel_label = tk.Label(self.root, text="Velocity", bg="white")
        self.p2_vel_label.grid(column=0, row=point_off + 3, sticky=tk.SW, padx=5, pady=5)
        self.p2_vel_x_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_vel_x_scale.set(self.points[1][0]["vel"])
        self.p2_vel_x_scale.grid(column=1, row=point_off + 3, padx=5, pady=5)
        self.p2_vel_y_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_vel_y_scale.set(self.points[1][1]["vel"])
        self.p2_vel_y_scale.grid(column=2, row=point_off + 3, padx=5, pady=5)
        self.p2_acc_label = tk.Label(self.root, text="Acceleration", bg="white")
        self.p2_acc_label.grid(column=0, row=point_off + 4, sticky=tk.SW, padx=5, pady=5)
        self.p2_acc_x_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_acc_x_scale.set(self.points[1][0]["acc"])
        self.p2_acc_x_scale.grid(column=1, row=point_off + 4, padx=5, pady=5)
        self.p2_acc_y_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p2_acc_y_scale.set(self.points[1][1]["acc"])
        self.p2_acc_y_scale.grid(column=2, row=point_off + 4, padx=5, pady=5)

        point_off = 2 * 5
        self.p3_title = tk.Label(self.root, text="Adjustments for Point 3", bg="lightgreen")
        self.p3_title.grid(columnspan=3, column=1, row=point_off + 0, padx=5, pady=5)
        self.p3_x_dir = tk.Label(self.root, text="x-Direction", bg="white")
        self.p3_x_dir.grid(column=1, row=point_off + 1, padx=5, pady=5)
        self.p3_y_dir = tk.Label(self.root, text="y-Direction", bg="white")
        self.p3_y_dir.grid(column=2, row=point_off + 1, padx=5, pady=5)

        self.p3_pos_label = tk.Label(self.root, text="Position", bg="white")
        self.p3_pos_label.grid(column=0, row=point_off + 2, sticky=tk.SW, padx=5, pady=5)
        self.p3_pos_x_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_pos_x_scale.set(self.points[2][0]["pos"])
        self.p3_pos_x_scale.grid(column=1, row=point_off + 2, padx=5, pady=5)
        self.p3_pos_y_scale = tk.Scale(self.root, from_=self.__min_pos, to=self.__max_pos, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__pos_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_pos_y_scale.set(self.points[2][1]["pos"])
        self.p3_pos_y_scale.grid(column=2, row=point_off + 2, padx=5, pady=5)
        self.p3_vel_label = tk.Label(self.root, text="Velocity", bg="white")
        self.p3_vel_label.grid(column=0, row=point_off + 3, sticky=tk.SW, padx=5, pady=5)
        self.p3_vel_x_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_vel_x_scale.set(self.points[2][0]["vel"])
        self.p3_vel_x_scale.grid(column=1, row=point_off + 3, padx=5, pady=5)
        self.p3_vel_y_scale = tk.Scale(self.root, from_=self.__min_vel, to=self.__max_vel, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__vel_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_vel_y_scale.set(self.points[2][1]["vel"])
        self.p3_vel_y_scale.grid(column=2, row=point_off + 3, padx=5, pady=5)
        self.p3_acc_label = tk.Label(self.root, text="Acceleration", bg="white")
        self.p3_acc_label.grid(column=0, row=point_off + 4, sticky=tk.SW, padx=5, pady=5)
        self.p3_acc_x_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_acc_x_scale.set(self.points[2][0]["acc"])
        self.p3_acc_x_scale.grid(column=1, row=point_off + 4, padx=5, pady=5)
        self.p3_acc_y_scale = tk.Scale(self.root, from_=self.__min_acc, to=self.__max_acc, orient=tk.HORIZONTAL,
                                       length=150, command=self.update_points, resolution=self.__acc_step_size,
                                       bg="white", bd=0, highlightthickness=0)
        self.p3_acc_y_scale.set(self.points[2][1]["acc"])
        self.p3_acc_y_scale.grid(column=2, row=point_off + 4, padx=5, pady=5)

        point_off = 3 * 5

        self.p2_collision_label = tk.Label(self.root, text="Offset for collision trajectory", fg="black",
                                           bg="lightgray")
        self.p2_collision_label.grid(column=1, row=point_off, columnspan=2)

        self.collision_bool = tk.BooleanVar()
        self.collision_check = tk.Checkbutton(self.root, text="show collision\n alternative", bg="white",
                                              variable=self.collision_bool, command=self.update_show_collision, bd=0,
                                              highlightthickness=0)
        self.collision_check.grid(column=0, row=point_off + 1)
        self.collision_bool.set(self.show_collision)

        self.p2_collision_scale = tk.Scale(self.root, from_=self.__min_collision_offset, to=self.__max_collision_offset,
                                           resolution=self.__collision_step_size, length=150,
                                           command=self.update_points, orient=tk.HORIZONTAL, bg="white", bd=0,
                                           highlightthickness=0)
        self.p2_collision_scale.set(self.__collision_offset)
        self.p2_collision_scale.grid(column=1, row=point_off + 1, columnspan=2)

    def get_collision_offset(self):
        return self.p2_collision_scale.get()

    def get_pos_x_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_pos_x_scale.get()
        elif pt_nr == 1:
            return self.p2_pos_x_scale.get()
        else:
            return self.p3_pos_x_scale.get()

    def get_pos_y_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_pos_y_scale.get()
        if pt_nr == 1:
            return self.p2_pos_y_scale.get()
        else:
            return self.p3_pos_y_scale.get()

    def get_vel_x_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_vel_x_scale.get()
        if pt_nr == 1:
            return self.p2_vel_x_scale.get()
        else:
            return self.p3_vel_x_scale.get()

    def get_vel_y_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_vel_y_scale.get()
        if pt_nr == 1:
            return self.p2_vel_y_scale.get()
        else:
            return self.p3_vel_y_scale.get()

    def get_acc_x_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_acc_x_scale.get()
        if pt_nr == 1:
            return self.p2_acc_x_scale.get()
        else:
            return self.p3_acc_x_scale.get()

    def get_acc_y_val(self, pt_nr):
        if pt_nr == 0:
            return self.p1_acc_y_scale.get()
        if pt_nr == 1:
            return self.p2_acc_y_scale.get()
        else:
            return self.p3_acc_y_scale.get()

    def update_order(self):
        self.generator_func = gen_spline_5 if self.order_bool.get() else gen_spline_3
        # print(f"Update order: {self.order_bool.get()}")
        self.update_points(None)

    def update_show_collision(self):
        self.show_collision = self.collision_bool.get()
        self.update_points(None)

    def update_points(self, _):
        self.__collision_offset = self.get_collision_offset()

        for i, (pt, cpt) in enumerate(zip(self.points, self.collision_points)):
            pt[0]["pos"] = cpt[0]["pos"] = self.get_pos_x_val(i)
            pt[0]["vel"] = cpt[0]["vel"] = self.get_vel_x_val(i)
            pt[0]["acc"] = cpt[0]["acc"] = self.get_acc_x_val(i)

            pt[1]["pos"] = self.get_pos_y_val(i)
            cpt[1]["pos"] = pt[1]["pos"] + (self.__collision_offset if i == 1 else 0)

            pt[1]["vel"] = cpt[1]["vel"] = self.get_vel_y_val(i)
            pt[1]["acc"] = cpt[1]["acc"] = self.get_acc_y_val(i)
            # print(pt)
        # print(self.points)
        # plot_splines(self.points, gen_splines_xy(self.points, spline_generator=self.generator_func),
        #              pyplot_axs=self.axs,
        #              update=True, step=self.plot_step_size)
        # self.fig.show()
        self.plot()

    def plot(self, update=True):
        point_lst = []
        plot_lst = []

        plot_points_normal = generate_plot_data(self.points,
                                                gen_splines_xy(self.points, spline_generator=self.generator_func),
                                                step=self.plot_step_size)
        plot_lst.append(plot_points_normal)
        point_lst.append(self.points)
        if self.show_collision:
            plot_points_collision = generate_plot_data(self.points, gen_splines_xy(self.collision_points,
                                                                                   spline_generator=self.generator_func),
                                                       step=self.plot_step_size)
            plot_lst.append(plot_points_collision)
            point_lst.append(self.collision_points)

        plot_from_plot_data(plot_lst, point_lst, self.axs, update=update)
        self.fig.show()


# =================== END CLASSES     ===============================================
# =================== START FUNCTIONS ===============================================

def gen_spline_auto(point_params: list, verbose=False) -> list:
    if len(point_params) < 2:
        raise AttributeError("Not enough points provided")
    if verbose:
        print(f"Generating splines for points {point_params}")

    # tstart = time.perf_counter()
    spline_coefficients = []
    spline_domains = []
    for i, _ in enumerate(point_params[:-1]):
        t = [point_params[i + j]["t"] for j in (0, 1)]

        pos = [point_params[i + j]["pos"] if "pos" in point_params[i+j] else None for j in (0, 1)]
        vel = [point_params[i + j]["vel"] if "vel" in point_params[i+j] else None for j in (0, 1)]
        acc = [point_params[i + j]["acc"] if "acc" in point_params[i+j] else None for j in (0, 1)]

        # print(f"Using values: pos: {pos}, vel: {vel}, acc: {acc}")

        A = np.array([
            [t[0] ** 5, t[0] ** 4, t[0] ** 3, t[0] ** 2, t[0], 1],
            [t[1] ** 5, t[1] ** 4, t[1] ** 3, t[1] ** 2, t[1], 1],
            [5 * t[0] ** 4, 4 * t[0] ** 3, 3 * t[0] ** 2, 2 * t[0], 1, 0],
            [5 * t[1] ** 4, 4 * t[1] ** 3, 3 * t[1] ** 2, 2 * t[1], 1, 0],
            [20 * t[0] ** 3, 12 * t[0] ** 2, 6 * t[0], 2, 0, 0],
            [20 * t[1] ** 3, 12 * t[1] ** 2, 6 * t[1], 2, 0, 0]])

        b = np.array([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]], dtype=float)
        remove_ctr = 0

        for idx, val in enumerate(b):

            if np.isnan(val):
                # print(A, b)
                A = np.delete(A, 0, 1) # erste Spalte von A entfernen
                A = np.delete(A, idx-remove_ctr, 0) # Zeile aus A entfernen
                b = np.delete(b, idx-remove_ctr, 0) # Element aus b entfernen
                remove_ctr += 1
                if verbose:
                    print(f"Missing entry, reducing polynomial degree")
                # print(A, b)

        coeffs = np.linalg.solve(A, b)
        if verbose:
            print(f"Solving the following System A*x=b : \nA: {A}\nb: {b}")
            print(f"Coefficients: {coeffs}")
        spline_coefficients.append(coeffs)
        spline_domains.append(t)
    # tend = time.perf_counter()
    # print(f"Time passed while calculating splines for one dimension: {tend-tstart}")
    return Spline(spline_coefficients, spline_domains)


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


def gen_splines_xyz(point_param_lists: list, spline_generator=gen_spline_3) -> list:
    points_x = [p[0] for p in point_param_lists]
    points_y = [p[1] for p in point_param_lists]
    points_z = [p[2] for p in point_param_lists]

    return [spline_generator(points_x), spline_generator(points_y), spline_generator(points_z)]


def setup_plot(axs):
    axs[0].set_aspect(aspect="equal", adjustable='datalim')

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[0].set_title("Splines in x-direction over time")

    axs[1].set_aspect(aspect="equal", adjustable='datalim')

    axs[1].set_xlabel("t")
    axs[1].set_ylabel("y")
    axs[1].set_title("Splines in y-direction over time")

    xs[2].set_aspect(aspect="equal", adjustable='datalim')

    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Combined Splines in xy-plane")


def generate_plot_data(point_param_lists: list, splines_xy_list=None, step=0.01):
    splines_xy = splines_xy_list
    if splines_xy_list is None:
        splines_xy = gen_splines_xy(point_param_lists)

    t = [np.arange(point_param_lists[i][0]["t"], point_param_lists[i + 1][0]["t"] + step, step) for i in
         range(len(point_param_lists) - 1)]
    x_pts = [list(map(splines_xy[0], t_)) for t_ in t]
    y_pts = [list(map(splines_xy[1], t_)) for t_ in t]

    return t, x_pts, y_pts


def plot_from_plot_data(plot_data: list, point_parameter_lists: list, pyplot_axs=None, update=False):
    if pyplot_axs is None:  # or pyplot_axs.shape != (1, 3):
        fig, axs = plt.subplots(1, 3)
    else:
        axs = pyplot_axs

    if update:
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()

    axs[0].set_aspect(aspect="equal", adjustable='datalim')
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[0].set_title("Splines in x-direction over time")

    axs[1].set_aspect(aspect="equal", adjustable='datalim')
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("y")
    axs[1].set_title("Splines in y-direction over time")

    axs[2].set_aspect(aspect="equal", adjustable='datalim')
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Combined Splines in xy-plane")

    for plot_points, point_param_lists in zip(plot_data, point_parameter_lists):
        t, x_pts, y_pts = plot_points

        # Plot X over t
        for ts, s in zip(t, x_pts):
            axs[0].plot(ts, s)
        for point in point_param_lists:
            axs[0].plot(point[0]["t"], point[0]["pos"], "xk")

        # Plot Y over t
        for ts, s in zip(t, y_pts):
            axs[1].plot(ts, s)
        for point in point_param_lists:
            axs[1].plot(point[1]["t"], point[1]["pos"], "xk")

        # Plot Y over X
        for xs, ys in zip(x_pts, y_pts):
            axs[2].plot(xs, ys)
        for point in point_param_lists:
            axs[2].plot(point[0]["pos"], point[1]["pos"], "xk")


def plot_splines(point_param_lists: list, splines_xy_list=None, step=0.01, pyplot_axs=None, update=False):
    plot_from_plot_data([generate_plot_data(point_param_lists, splines_xy_list, step)], [point_param_lists], pyplot_axs,
                        update)


def get_example_point_data() -> list:
    return [[{"t": 0, "pos": -1, "vel": -5, "acc": 10}, {"t": 0, "pos": 0, "vel": 0.1, "acc": 0.1}],
            [{"t": 0.5, "pos": 0, "vel": 20, "acc": 0}, {"t": 0.5, "pos": 1, "vel": 0, "acc": -2}],
            [{"t": 1.0, "pos": 1, "vel": -5, "acc": -0.25}, {"t": 1, "pos": 0, "vel": -0.25, "acc": 0.1}]]


def get_example_point_data_3D() -> list:
    return [[{"t": 0, "pos": -1, "vel": -5, "acc": 10}, {"t": 0, "pos": 0, "vel": 0.1, "acc": 0.1},
             {"t": 0, "pos": 0, "vel": -0.01, "acc": -0.01}],
            [{"t": 0.5, "pos": 0, "vel": 20, "acc": 0}, {"t": 0.5, "pos": 1, "vel": 0, "acc": -2},
             {"t": 0.5, "pos": 0.1, "vel": 0, "acc": 0.1}],
            [{"t": 1.0, "pos": 1, "vel": -5, "acc": -0.25}, {"t": 1, "pos": 0, "vel": -0.25, "acc": 0.1},
             {"t": 1, "pos": 0, "vel": 0.01, "acc": 0.01}]]


if __name__ == "__main__":
    # ex_points = get_example_point_data()
    # fig, axs = plt.subplots(1, 3)
    # plot_splines(ex_points, pyplot_axs=axs)
    # plot_splines(ex_points, gen_splines_xy(ex_points, spline_generator=gen_spline_5), pyplot_axs=axs)
    # plt.show()
    win_tk = AdjusterGui()

    # data = get_example_point_data_3D()
    # del data[0][0]["vel"]
    # del data[1][0]["vel"]
    # print(data)
    # points_x = [p[0] for p in data]
    # points_y = [p[1] for p in data]
    # points_z = [p[2] for p in data]
    # gen_spline_auto(points_x)

    # splines_3D = gen_splines_xyz(get_example_point_data_3D(), gen_spline_5)
    # print(splines_3D[0])
    # print(splines_3D[1])
    # print(splines_3D[2])

# TODO
# Make scaling on axes the same
