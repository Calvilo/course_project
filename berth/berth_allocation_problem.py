import random
from re import L
from numpy.lib.function_base import average
import pulp as p
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt

def plot_rect(ax, xy, w, h):
    ax.plot(
        [xy[0], xy[0]], [xy[1], xy[1] + h], linewidth="1", label="test", color="green"
    )
    ax.plot(
        [xy[0] + w, xy[0] + w],
        [xy[1], xy[1] + h],
        linewidth="1",
        label="test",
        color="green",
    )
    ax.plot(
        [xy[0], xy[0] + w], [xy[1], xy[1]], linewidth="1", label="test", color="green"
    )
    ax.plot(
        [xy[0], xy[0] + w],
        [xy[1] + h, xy[1] + h],
        linewidth="1",
        label="test",
        color="green",
    )

    ax.text(
        xy[0],
        xy[1],
        str(xy[1]) + "," + str(xy[1] + h) + "\n" + str(xy[0]) + "," + str(xy[0] + w),
        fontdict={"size": 8},
    )


def find_intersection(x_sol, y_sol, vessel_len, operation_time, vessel_n):
    res = []
    z_x_temp = {}
    z_y_temp = {}
    for i in range(vessel_n):
        for j in range(vessel_n):
            if i != j:
                z_x_temp[(i, j)] = 1 if (x_sol[i] <= x_sol[j] - vessel_len[i]) else 0
                z_y_temp[(i, j)] = (
                    1 if (y_sol[i] <= y_sol[j] - operation_time[i]) else 0
                )
                # if x_sol[i] <= x_sol[j] - vessel_len[i]:
                #    z_x_temp[(i, j)] = 1
                # else:
                #    z_x_temp[(i, j)] = 0
                # if y_sol[i] + operation_time[i] <= y_sol[j]:
                #    z_y_temp[(i, j)] = 1
                # else:
                #    z_y_temp[(i, j)] = 0
    for i in range(vessel_n):
        for j in range(i):
            if (
                z_x_temp[(i, j)]
                + z_y_temp[(i, j)]
                + z_x_temp[(j, i)]
                + z_y_temp[(j, i)]
                == 0
            ):
                res.append((i, j))
    return res


if __name__ == "__main__":

    opts, _ = getopt.getopt(
        sys.argv[1:],
        "T:L:N:t:l:",
        [
            "max_time=",
            "berth_len=",
            "vessel_num=",
            "average_time=",
            "average_len=",
            "average_time_cost=",
            "average_dist_cost=",
        ],
    )
    ## 问题参数
    max_berth_len = 1500
    vessel_num = 35
    time_max = 400
    max_operation_time = 40
    average_time = 24
    average_len = 80
    average_time_cost = 500
    average_dist_cost = 500

    MM = 100000000

    for opt, val in opts:
        if opt in ("--berth_len", "-L"):
            print("max berth len is specified as :", val)
            max_berth_len = int(val)
        if opt in ("-N", "--vessel_num"):
            print("vessel num is specified as :", val)
            vessel_num = int(val)
        if opt in ("-T", "--max_time"):
            print("max time is specified as : ", val)
            time_max = int(val)
        if opt in ("-t", "--average_time"):
            print("average time is specified as : ", val)
            average_time = int(val)
        if opt in ("-l", "--average_len"):
            print("average len is specified as : ", val)
            average_len = int(val)
        if opt in ("--average_time_cost"):
            print("average time cost is specified as : ", val)
            average_time_cost = int(val)
        if opt in ("--average_dist_cost"):
            print("average dist cost is specified as : ", val)
            average_dist_cost = int(val)

    vessel_len_series = [
        random.randint(average_len - 10, average_len + 10) for i in range(vessel_num)
    ]
    operate_time_series = [
        random.randint(average_time - 5, average_time + 5) for i in range(vessel_num)
    ]
    best_berth_series = [random.randint(0, max_berth_len) for i in range(vessel_num)]
    arrive_time_series = [random.randint(0, time_max) for i in range(vessel_num)]
    departure_time_series = np.array(arrive_time_series) + np.array(operate_time_series)
    dist_cost_series = [
        random.randint(average_dist_cost - 100, average_dist_cost + 100)
        for i in range(vessel_num)
    ]
    time_cost_series = [
        random.randint(average_time_cost - 100, average_time_cost + 100)
        for i in range(vessel_num)
    ]
    ## 变量及问题定义
    pro = p.LpProblem("berth", p.LpMinimize)

    z_index = [(i, j) for i in range(vessel_num) for j in range(vessel_num) if (i != j)]
    index = range(vessel_num)

    # variables definition
    x = p.LpVariable.dict("x", index, 0, None, cat=p.LpContinuous)
    y = p.LpVariable.dict("y", index, 0, None, cat=p.LpContinuous)
    mid_x = p.LpVariable.dict("mid_x", index, 0, None, cat=p.LpContinuous)
    mid_y = p.LpVariable.dict("mid_for_y", index, 0, None, cat=p.LpContinuous)
    z_x = p.LpVariable.dicts("z_x", z_index, cat=p.LpBinary)
    z_y = p.LpVariable.dicts("z_y", z_index, cat=p.LpBinary)

    # pro definition
    pro += sum(
        [dist_cost_series[i] * mid_x[i] + time_cost_series[i] * mid_y[i] for i in index]
    )
    #  constrait definition
    for i in index:
        # abs
        pro += mid_x[i] >= 0
        pro += -mid_x[i] <= x[i] - best_berth_series[i]
        pro += mid_x[i] >= x[i] - best_berth_series[i]
        # max
        pro += mid_y[i] >= 0
        pro += mid_y[i] >= y[i] + operate_time_series[i] - departure_time_series[i]
        # boundray
        pro += x[i] + vessel_len_series[i] <= max_berth_len
        pro += y[i] >= arrive_time_series[i]
        pro += x[i] >= 0

    print("solve res is ：", pro.solve(), "\n")
    intersections = [(0, 0)]
    ## 迭代求解
    while intersections and pro.status == 1:
        x_sol = [p.value(x[i]) for i in index]
        y_sol = [p.value(y[i]) for i in index]

        print(
            "##########find_intersection :",
            find_intersection(
                x_sol, y_sol, vessel_len_series, operate_time_series, vessel_num
            ),
        )
        intersections = find_intersection(
            x_sol, y_sol, vessel_len_series, operate_time_series, vessel_num
        )

        for item in intersections:
            pro += x[item[0]] + vessel_len_series[item[0]] <= x[item[1]] + MM * (
                1 - z_x[(item[0], item[1])]
            )
            pro += x[item[1]] + vessel_len_series[item[1]] <= x[item[0]] + MM * (
                1 - z_x[(item[1], item[0])]
            )
            pro += y[item[0]] + operate_time_series[item[0]] <= y[item[1]] + MM * (
                1 - z_y[(item[0], item[1])]
            )
            pro += y[item[1]] + operate_time_series[item[1]] <= y[item[0]] + MM * (
                1 - z_y[(item[1], item[0])]
            )
            pro += (
                z_x[item]
                + z_x[(item[1], item[0])]
                + z_y[item]
                + z_y[(item[1], item[0])]
                >= 1
            )
        pro.solve()
        print(p.LpStatus[pro.status])

    pro.writeLP("lp_vessle_num_" + str(vessel_num) + ".txt")
    pro.writeMPS("mps_vessel_num_" + str(vessel_num) + ".txt")
    print("best verth loc:", best_berth_series)
    print("vessel_len:", vessel_len_series)
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for i in index:
        plot_rect(
            ax,
            (x[i].varValue, y[i].varValue),
            vessel_len_series[i],
            operate_time_series[i],
        )
        plt.text(x[i].varValue, y[i].varValue - time_max / 60, i, color="green")
        plt.text(
            best_berth_series[i] - max_berth_len / 100,
            arrive_time_series[i],
            i,
            color="blue",
        )

    ax.scatter(best_berth_series, arrive_time_series, color="blue")
    ax.set_title(
        "berth allocation result, vessel num is "
        + str(vessel_num)
        + ",best obj is: "
        + str(p.value(pro.objective))
    )
    ax.set_xlabel("Berths positon")
    ax.set_ylabel("Date (hour) ")
    ax.grid(True)
    plt.savefig("vessel_num_" + str(vessel_num) + ".svg")
    plt.show()
    for v in pro.variables():
        print(v.name, "=", v.varValue)
