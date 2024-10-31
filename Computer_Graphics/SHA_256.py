import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ipywidgets as widgets
from ipywidgets import interact



# 函数：绘制立方体
def plot_cube(ax, cube_min, cube_max, color = 'cyan', alpha = 0.3):
    # 立方体的8个顶点
    r = np.array([[cube_min[0], cube_min[1], cube_min[2]],
                  [cube_max[0], cube_min[1], cube_min[2]],
                  [cube_max[0], cube_max[1], cube_min[2]],
                  [cube_min[0], cube_max[1], cube_min[2]],
                  [cube_min[0], cube_min[1], cube_max[2]],
                  [cube_max[0], cube_min[1], cube_max[2]],
                  [cube_max[0], cube_max[1], cube_max[2]],
                  [cube_min[0], cube_max[1], cube_max[2]]])

    # 立方体的12条边
    verts = [[r[0], r[1], r[2], r[3]],
             [r[4], r[5], r[6], r[7]],
             [r[0], r[1], r[5], r[4]],
             [r[2], r[3], r[7], r[6]],
             [r[1], r[2], r[6], r[5]],
             [r[4], r[7], r[3], r[0]]]

    # 绘制立方体
    ax.add_collection3d(Poly3DCollection(verts, facecolors = color, linewidths = 1, edgecolors = 'r', alpha = alpha))


# 函数：绘制直线
def plot_line(ax, origin, direction, t_min = -10, t_max = 10, color = 'blue'):
    t = np.linspace(t_min, t_max, 100)
    line_points = origin[:, np.newaxis] + direction[:, np.newaxis] * t
    ax.plot(line_points[0], line_points[1], line_points[2], color = color)


def check_intersection(cube_min, cube_max, O, D):
    # 判断直线是否与立方体相交，直线的参数方程为：P = O + t * D
    # 其中 P 为直线上的点，O 为直线的起点，D 为方向向量，t 为参数
    # 立方体的六个面分别为 x = cube_min[0], x = cube_max[0], y = cube_min[1], y = cube_max[1], z = cube_min[2], z = cube_max[2]
    # 将直线的参数方程代入立方体的六个面的方程中，可以得到直线与立方体六个面的交点
    # 如果交点在立方体内部，则直线与立方体相交

    # 计算直线的参数方程
    t_enter = -np.inf
    t_exit = np.inf

    for i in range(3):
        t_min = (cube_min[i] - O[i]) / D[i]
        t_max = (cube_max[i] - O[i]) / D[i]
        t_enter = max(t_enter, t_min)
        t_exit = min(t_exit, t_max)
    if t_enter <= t_exit:
        return True
    return False




# 可视化函数
def visualize_line_and_cube(cube_min_x = 0, cube_min_y = 0, cube_min_z = 0,
                            cube_max_x = 3, cube_max_y = 3, cube_max_z = 3,
                            line_origin_x = 0, line_origin_y = 0, line_origin_z = 0,
                            line_dir_x = 0, line_dir_y = 1, line_dir_z = 0):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    # 设置坐标轴范围
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    # 绘制立方体和直线
    cube_min = [cube_min_x, cube_min_y, cube_min_z]
    cube_max = [cube_max_x, cube_max_y, cube_max_z]
    line_origin = np.array([line_origin_x, line_origin_y, line_origin_z])
    line_direction = np.array([line_dir_x, line_dir_y, line_dir_z])

    plot_cube(ax, cube_min, cube_max)
    plot_line(ax, line_origin, line_direction)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    res = check_intersection([cube_min_x, cube_min_y, cube_min_z], [cube_max_x, cube_max_y, cube_max_z],[line_origin_x, line_origin_y, line_origin_z], [line_dir_x, line_dir_y, line_dir_z])
    if res:
        print('The line intersects with the cube.')
    else:
        print('The line does not intersect with the cube.')


# 使用ipywidgets交互
interact(visualize_line_and_cube,
         cube_min_x = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 0, description = 'Cube Min X'),
         cube_min_y = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 0, description = 'Cube Min Y'),
         cube_min_z = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 0, description = 'Cube Min Z'),
         cube_max_x = widgets.FloatSlider(min = -5, max = 10, step = 0.1, value = 3, description = 'Cube Max X'),
         cube_max_y = widgets.FloatSlider(min = -5, max = 10, step = 0.1, value = 3, description = 'Cube Max Y'),
         cube_max_z = widgets.FloatSlider(min = -5, max = 10, step = 0.1, value = 3, description = 'Cube Max Z'),
         line_origin_x = widgets.FloatSlider(min = -10, max = 10, step = 0.1, value = 0, description = 'Line Origin X'),
         line_origin_y = widgets.FloatSlider(min = -10, max = 10, step = 0.1, value = 0, description = 'Line Origin Y'),
         line_origin_z = widgets.FloatSlider(min = -10, max = 10, step = 0.1, value = 0, description = 'Line Origin Z'),
         line_dir_x = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 1, description = 'Line Dir X'),
         line_dir_y = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 1, description = 'Line Dir Y'),
         line_dir_z = widgets.FloatSlider(min = -5, max = 5, step = 0.1, value = 1, description = 'Line Dir Z')
         )
