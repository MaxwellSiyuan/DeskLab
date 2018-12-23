# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

class MagnetBar:
    empCount = 0

    def __init__(self, x_length, y_length, center, angle, canvas):
        self.x_length = x_length
        self.y_length = y_length
        self.center = center
        self.angle = angle
        self.sin_theta = math.sin(angle)
        self.cos_theta = math.cos(angle)
        self.canvas = canvas

        self.margin_point = []
        self.get_margin_point()
        self.dipoles = []
        self.init_dipoles()

        # 全部写完后可以删掉
        # self.draw_margin()
        #         # self.draw_contour()

    def get_margin_point(self):
        # 1:left up;
        # 2:left down;
        # 3:right top;
        # 4:right down
        x_c1 = self.center[0] - self.y_length/2 * math.sin(self.angle)
        y_c1 = self.center[1] - self.y_length/2 * math.cos(self.angle)
        x_c2 = self.center[0] + self.y_length/2 * math.sin(self.angle)
        y_c2 = self.center[1] + self.y_length/2 * math.cos(self.angle)

        # 最左边的顶点的位置坐标
        x_l1 = int(x_c1 - self.x_length / 2 * math.cos(self.angle))
        x_l2 = int(x_c2 - self.x_length / 2 * math.cos(self.angle))
        y_l1 = int(y_c1 + self.x_length / 2 * math.sin(self.angle))
        y_l2 = int(y_c2 + self.x_length / 2 * math.sin(self.angle))

        self.margin_point.append((x_l1, y_l1))
        self.margin_point.append((x_l2, y_l2))

        # 最右边的顶点的位置坐标
        x_r1 = int(x_c1 + self.x_length / 2 * math.cos(self.angle))
        x_r2 = int(x_c2 + self.x_length / 2 * math.cos(self.angle))
        y_r1 = int(y_c1 - self.x_length / 2 * math.sin(self.angle))
        y_r2 = int(y_c2 - self.x_length / 2 * math.sin(self.angle))

        self.margin_point.append((x_r1, y_r1))
        self.margin_point.append((x_r2, y_r2))

    def is_inside(self, point):
        x = point[0]
        y = point[1]

        def is_inside(center, point):
            if point[0] >= center[0] + self.x_length / 2.0 - 3:
                return False
            elif point[0] <= center[0] - self.x_length / 2.0 + 3:
                return False
            elif point[1] >= center[1] + self.y_length / 2.0 - 3:
                return False
            elif point[1] <= center[1] - self.y_length / 2.0 + 3:
                return False
            else:
                return True

        if self.angle == 0:
            return is_inside(self.center, point)
        else:
            r_point_x = (x - self.center[0]) * self.cos_theta - (y - self.center[1]) * self.sin_theta
            r_point_y = (x - self.center[0]) * self.sin_theta + (y - self.center[1]) * self.cos_theta
            r_point = (r_point_x, r_point_y)
            return is_inside((0, 0), r_point)

    def init_dipoles(self):
        row_num = 12
        column_num = 5

        left_middle_x = (self.margin_point[0][0] + self.margin_point[1][0]) / 2
        left_middle_y = (self.margin_point[0][1] + self.margin_point[1][1]) / 2
        right_middle_x = (self.margin_point[2][0] + self.margin_point[3][0]) / 2
        right_middle_y = (self.margin_point[2][1] + self.margin_point[3][1]) / 2

        left_middle_middle1_x = (self.margin_point[0][0] + left_middle_x) / 2
        left_middle_middle1_y = (self.margin_point[0][1] + left_middle_y) / 2
        right_middle_middle1_x = (self.margin_point[2][0] + right_middle_x) / 2
        right_middle_middle1_y = (self.margin_point[2][1] + right_middle_y) / 2

        left_middle_middle2_x = (self.margin_point[1][0] + left_middle_x) / 2
        left_middle_middle2_y = (self.margin_point[1][1] + left_middle_y) / 2
        right_middle_middle2_x = (self.margin_point[3][0] + right_middle_x) / 2
        right_middle_middle2_y = (self.margin_point[3][1] + right_middle_y) / 2

        for i in range(1, row_num):
            x = int(float(i) / row_num * float(left_middle_middle1_x) + float(row_num - i) / row_num *
                    float(right_middle_middle1_x))
            y = int(float(i) / row_num * float(left_middle_middle1_y) + float(row_num - i) / row_num *
                    float(right_middle_middle1_y))
            self.dipoles.append((x, y))

            x = int(float(i) / row_num * float(left_middle_x) + float(row_num - i) / row_num * float(right_middle_x))
            y = int(float(i) / row_num * left_middle_y + float(row_num - i) / row_num * right_middle_y)
            self.dipoles.append((x, y))

            x = int(float(i) / row_num * float(left_middle_middle2_x) + float(row_num - i) / row_num *
                    float(right_middle_middle2_x))
            y = int(float(i) / row_num * float(left_middle_middle2_y) + float(row_num - i) / row_num *
                    float(right_middle_middle2_y))
            self.dipoles.append((x, y))

    def get_b(self, x, y):
        # print x, y
        # b分别是x方向和y方向的磁场大小
        b = np.empty(2)
        b[0] = 0.0
        b[1] = 0.0
        scale = 8.0
        c = 3.0
        add_b = []
        for i in range(0, len(self.dipoles)):
            dx = x - self.dipoles[i][0]
            dy = y - self.dipoles[i][1]
            r2 = dx * dx + dy * dy
            if r2 == 0:
                add_b_x = 0
                add_b_y = 0
            else:
                r = math.sqrt(r2)
                r3 = r * r2
                cos = dx / r
                sin = dy / r
                m_dot_r = self.cos_theta * cos - self.sin_theta * sin
                # add_b_x = math.fabs(scale * (c * cos * cos - 1) / r3)
                # add_b_y = math.fabs(scale * (c * sin * cos) / r3)
                #
                # if dx < 0:
                #     add_b_x = -add_b_x
                # if dy < 0:
                #     add_b_y = -add_b_y
                add_b_x = scale * (c * cos * m_dot_r - self.cos_theta) / r3
                add_b_y = scale * (c * sin * m_dot_r + self.sin_theta) / r3
                # print add_b_x * add_b_x + add_b_y * add_b_y

            add_b.append((add_b_x, add_b_y, add_b_x * add_b_x + add_b_y * add_b_y))

            # b[0] = b[0] - add_b_x
            # b[1] = b[1] - add_b_y

        # 想着减少舍入误差，但是貌似没什么乱用，懒得改了留着吧
        add_b.sort(key=(lambda x: x[2]))
        add_b = list(zip(*add_b))
        b[0] = -sum(add_b[0])
        b[1] = -sum(add_b[1])
        return b

    def draw_margin(self):
        margin = self.margin_point
        red = (0, 0, 255)
        for i in range(len(margin)):
            cv2.circle(self.canvas, margin[i], 4, red)

    def draw_dipoles(self):
        purple = (153, 50, 204)
        for i in range(len(self.dipoles)):
            x = self.dipoles[i][0]
            y = self.dipoles[i][1]
            cv2.circle(self.canvas, (int(x), int(y)), 4, purple)
            # print self.dipoles[i]

    def draw_contour(self):
        purple = (153, 50, 204)
        cv2.rectangle(self.canvas, self.margin_point[0], self.margin_point[3], purple)


class MultiBars:
    empCount = 0

    def __init__(self, bar_list, canvas):
        self.bar_list = bar_list
        # self.dipoles = []
        # self.init_dipoles()
        self.canvas = canvas

    def is_inside(self, point):
        for bar in self.bar_list:
            is_inside = bar.is_inside(point)
            if is_inside:
                return is_inside
        return is_inside

    '''
    def init_dipoles(self):
        for bar in self.bar_list:
            self.dipoles.extend(bar.dipoles)
    '''

    def get_b(self, x, y):
        b = np.empty(2)
        b[0] = 0.0
        b[1] = 0.0

        for bar in self.bar_list:
            b[0] = b[0] + bar.get_b(x, y)[0]
            b[1] = b[1] + bar.get_b(x, y)[1]
        return b

    def draw_magnets(self):
        red = (0, 0, 255)

        for bar in self.bar_list:
            # 设置左端头的出发点
            start_num = 10
            start_point = []
            for i in range(1, start_num):
                x = int(float(i) / start_num * float(bar.margin_point[0][0]) + float(start_num - i) /
                        start_num * float(bar.margin_point[1][0]))
                y = int(float(i) / start_num * float(bar.margin_point[0][1]) + float(start_num - i) /
                        start_num * float(bar.margin_point[1][1]))
                cv2.circle(self.canvas, (int(x), int(y)), 4, red)
                start_point.append((int(x), int(y)))

            for i in range(start_num - 1):
                k = 0
                new_x = start_point[i][0]
                new_y = start_point[i][1]
                # 这里如果画布大小改变的话参数需要跟着变
                while ((not self.is_inside((new_x, new_y))) and (new_x < 1280)
                       and (new_y < 720) and (new_x > 0) and (new_y > 0)):
                    old_x = new_x
                    old_y = new_y

                    new_b = self.get_b(new_x, new_y)
                    old_b = new_b
                    b_length = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])

                    new_x = old_x + old_b[0]
                    new_y = old_y + old_b[1]

                    new_b = self.get_b(new_x, new_y)
                    b_length_new = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])
                    cos = (new_b[0] * old_b[0] + new_b[1] * old_b[1]) / (b_length * b_length_new)
                    while cos < 1 / 2:
                        old_b[0] = old_b[0] / 2
                        old_b[1] = old_b[1] / 2
                        b_length = b_length / 2

                        new_x = old_x + old_b[0]
                        new_y = old_y + old_b[1]

                        new_b = self.get_b(new_x, new_y)
                        b_length_new = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])
                        cos = (new_b[0] * old_b[0] + new_b[1] * old_b[1]) / (b_length * b_length_new)

                    uniform = max(math.fabs(old_b[0]), math.fabs(old_b[1]))
                    uniform_min = min(math.fabs(old_b[0]), math.fabs(old_b[1]))

                    if uniform_min < 1:
                        old_b[0] = old_b[0] / uniform_min
                        old_b[1] = old_b[1] / uniform_min
                        b_length = b_length / uniform_min
                    elif uniform > 5:
                        old_b[0] = old_b[0] / uniform
                        old_b[1] = old_b[1] / uniform
                        b_length = b_length / uniform

                    while b_length > 5:
                        old_b[0] = old_b[0] / 2
                        old_b[1] = old_b[1] / 2
                        b_length = b_length / 2

                    new_x = old_x + old_b[0]
                    new_y = old_y + old_b[1]
                    # print(old_b[0], old_b[1])
                    cv2.line(self.canvas, (int(old_x), int(old_y)), (int(new_x), int(new_y)), red, 3)
                    # cv2.circle(img, (int(new_x), int(new_y)), 1, color)
                    k = k + 1

            # 设置右端头的出发点
            right_start_point = []
            for i in range(1, start_num):
                x = int(float(i) / start_num * float(bar.margin_point[2][0]) + float(start_num - i) /
                        start_num * float(bar.margin_point[3][0]))
                y = int(float(i) / start_num * float(bar.margin_point[2][1]) + float(start_num - i) /
                        start_num * float(bar.margin_point[3][1]))
                cv2.circle(bar.canvas, (int(x), int(y)), 4, red)
                right_start_point.append((int(x), int(y)))

            # 画右端头开始的线
            for i in range(start_num - 1):
                k = 0
                new_x = right_start_point[i][0]
                new_y = right_start_point[i][1]
                # 这里如果画布大小改变的话参数需要跟着变
                while ((not self.is_inside((new_x, new_y))) and (new_x < 1280) and (new_y < 720)
                       and (new_x > 0) and (new_y > 0)):
                    old_x = new_x
                    old_y = new_y

                    new_b = self.get_b(new_x, new_y)
                    new_b[0] = -new_b[0]
                    new_b[1] = -new_b[1]
                    old_b = new_b
                    b_length = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])

                    new_x = old_x + old_b[0]
                    new_y = old_y + old_b[1]

                    new_b = self.get_b(new_x, new_y)
                    new_b[0] = - new_b[0]
                    new_b[1] = - new_b[1]
                    b_length_new = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])
                    cos = (new_b[0] * old_b[0] + new_b[1] * old_b[1]) / (b_length * b_length_new)
                    while cos < 1 / 2:
                        old_b[0] = old_b[0] / 2
                        old_b[1] = old_b[1] / 2
                        b_length = b_length / 2

                        new_x = old_x + old_b[0]
                        new_y = old_y + old_b[1]

                        new_b = self.get_b(new_x, new_y)
                        b_length_new = math.sqrt(new_b[0] * new_b[0] + new_b[1] * new_b[1])
                        cos = (new_b[0] * old_b[0] + new_b[1] * old_b[1]) / (b_length * b_length_new)

                    uniform = max(math.fabs(old_b[0]), math.fabs(old_b[1]))
                    uniform_min = min(math.fabs(old_b[0]), math.fabs(old_b[1]))

                    if uniform_min < 1:
                        old_b[0] = old_b[0] / uniform_min
                        old_b[1] = old_b[1] / uniform_min
                        b_length = b_length / uniform_min
                    elif uniform > 5:
                        old_b[0] = old_b[0] / uniform
                        old_b[1] = old_b[1] / uniform
                        b_length = b_length / uniform

                    while b_length > 5:
                        old_b[0] = old_b[0] / 2
                        old_b[1] = old_b[1] / 2
                        b_length = b_length / 2

                    # print(old_b)
                    new_x = old_x + old_b[0]
                    new_y = old_y + old_b[1]

                    cv2.line(self.canvas, (int(old_x), int(old_y)), (int(new_x), int(new_y)), red, 3)
                    # cv2.circle(img, (int(new_x), int(new_y)), 1, color)
                    k = k + 1


# if __name__ == "__main__":
#     canvas = np.zeros((720, 1280, 3), dtype="uint8")
#     x_length = 100
#     y_length = 20
#     x_center1 = 650
#     y_center1 = 200
#     x_center2 = 650
#     y_center2 = 500
#     bar1 = MagnetBar(x_length, y_length, (x_center1, y_center1), 0, canvas)
#     bar2 = MagnetBar(x_length, y_length, (x_center2, y_center2), math.pi, canvas)
#     bars = []
#     bars.append(bar1)
#     bars.append(bar2)
#
#     multi_bar = MultiBars(bars, canvas)
#     multi_bar.draw_magnets()
#
#     # plt.imshow(canvas[..., ::-1])
#     # plt.show()
#
#     cv2.imshow("Canvas", canvas)
#     # cv2.imwrite("canvas.jpg", canvas)
#     cv2.waitKey(0)
