import numpy as np

class LinearInterpolation():
    def __init__(self, joints):
        """
        :param: joints: N X 2 array
            name of objective
        """
        super(self.__class__, self).__init__()
        self.joints = joints
        self.joint_num = self.joints.shape[1]
        self.joints_detection = self.joints[:, :, 2] > 0.1
        self.joint_missing_regions = {}

        for i in range(self.joint_num):
            self.joint_missing_regions[i] = []

    def missing_regions_detection(self):

        for i in range(self.joint_num):
            start, end, flag = 0, 0, False
            for j in range(self.joints_detection.shape[0] - 1):
                if self.joints_detection[j][i] and not self.joints_detection[j + 1][i]:
                    flag = True
                    start = j
                if not self.joints_detection[j][i] and self.joints_detection[j + 1][i]:
                    end = j + 1
                    if flag:
                        self.joint_missing_regions[i].append([start, end])
                        flag = False
    def linear(self, p0, p1, t):
        """
        :param: p0: 1 x 2 numpy
            the first data point
        :param: p1: 1 x 2 numpy
            the second data point
        :param: t1: int
            time interval
        """
        try:
            abs(t) < 1e-6
        except ValueError:
            print('t0 and t1 must be different')

        return p0, (p1 - p0) / t

    def __call__(self):
        self.missing_regions_detection()

        filter_joint = self.joints[:, :, :2].copy()

        for i in range(self.joint_num):
            for [start_index, end_index] in self.joint_missing_regions[i]:
                interval = end_index - start_index
                b, k = self.linear(self.joints[start_index, i, :2], self.joints[end_index, i, :2], end_index - start_index)
                filter_joint[start_index: end_index, i, :] = b + np.outer(np.arange(interval), k)

        return filter_joint
