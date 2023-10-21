from itertools import combinations
import numpy as np


class Dolve:
    def __init__(self, n_pe, s1, s2, s3, s4, s5_s1, s5_s2, s5_s3, s5_s4, true_value):
        self.true_value = true_value
        self.n_pe = n_pe
        self.n = n_pe - 1
        self.v = np.zeros((n_pe, 1))
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5_s1 = s5_s1
        self.s5_s2 = s5_s2
        self.s5_s3 = s5_s3
        self.s5_s4 = s5_s4

    def update_v(self):
        self.v1 = [self.s1, self.s2, self.s3, self.s4, self.s5_s1]
        self.v2 = [self.s2, self.s1, self.s3, self.s4, self.s5_s2]
        self.v3 = [self.s3, self.s1, self.s2, self.s4, self.s5_s3]
        self.v4 = [self.s4, self.s1, self.s2, self.s3, self.s5_s4]

    def update_s(self, avg_p1, avg_p2, avg_p3, avg_p4):
        self.s1 = avg_p1
        self.s2 = avg_p2
        self.s3 = avg_p3
        self.s4 = avg_p4

    def epoch(self):
        self.update_v()
        p1 = np.sort(self.v1)
        avg_p1 = np.sum(p1[1:4]) / self.n
        p2 = np.sort(self.v2)
        avg_p2 = np.sum(p2[1:4]) / self.n
        p3 = np.sort(self.v3)
        avg_p3 = np.sum(p3[1:4]) / self.n
        p4 = np.sort(self.v4)
        avg_p4 = np.sum(p4[1:4]) / self.n
        self.update_s(avg_p1, avg_p2, avg_p3, avg_p4)
        self.v[0] = avg_p1
        self.v[1] = avg_p2
        self.v[2] = avg_p3
        self.v[3] = avg_p4
        print(self.v)

    def print_results(self):
        print(">> Dolve Results: ")
        print('s1: ', self.s1)
        print('s2: ', self.s2)
        print('s3: ', self.s3)
        print('s4: ', self.s4)

    def run(self, delta, type: str = 'dolve'):
        # DOLVE: delta defined as the region containing all correct readings
        # MAHANEY: or the longest distance from the actual value to an acceptable reading
        err = np.inf
        errs = []
        while err > delta:
            self.epoch()

            if type == 'dolve':
                err = np.abs(np.max(self.v)-np.min(self.v))
            else:
                err = np.abs(np.min(self.v) - self.true_value)
            print(f">> epoch: {len(errs)+1} - error: {err}")
            errs.append(err)
            if len(errs) > 100:
                break
        self.print_results()
        return errs


class Mahaney:
    def __init__(self, n_pe, s1, s2, s3, s4, s5_s1, s5_s2, s5_s3, s5_s4, true_value):
        self.true_value = true_value
        self.n_pe = n_pe
        self.n = n_pe
        self.v = np.zeros((n_pe, 1))
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5_s1 = s5_s1
        self.s5_s2 = s5_s2
        self.s5_s3 = s5_s3
        self.s5_s4 = s5_s4

    def update_v(self):
        self.v1 = [self.s1, self.s2, self.s3, self.s4, self.s5_s1]
        self.v2 = [self.s2, self.s1, self.s3, self.s4, self.s5_s2]
        self.v3 = [self.s3, self.s1, self.s2, self.s4, self.s5_s3]
        self.v4 = [self.s4, self.s1, self.s2, self.s3, self.s4, self.s5_s4]

    def update_s(self, avg_p1, avg_p2, avg_p3, avg_p4):
        self.s1 = avg_p1
        self.s2 = avg_p2
        self.s3 = avg_p3
        self.s4 = avg_p4

    def epoch(self):
        self.update_v()
        p1 = np.sort(self.v1)
        avg_p1 = p1.mean()
        p2 = np.sort(self.v2)
        avg_p2 = p2.mean()
        p3 = np.sort(self.v3)
        avg_p3 = p3.mean()
        p4 = np.sort(self.v4)
        avg_p4 = p4.mean()
        self.update_s(avg_p1, avg_p2, avg_p3, avg_p4)
        self.v[0] = avg_p1
        self.v[1] = avg_p2
        self.v[2] = avg_p3
        self.v[3] = avg_p4

    def print_results(self):
        print(">> Dolve Results: ")
        print('s1: ', self.s1)
        print('s2: ', self.s2)
        print('s3: ', self.s3)
        print('s4: ', self.s4)

    def run(self, delta, type: str = 'dolve'):
        # DOLVE: delta defined as the region containing all correct readings
        # MAHANEY: or the longest distance from the actual value to an acceptable reading
        err = np.inf
        errs = []
        while err > delta:
            self.epoch()

            if type == 'dolve':
                err = np.abs(np.max(self.v)-np.min(self.v))
            else:
                err = np.abs(np.min(self.v) - self.true_value)
            print(f">> epoch: {len(errs)+1} - error: {err}")
            errs.append(err)
            if len(errs) > 100:
                break

        self.print_results()
        return errs


class HybridAlgo:
    def __init__(self, n_pe, F, s1, s2, s3, s4, s5_s1, s5_s2, s5_s3, s5_s4, s1_err, s2_err, s3_err, s4_err, s5_s1_err, s5_s2_err, s5_s3_err, s5_s4_err, true_value):
        self.true_value = true_value
        self.N = n_pe
        self.F = F
        # self.v = np.zeros((n_pe, 1))
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5_s1 = s5_s1
        self.s5_s2 = s5_s2
        self.s5_s3 = s5_s3
        self.s5_s4 = s5_s4
        self.s1_err = s1_err
        self.s2_err = s2_err
        self.s3_err = s3_err
        self.s4_err = s4_err
        self.s5_s1_err = s5_s1_err
        self.s5_s2_err = s5_s2_err
        self.s5_s3_err = s5_s3_err
        self.s5_s4_err = s5_s4_err

    def initialize(self):
        self.r1 = (self.s1 - self.s1_err, self.s1 + self.s1_err)
        self.r2 = (self.s2 - self.s2_err, self.s2 + self.s2_err)
        self.r3 = (self.s3 - self.s3_err, self.s3 + self.s3_err)
        self.r4 = (self.s4 - self.s4_err, self.s4 + self.s4_err)
        self.r5_s1 = (self.s5_s1 - self.s5_s1_err, self.s5_s1 + self.s5_s1_err)
        self.r5_s2 = (self.s5_s2 - self.s5_s2_err, self.s5_s2 + self.s5_s2_err)
        self.r5_s3 = (self.s5_s3 - self.s5_s3_err, self.s5_s3 + self.s5_s3_err)
        self.r5_s4 = (self.s5_s4 - self.s5_s4_err, self.s5_s4 + self.s5_s4_err)

        self.r = [self.r1, self.r2, self.r3, self.r4]
        self.s = [self.s1, self.s2, self.s3, self.s4]
        self.data1 = [self.r1, self.r2, self.r3, self.r4, self.r5_s1]
        self.data2 = [self.r1, self.r2, self.r3, self.r4, self.r5_s2]
        self.data3 = [self.r1, self.r2, self.r3, self.r4, self.r5_s3]
        self.data4 = [self.r1, self.r2, self.r3, self.r4, self.r5_s4]

        self.data = [self.data1, self.data2, self.data3, self.data4]

    def sensor_fusion(self, V, N, F):
        sorted_v = sorted(V, key=lambda x: x[0])

        low_bounds = [sorted_v[i][0] for i in range(len(sorted_v))]
        upper_bounds = [sorted_v[i][1] for i in range(len(sorted_v))]

        r = low_bounds + upper_bounds

        all_ranges = [tuple(sorted(ra)) for ra in list(combinations(r, 2))]

        stats = dict()
        for range_ in all_ranges:
            for reading in V:
                if reading[0] <= range_[0] and range_[1] <= reading[1]:
                    if range_ in stats:
                        stats[range_] += 1
                    else:
                        stats[range_] = 1

        self.A = [(stat, stats[stat])
                  for stat in stats.keys() if stats[stat] >= N-F]
        self.A = self.filter_intervals(self.A)
        res = (min([low for (low, _), _ in self.A]),
               max([up for (_, up), _ in self.A]))
        return res

    def update(self, r, s):
        self.s1 = s[0]
        self.s2 = s[1]
        self.s3 = s[2]
        self.s4 = s[3]
        # self.r1 = (self.s1 - self.s1_err, self.s1 + self.s1_err)
        # self.r2 = (self.s2 - self.s2_err, self.s2 + self.s2_err)
        # self.r3 = (self.s3 - self.s3_err, self.s3 + self.s3_err)
        # self.r4 = (self.s4 - self.s4_err, self.s4 + self.s4_err)
        self.r1 = r[0]
        self.r2 = r[1]
        self.r3 = r[2]
        self.r4 = r[3]
        self.s = [self.s1, self.s2, self.s3, self.s4]
        self.r = [self.r1, self.r2, self.r3, self.r4]
        self.data1 = [self.r1, self.r2, self.r3, self.r4, self.r5_s1]
        self.data2 = [self.r1, self.r2, self.r3, self.r4, self.r5_s2]
        self.data3 = [self.r1, self.r2, self.r3, self.r4, self.r5_s3]
        self.data4 = [self.r1, self.r2, self.r3, self.r4, self.r5_s4]

        self.data = [self.data1, self.data2, self.data3, self.data4]

    def midpoint(self, a, b):
        return (a+b)/2

    def filter_intervals(self, A):
        filtered_data = []
        lower_bounds_set = set()
        A.sort(key=lambda x: x[0][1])
        for interval, value in A:
            lower_bound, upper_bound = interval
            # Check if the lower bound is not in the set
            if lower_bound not in lower_bounds_set:
                filtered_data.append(((lower_bound, upper_bound), value))
                lower_bounds_set.add(lower_bound)
        return filtered_data

    def update_s(self):
        tot = 0
        div = 0
        for ran, count in self.A:
            # print(self.midpoint(ran[0], ran[1]), count)
            tot += count * self.midpoint(ran[0], ran[1])
            div += count
        return tot/div

    def epoch(self, i):
        r = []
        s = []
        for data in self.data:
            r.append(self.sensor_fusion(data, self.N, self.F))
            s.append(self.update_s())
        self.update(r, s)
        print(f">> Epoch {i+1}: {s} - {r}")
        delta = np.abs(np.max(self.s)-np.min(self.s))
        return delta

    def run(self, n):
        self.initialize()
        precisions = []
        for i in range(n):
            delta = self.epoch(i)
            print(">> Precision: ", delta)
            precisions.append(delta)
        return precisions
