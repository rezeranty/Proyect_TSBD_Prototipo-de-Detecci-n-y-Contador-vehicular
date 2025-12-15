# -----------------------------
# SORT tracker (original)
# -----------------------------
import numpy as np
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = bbox
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.history = []

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox)
        self.hit_streak += 1

    def predict(self):
        self.kf.predict()
        return self.kf.x

    def get_state(self):
        return self.kf.x[:4]


class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold

    def update(self, dets):
        self.frame_count += 1

        if len(dets) == 0:
            return np.empty((0, 5))

        # update trackers
        trks = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        matched = linear_assignment(1 - np.array([[iou(det[:4], trk[:4]) for trk in trks] for det in dets]))

        # delete unmatched trackers and detections
        unmatched_dets = []
        for d, det in enumerate(dets):
            if d not in matched[:, 0]:
                unmatched_dets.append(d)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        ret = []
        for trk in self.trackers:
            bbox = trk.get_state()
            ret.append(np.concatenate((bbox, [trk.id])).reshape(1, -1))

        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))
