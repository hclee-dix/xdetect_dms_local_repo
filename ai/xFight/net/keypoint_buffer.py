class KeyPointSequence(object):
    def __init__(self, max_size=100):
        self.frames = 0
        self.kpts = []
        self.bboxes = []
        self.max_size = max_size

    def save(self, kpt, bbox):
        self.kpts.append(kpt)
        self.bboxes.append(bbox)
        self.frames += 1
        if self.frames == self.max_size:
            return True
        return False


class KeyPointBuff(object):
    def __init__(self, max_size=100):
        self.flag_track_interrupt = False
        self.keypoint_saver = dict()
        self.max_size = max_size
        self.id_to_pop = set()
        self.flag_to_pop = False

    def get_state(self):
        return self.flag_to_pop

    def update(self, kpt_res, mot_res):
        kpts = kpt_res.get('keypoint')[0]
        bboxes = kpt_res.get('bbox')
        mot_bboxes = mot_res.get('boxes')
        updated_id = set()

        for idx in range(len(kpts)):
            tracker_id = mot_bboxes[idx, 0]
            updated_id.add(tracker_id)

            kpt_seq = self.keypoint_saver.get(tracker_id,
                                              KeyPointSequence(self.max_size))
            is_full = kpt_seq.save(kpts[idx], bboxes[idx])
            self.keypoint_saver[tracker_id] = kpt_seq

            #Scene1: result should be popped when frames meet max size
            if is_full:
                self.id_to_pop.add(tracker_id)
                self.flag_to_pop = True

        #Scene2: result of a lost tracker should be popped
        interrupted_id = set(self.keypoint_saver.keys()) - updated_id
        if len(interrupted_id) > 0:
            self.flag_to_pop = True
            self.id_to_pop.update(interrupted_id)

    def get_collected_keypoint(self):
        """
            Output (List): List of keypoint results for Skeletonbased Recognition task, where 
                           the format of each element is [tracker_id, KeyPointSequence of tracker_id]
        """
        output = []
        for tracker_id in self.id_to_pop:
            output.append([tracker_id, self.keypoint_saver[tracker_id]])
            del (self.keypoint_saver[tracker_id])
        self.flag_to_pop = False
        self.id_to_pop.clear()
        return output