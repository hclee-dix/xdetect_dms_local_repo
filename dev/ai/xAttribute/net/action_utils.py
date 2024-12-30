# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class ActionVisualHelper(object):
    def __init__(self, frame_life=20):
        self.frame_life = frame_life
        self.action_history = {}

    def get_visualize_ids(self):
        id_detected = self.check_detected()
        return id_detected

    def check_detected(self):
        id_detected = set()
        conf_detected = set()
        deperate_id = []
        for mot_id in self.action_history:
            self.action_history[mot_id]["life_remain"] -= 1
            if int(self.action_history[mot_id]["class"]) == 0:
                id_detected.add(mot_id)
                conf_detected.add(float(self.action_history[mot_id]["score"]))
            if self.action_history[mot_id]["life_remain"] == 0:
                deperate_id.append(mot_id)
        for mot_id in deperate_id:
            del (self.action_history[mot_id])
        return id_detected,conf_detected

    def update(self, action_res_list):
        for mot_id, action_res in action_res_list:
            if mot_id in self.action_history:
                if int(action_res["class"]) != 0 and int(self.action_history[
                        mot_id]["class"]) == 0:
                    continue
            action_info = self.action_history.get(mot_id, {})
            action_info["class"] = action_res["class"]
            action_info["score"] = action_res["score"]
            action_info["life_remain"] = self.frame_life
            self.action_history[mot_id] = action_info
