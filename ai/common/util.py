import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F


def softmax_numpy(data,dim=-1)->npt.NDArray[np.float32]:
    exp = np.exp(data-np.max(data,axis=dim,keepdims=True))
    return exp / np.sum(exp,axis=dim,keepdims=True)


def softmax(data,dim=-1):
    if isinstance(data,np.ndarray):
        data = data.copy()
        return softmax_numpy(data,dim)
    elif isinstance(data,torch.Tensor):
        data = data.clone()
        return F.softmax(data,dim)
    else: return data

class EMA:
    def __init__(self, N=1, C=26, window_size=30):
        self.N = N  # 객체 수
        self.C = C  # 값 개수
        self.window_size = window_size  # EMA 윈도우 크기
        self.alpha = 2 / (window_size + 1)  # 감쇠 계수

        self.ema = {}
        self.ema_buffer = {}
        self.actual_buffer = {}
        self.current_index = {}

    def update(self, keys, data):
        keys = np.array(keys)
        if keys.shape[0] != data.shape[0]:
            return
        
        for i, key in enumerate(keys):
            if key not in self.ema:
                self.ema[key] = np.zeros(self.C)
                self.ema_buffer[key] = np.zeros((self.C, self.window_size))
                self.actual_buffer[key] = np.zeros((self.C, self.window_size))
                self.current_index[key] = 0
            
            self.ema[key] = self.alpha * data[i] + (1 - self.alpha) * self.ema[key]

            index = self.current_index[key]
            self.ema_buffer[key][:, index] = self.ema[key]
            self.actual_buffer[key][:, index] = data[i]
            self.current_index[key] = (index + 1) % self.window_size
    
    def get(self,key,last=False):
        if isinstance(key,np.ndarray):
            return np.array([self.get(k) for k in key])
        if key is None:
            return np.array([self.get(k) for k in self.ema.keys()])
        elif key not in self.ema_buffer:
            return np.array([])
        elif key in self.ema_buffer:
            return np.roll(self.ema_buffer[key], -self.current_index[key], axis=1)
        else:
            return np.array([])
 
class ChunkBuffer:
    def __init__(self, max_length, columns, initial_capacity=10, chunk_size=10):
        self.max_length = max_length
        self.columns = columns
        self.chunk_size = chunk_size
        self.capacity = initial_capacity
        self.uids = np.full(self.capacity, -1, dtype=int)  # 초기에는 모든 UID를 -1로 설정
        self.buffers = np.zeros((self.capacity, max_length)+ columns)
        self.current_lengths = np.zeros(self.capacity, dtype=int)
        self.empty_slots = np.arange(self.capacity, dtype=int)  # 초기 빈 슬롯 관리
        self.uid_to_index = {}
        self.state = False
        
    def is_full(self):
        return self.state

    def _expand_buffers(self):
        """배열을 Chunk 단위로 확장"""
        new_buffers = np.zeros((self.chunk_size, self.max_length) + self.columns)
        self.buffers = np.vstack([self.buffers, new_buffers])
        self.current_lengths = np.append(self.current_lengths, np.zeros(self.chunk_size, dtype=int))
        self.uids = np.append(self.uids, np.full(self.chunk_size, -1, dtype=int))
        new_slots = np.arange(self.capacity, self.capacity + self.chunk_size)
        self.empty_slots = np.append(self.empty_slots, new_slots)
        self.capacity += self.chunk_size

    def _get_index(self, uid):
        """UID의 인덱스를 가져오거나 새로 추가"""
        if uid in self.uid_to_index:
            return self.uid_to_index[uid]
        if self.empty_slots.size == 0:
            self._expand_buffers()
        index = self.empty_slots[-1]
        self.empty_slots = self.empty_slots[:-1]
        self.uid_to_index[uid] = index
        return index

    def add(self, array):
        """배열 데이터를 추가"""
        for row in array:
            uid = int(row[-1])
            index = self._get_index(uid)
            if self.current_lengths[index] < self.max_length:
                self.buffers[index, self.current_lengths[index]] = row
                self.current_lengths[index] += 1
                self.uids[index] = uid  # UID 저장
                if self.current_lengths[index] >= self.max_length:
                    self.state = True

    def retrieve_and_clear(self, uid=None):
        """
        UID에 해당하는 데이터를 반환하고 초기화.
        - uid: 특정 UID (int). None이면 가득 찬 UID들에 대한 데이터를 모두 반환.
        """
        retrieved_data = None  # 반환할 데이터를 저장
        full_uids = np.empty(0)
        if uid is not None:
            # 단일 UID 처리
            if uid not in self.uid_to_index:
                return np.empty(0)
            index = self.uid_to_index[uid]
            if self.current_lengths[index] == 0:
                return np.empty(0)
            retrieved_data = self.buffers[index,:self.current_lengths[index]].copy()
            self.current_lengths[index] = 0
            self.uids[index] = -1
            self.empty_slots = np.append(self.empty_slots, index)
            del self.uid_to_index[uid]
        else:
            # 가득 찬 UID 처리
            full_indices = np.where(self.current_lengths == self.max_length)[0]
            if full_indices.size == 0:
                return np.empty(0)

            full_uids = self.uids[full_indices].copy()

            retrieved_data = self.buffers[full_indices,:self.max_length].copy()
            self.uids[full_indices] = -1
            self.current_lengths[full_indices] = 0
            self.empty_slots = np.append(self.empty_slots, full_indices)
            for uid in full_uids:
                del self.uid_to_index[uid]

        # 상태 업데이트 및 반환
        self.state = not np.all(self.current_lengths < self.max_length)
        return retrieved_data