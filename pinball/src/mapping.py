import numpy as np


class Mapper:
    def __init__(self, low, high, k):
        self.low = low
        self.span = (high - low) / k
        self.k = k

    def perform(self, pos):
        idxy = (pos - self.low) // self.span
        assert (idxy <= self.k).all()
        return int(idxy[0] * self.k + idxy[1])


if __name__ == "__main__":
    low = np.array([-1., -1.])
    high = np.array([1., 1.])
    mapper = Mapper(low, high, k=4)
    print(mapper.perform([0, 0]))
    print(mapper.perform([-1, -1]))
    print(mapper.perform([0.99, 0.99]))
    print(mapper.perform([0.89, 0]))
    print(mapper.perform([0, 0.99]))