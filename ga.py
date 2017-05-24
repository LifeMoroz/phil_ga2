import random

from GA2.nn import NN
from NN1.image_loader import ImageLoader


P_SIZE = 10
SHAPE = (18, 18)


def size():
    return SHAPE[0] * SHAPE[1]


_arr = [NN() for _ in range(P_SIZE)]

image_loader = ImageLoader(SHAPE)
training_set = image_loader.load_image_set('../NN1/training_set', reshape_to_vector=True, np_format=False)


def cross(mother, father) -> NN:
    part_size = random.randint(1, size() - 1)
    order = mother, father # if random.randint(0, 1) else father, mother
    n = NN()
    for i in range(part_size):
        new_ = order[0].hidden_weights[i]
        n.hidden_weights[i] = new_

    for i in range(part_size, size()):
        n.hidden_weights[i] = order[1].hidden_weights[i]

    part_size = random.randint(1, len(n.output_weights) - 1)
    for i in range(part_size):
        n.output_weights[i] = order[0].output_weights[i]

    for i in range(part_size, len(n.output_weights)):
        n.output_weights[i] = order[1].output_weights[i]
    return n


def mutate(self):
    if random.random() < 0.2:
        new = NN()
        for i in range(size()):
            if random.random() < 0.3:
                new.hidden_weights[i] = self.hidden_weights[i]
        return new


def iterate():
    for i in range(len(_arr)):
        mutated = mutate(_arr[i])
        if mutated:
            _arr.append(mutated)

    ssize = len(_arr)
    for i in range(ssize):
        for j in range(ssize):
            if i != j:
                _arr.append(cross(_arr[i], _arr[j]))
    res = []
    print(len(_arr))
    for n in _arr:
        res.append((n.execute(training_set), n))
    res.sort(key=lambda x: x[0])
    return [r[1] for r in res], res[0][0]


ITERS = 10


for _ in range(ITERS):
    _arr, _error = iterate()
    _arr = _arr[:P_SIZE]
    _arr.extend([NN(), NN()])
    print(_error)
