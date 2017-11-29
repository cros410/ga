import random
import numpy as np

l = [11,10,5]
print reduce(lambda x, y: x + y, l) / len(l)
