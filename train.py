from src.iter_classification import *

if __name__ == '__main__':
    sets_ = parser()
    t_iter = TrainingIter(sets_)
    t_iter.run()
