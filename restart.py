from iter_classification import TrainingIter

if __name__ == '__main__':
    t_iter = TrainingIter()
    _logdir = t_iter.get_load_weight(yd="20210327", hms="111131", epoch=3)
    t_iter.restart(_logdir)
