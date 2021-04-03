from src.iter_classification import *

if __name__ == '__main__':
    sets_ = parser()
    t_iter = TrainingIter(sets_)
    t_iter.restart()
    
    #### When you dont want to set command line arguments, 
    #### following member method get_load_weight() provide to load model 
    # _logdir = t_iter.get_load_weight(yd="20210402", hms="125654", epoch=2)
    # t_iter.restart()

