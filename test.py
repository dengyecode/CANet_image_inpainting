from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice
import time

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    for i, data in enumerate(dataset):
        #start = time.time()

        model.set_input(data)
        model.test()
        #stop = time.time()
        #print(str(stop - start) + "s")