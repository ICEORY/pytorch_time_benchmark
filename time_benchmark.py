# reference https://discuss.pytorch.org/t/measuring-gpu-tensor-operation-speed/2513/4
import torch
import timeit
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
def main():
    x = torch.cuda.FloatTensor(10000, 500).normal_()
    w = torch.cuda.FloatTensor(200, 500).normal_()

    # ensure that context initialization and normal_() operations
    # finish before you start measuring time

    torch.cuda.synchronize()
    torch.cuda.synchronize()

    for i in range(10):
        a = timeit.default_timer()
        y = x.mm(w.t())
        torch.cuda.synchronize() # wait for mm to finish
        b = timeit.default_timer()
        print "loop #%d, batch GPU %f s"%(i, b-a)

if __name__ == '__main__':
    main()
    