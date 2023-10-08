import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
import resource
# Function to set memory limit based on system memory
def set_memory_limit():
    # Get total system memory in bytes
    total_memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    
    # Calculate the memory limit (95% of total system memory)
    memory_limit_bytes = int(0.95 * total_memory_bytes)
    
    # Set the memory limit using resource module
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


if __name__ == '__main__': 
    # If memory leak is observed
    # set_memory_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--validation_slope_print', default=False, action='store_true',help='If set, prints the slope of the validation loss.')
    opt = parser.parse_args()
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=1.0, maxMemory=1.0, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    # Debugging options
    # Set CUDA_LAUNCH_BLOCKING to 1 in debug mode 
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Set TORCH_CUDA_USE_DSA to 1 for using CUDA Dynamic Shared Memory
    # os.environ["TORCH_CUDA_USE_DSA"] = "1"
    from training.idr_train import IDRTrainRunner
    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=gpu,
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 scan_id=opt.scan_id,
                                 train_cameras=opt.train_cameras,
                                 validation_slope_print=opt.validation_slope_print)

    trainrunner.run()
