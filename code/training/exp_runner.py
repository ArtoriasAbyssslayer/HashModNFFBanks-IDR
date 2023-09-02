import sys
sys.path.append('../code')
import argparse
import GPUtil



if __name__ == '__main__':
    
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
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = 0
    else:
        gpu = opt.gpu
    import os
    # Set CUDA_LAUNCH_BLOCKING to 1 to allocate full GPU memory
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
