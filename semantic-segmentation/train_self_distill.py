import argparse
import os
import datetime
import logging
import time
import math
import numpy as np

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix+'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def train(cfg, local_rank, distributed):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    if local_rank==0:
        print(feature_extractor)
        print(classifier)

    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    iteration = 0
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        # if "optimizer_fea" in checkpoint:
        #     logger.info("Loading optimizer_fea from {}".format(cfg.resume))
        #     optimizer.load(checkpoint['optimizer_fea'])
        # if "optimizer_cls" in checkpoint:
        #     logger.info("Loading optimizer_cls from {}".format(cfg.resume))
        #     optimizer.load(checkpoint['optimizer_cls'])
        # if "iteration" in checkpoint:
        #     iteration = checkpoint['iteration']
    
    src_train_data = build_dataset(cfg, mode='train', is_source=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        src_train_data, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    start_training_time = time.time()
    end = time.time()

    for i, (src_input, src_label, _) in enumerate(train_loader):
        data_time = time.time() - end
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr*10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        size = src_label.shape[-2:]
        pred = classifier(feature_extractor(src_input), size)
        loss = criterion(pred, src_label)
        loss.backward()

        optimizer_fea.step()
        optimizer_cls.step()
        meters.update(loss_seg=loss.item())
        iteration+=1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
    
        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == max_iters) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict()}, filename)
        
        if iteration == max_iters:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iters)
        )
    )

    return feature_extractor, classifier

def run_test(cfg, model, local_rank, distributed):
    logger = logging.getLogger("FADA.tester")
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    feature_extractor, classifier = model
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]
            pred = classifier(feature_extractor(x))
            pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
            
            output = pred.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        RANK = int(os.environ["RANK"])
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            NGPUS_PER_NODE = torch.cuda.device_count()
        assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        GPU = RANK % NGPUS_PER_NODE
        torch.cuda.set_device(GPU)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             rank=RANK, world_size=WORLD_SIZE)
        NUM_GPUS = WORLD_SIZE
        print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
