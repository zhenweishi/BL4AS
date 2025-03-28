import gc
import os
import warnings

import torch
import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')
import wandb
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('lib/')

import argparse

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers
from easydict import EasyDict as edict


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('conf_file', type=str, help='path to config file')
    parser.add_argument('-p', '--parser', type=str, default='monai', choices=['monai', 'omega', 'auto'], help='config parser')
    parser.add_argument('--rank', type=int, default=0)

    args = parser.parse_args()
    
    args = get_conf(conf_file=args.conf_file, conf_parser=args.parser, rank=args.rank)
    if "imports" in args:
        del args["imports"]
    args = edict(**args)

    # set seed if required
    set_seed(args.get("seed", None))

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.get("ngpus_per_node", None) is None:
        ngpus_per_node = torch.cuda.device_count()
        print("device_count:", ngpus_per_node)
        args.ngpus_per_node = ngpus_per_node
    else:
        ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, 
                nprocs=ngpus_per_node, 
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    if args.rank == 0:
        if getattr(args, 'use_tensorboard', False):
            args.summary_writer = SummaryWriter(log_dir=args.log_dir)
        if getattr(args, 'use_wandb', False):        
            if args.wandb_id is None:
                args.wandb_id = wandb.util.generate_id()
                run = wandb.init(project=f"{args.proj_name}_{args.dataset}", 
                                name=args.run_name, 
                                config=vars(args),
                                id=args.wandb_id,
                                resume='allow',
                                dir=args.log_dir)
                
    nfolds = args.get("nfolds", 1)
    pretrains = args.get("pretrain")
    resumes = args.get("resume")

    nruns = nfolds
    if isinstance(pretrains, list) and pretrains is not None:
        nruns = max(len(pretrains), nruns)
    if isinstance(resumes, list) and resumes is not None:
        nruns = max(len(resumes), nruns)

    if isinstance(pretrains, str):
        pretrains = [pretrains] * nruns
    elif pretrains is None:
        pretrains = [None] * nruns
    
    if isinstance(resumes, str):
        resumes = [resumes] * nruns
    elif resumes is None:
        resumes = [None] * nruns

    if nfolds == 1:
        folds = range(nruns) #[0] * nruns
    else:
        folds = range(nfolds)

    # print(len(folds), len(pretrains), len(resumes))
    # print(nruns)
    print("len(folds):", len(folds))
    print("len(pretrains):", len(pretrains))
    print("len(resumes):", len(resumes))
    for i, (fold, pretrain, resume) in enumerate(zip(folds, pretrains, resumes)):
        # if i != 3:
        #     continue

        args.fold = fold

        # init trainer
        trainer_class = getattr(trainers, f'{args.trainer_name}', None)
        assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
        trainer = trainer_class(args)
        if args.rank == 0:
            print(f"*********************** {i} ***********************")
        trainer.build_model(pretrain)
        trainer.build_optimizer()
        trainer.build_dataloader(fold=fold)
        if args.resume:
            trainer.resume(resume)
        trainer.run()

        # trainer.model = None
        # trainer.wrapped_model = None
        # trainer.optimizer = None
        # trainer.dataloader = None
        gc.collect()
        torch.cuda.empty_cache()
        # break

        if args.get("one_fold_only", False):
            break


    if args.rank == 0:
        if getattr(args, 'use_tensorboard', False):
            args.summary_writer.close()
        if getattr(args, 'use_wandb', False):
            run.finish()

    if "__test_metrics__" in args:
        import pandas as pd
        from pathlib import Path
        df = pd.DataFrame(args["__test_metrics__"])
        df.insert(0, "Fold", range(len(df)))
        df.to_csv(Path(args.log_dir).parent.parent / "test_metrics.csv", index=False)
        print("------- in csv --------")
        print(df.to_csv(index=False))

if __name__ == '__main__':
    main()