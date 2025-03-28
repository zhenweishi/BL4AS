import json
import math
import os
from pathlib import Path
import pdb
import time
import torch
import torchmetrics
from torchmetrics import AUROC, AveragePrecision, Accuracy
import timm.optim
import monai
import yaml
import gc
import copy

from lib.utils import calculate_metrics

from .base_trainer import BaseTrainer

__all__ = ['ClsTrainer']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def get_display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

def get_metric(metric_name, output, target, num_classes=2, average="macro", threshold=0.5):
    if num_classes > 2:
        preds = output.softmax(dim=1)
        try:
            metric = getattr(torchmetrics, metric_name)(task="multiclass", num_classes=num_classes, average=average, threshold=threshold)
        except:
            metric = getattr(torchmetrics, metric_name)(task="multiclass", num_classes=num_classes, average=average)
    elif num_classes == 2:
        preds = output.softmax(dim=1)[:, 1]
        try:
            metric = getattr(torchmetrics, metric_name)(task="binary", threshold=threshold)
        except:
            metric = getattr(torchmetrics, metric_name)(task="binary")
    else:
        preds = output.flatten()
        try:
            metric = getattr(torchmetrics, metric_name)(task="binary", threshold=threshold)
        except:
            metric = getattr(torchmetrics, metric_name)(task="binary")
    return metric(preds, target)

class ClsTrainer(BaseTrainer):
    r"""
    ViT 3D Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self, pretrain=None):
        args = self.args
        if pretrain is None:
            pretrain = args.pretrain

        if self.model_name != 'Unknown' and self.model is None:
            print(f"=> creating model {self.model_name}")
            # self.model = args.model_obj
            self.model = copy.deepcopy(args.model_obj)
            if args.get("no_PLFB", False):
                self.model.decoder.forward = self.model.decoder.forward_without_PLFB
                print("=> No PLFB")

            linear_keyword = 'head'
            
            # load pretrained weights
            if pretrain is not None and os.path.exists(pretrain):
                print(f"=> Start loading pretrained weights from {pretrain}")
                checkpoint = torch.load(pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder.') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k] 
                    if k.startswith('base_encoder.') and not k.startswith('base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("base_encoder."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k] 
                    if k.startswith('encoder.') and not k.startswith('encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k] 
                    if k == 'encoder_pos_embed':
                        pe = torch.zeros([1, 1, state_dict[k].size(-1)])
                        state_dict['pos_embed'] = torch.cat([pe, state_dict[k]], dim=1)
                        del state_dict[k]
                    if k == 'patch_embed.proj.weight' and \
                        state_dict['patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                        del state_dict['patch_embed.proj.weight']
                        del state_dict['patch_embed.proj.bias']
                    if k == 'pos_embed' and \
                        state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                        del state_dict[k]
                    if k in state_dict:
                        del state_dict[k]
                    
                msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                # import pickle
                # pickle.dump(self.model.encoder, open("/mnt/tmp/model_new.pkl", "wb"))
                # with open("/mnt/tmp/model_new.txt", "w") as f:
                #     f.write(str(self.model.encoder))
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {pretrain}")
            
            # self.model.encoder = torch.nn.Sequential(self.model.encoder, torch.nn.Linear(1000, args.num_classes))
            # freeze all layers but the last fc
            if args.get("encoder_freeze_all_except_fc", False):
                for name, param in self.model.encoder.named_parameters():
                    if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                        param.requires_grad = False
                    if args.get("encoder_train_patch_embed", False):
                        if name.startswith('patch_embed'):
                            param.requires_grad = True
                # init the fc layer
                getattr(self.model.encoder, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
                getattr(self.model.encoder, linear_keyword).bias.data.zero_() 
                
            self.loss_ce = torch.nn.CrossEntropyLoss().cuda(args.gpu)
            self.loss_bce = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()

        print(f"=> changing learning rate to match batch size, lr *= batch_size / {args.get('pretrain_batch_size', 128)}: ", self.lr)
        self.lr = args.lr * args.batch_size / args.get('pretrain_batch_size', 128)

        if args.optimizer == 'lars':
            self.optimizer = timm.optim.LARS(optim_params, lr=self.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params, lr=self.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(optim_params, lr=self.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(optim_params, lr=self.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer {args.optimizer}")
        
    def build_dataloader(self, fold=0):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            self.fold = fold
            if args.get("nfolds", 1) > 1:
                self.nfolds = args.get("nfolds", 1)
                nfolds_conf = args.nfolds_conf
                arg_name = nfolds_conf["arg_name"]
                to_parse = {}

                if not args.get("test_only", False):
                    args.train_dataset[arg_name] = nfolds_conf["train"][fold]
                    args.val_dataset[arg_name] = nfolds_conf["val"][fold]
                # pdb.set_trace()
                if args.get("test_dataset", None) is not None:
                    args.test_dataset[arg_name] = nfolds_conf["test"][fold]

                if not args.get("test_only", False):
                    to_parse["train_dataset_obj"] = args.train_dataset
                    to_parse["val_dataset_obj"] = args.val_dataset
                if args.get("test_dataset", None) is not None:
                    to_parse["test_dataset_obj"] = args.test_dataset

                conf_str = json.dumps(to_parse).replace("_target_x", "_target_")
                # parser = monai.bundle.ConfigParser(yaml.load(conf_str, Loader=yaml.FullLoader))
                parser = monai.bundle.ConfigParser(json.loads(conf_str))
                new_args = parser.get_parsed_content()

                train_dataset = new_args.get("train_dataset_obj", None)
                val_dataset = new_args.get("val_dataset_obj", None)
                test_dataset = new_args.get("test_dataset_obj", None)

            else:
                self.nfolds = 1
                train_dataset = args.train_dataset_obj
                val_dataset = args.val_dataset_obj
                test_dataset = args.get("test_dataset_obj", None)

            print(f"=> Train dataset length: {len(train_dataset) if train_dataset is not None else 0}")
            print(f"=> Val dataset length: {len(val_dataset) if val_dataset is not None else 0}")
            print(f"=> Test dataset length: {len(test_dataset) if test_dataset is not None else 0}")

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if train_dataset is not None else None
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if val_dataset is not None else None
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) if test_dataset is not None else None

            else:
                train_sampler = None
                val_sampler = None
                test_sampler = None

            if not args.get("test_only", False):
                self.dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.batch_size, 
                                                            shuffle=(train_sampler is None),
                                                            num_workers=self.workers, 
                                                            pin_memory=True, 
                                                            sampler=train_sampler, 
                                                            drop_last=True)
                self.iters_per_epoch = len(self.dataloader)

                self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                                batch_size=self.batch_size, 
                                                                shuffle=(val_sampler is None),
                                                                num_workers=self.workers, 
                                                                pin_memory=True, 
                                                                sampler=val_sampler, 
                                                                drop_last=False)
                self.val_iters_per_epoch = len(self.val_dataloader)
            else:
                self.iters_per_epoch = 0
                self.val_iters_per_epoch = 0
            if test_dataset is not None:
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                                    batch_size=self.batch_size, 
                                                                    shuffle=False,
                                                                    num_workers=self.workers, 
                                                                    pin_memory=True, 
                                                                    sampler=test_sampler,
                                                                    drop_last=False)
                self.test_iters_per_epoch = len(self.test_dataloader)

        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        
    def run(self):
        args = self.args
        fold = args.get("fold", 0)

        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = -torch.inf
        eval_freq = args.get("eval_freq", 0) # if eval_freq > 0, use return of `evaluate` as metric
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            if args.get("test_only", False):
                break

            # train for one epoch
            this_train_metric = self.epoch_train(epoch, niters)
            if eval_freq < 1: # no evaluation, then use train metric as metric
                this_metric = this_train_metric
                
            # evaluate on validation set
            if eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
                this_metric = self.evaluate(epoch, niters)

            # save checkpoint
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )
                if getattr(args, "save_best", False) and this_metric > best_metric:
                    best_metric = this_metric
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/best{"_fold" + str(fold) if self.nfolds > 1 else ""}.pth.tar'
                    )
                    if eval_freq < 1:
                        print("Saved best on train metric at epoch %d, AUC: %.3f" % (epoch, best_metric))
                    else:
                        print("Saved best on val metric at epoch %d, AUC: %.3f" % (epoch, best_metric))

        if getattr(self, "test_dataloader", None) is not None:
            self.test()
        else:
            print("Test dataloader is not created. Please create dataloader first.")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        loss_ce = self.loss_ce
        loss_bce = self.loss_bce

        # meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        auc = AverageMeter('AUC', ':6.2f')
        ap = AverageMeter('AP', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, auc],
            prefix="Epoch: [{}]".format(epoch))

        """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
        if args.get("encoder_eval_in_train", True):
            model.encoder.eval()
        else:
            model.encoder.train()
        model.decoder.train()

        end = time.time()
        iters_per_epoch = len(train_loader)
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            c2_c0_image = data["C2-C0_image"] # N1DHW
            c2_image = data["C2_image"] # N1DHW
            c5_c2_image = data["C5-C2_image"] # N1DHW
            mask = data["mask"]
            bm_label = data.get(args.get("bm_key", "is_malignant"), None)
            bg_bm_label = torch.zeros_like(bm_label)

            if args.gpu is not None:
                c2_c0_image = c2_c0_image.to(args.gpu, non_blocking=True)
                c2_image = c2_image.to(args.gpu, non_blocking=True)
                c5_c2_image = c5_c2_image.to(args.gpu, non_blocking=True)
                mask = mask.to(args.gpu, non_blocking=True)
                bm_label = bm_label.to(args.gpu, non_blocking=True)
                bg_bm_label = bg_bm_label.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                c025_image = torch.cat([c2_c0_image, c2_image, c5_c2_image], dim=1) # (N, 3, D, H, W)
                # c025_image = c025_image * mask
                _ = model(c025_image, mask)
                bm_logits = _.get("bm_logits", None)
                bi_logits = _.get("bi_logits", None)
                bg_bm_logits = _.get("bg_bm_logits", None)
                bg_bi_logits = _.get("bg_bi_logits", None)
                bm_logits_group = _.get("bm_logits_group", None)

                loss = .0
                if bm_logits is not None:
                    if args.num_classes == 1:
                        bm_loss = torch.nn.functional.binary_cross_entropy_with_logits(bm_logits.flatten(), bm_label.float())
                    else:
                        bm_loss = torch.nn.functional.cross_entropy(bm_logits, bm_label.long())
                    loss += bm_loss
                
                if bg_bm_logits is not None:
                    if args.get("bgbm_weight", 1.0) > 0:
                        bg_bm_loss = torch.nn.functional.binary_cross_entropy_with_logits(bg_bm_logits.flatten(), bg_bm_label.float())
                        loss += args.get("bgbm_weight", 1.0) * bg_bm_loss


                if bm_logits_group is not None:
                    for group in bm_logits_group:
                        if group is not None:
                            loss += torch.nn.functional.pairwise_distance(group, group)
                
                if loss.isnan().any():
                    print("loss nan detected")

            # compute gradient and do SGD step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = c2_c0_image.size(0)
            losses.update(loss.item(), batch_size)
            this_auc = get_metric("AUROC", bm_logits, bm_label, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
            this_ap = get_metric("AveragePrecision", bm_logits, bm_label, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
            auc.update(this_auc, batch_size)
            ap.update(this_ap, batch_size)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq > 0 and i % args.print_freq == 0:
                progress.display(i)
            
            if args.rank == 0:
                args.summary_writer.add_scalar("loss" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("auc" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', this_auc, epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("ap" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', this_ap, epoch * iters_per_epoch + i)
              
        print(' @ Train: Loss {losses.avg:.5f} AUC {auc.avg:.3f}'
            .format(losses=losses, auc=auc))
        return auc.avg

    def resume(self, resume_file=None):
        args = self.args
        if resume_file is None:
            resume_file = args.resume
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            if args.gpu is None:
                checkpoint = torch.load(resume_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(resume_file, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            msg = self.model.load_state_dict(checkpoint['state_dict'])
            if not args.get("test_only", False):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
            print(f'=> Loading messages: \n {msg}')
        else:
            if args.get("test_only", False):
                raise ValueError(f"=> No checkpoint found at '{resume_file}'")
            print("=> no checkpoint found at '{}'".format(resume_file))
        
    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        loss_ce = self.loss_ce
        loss_bce = self.loss_bce

        # meters
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        auc = AverageMeter('AUC', ':6.2f')
        ap = AverageMeter('AP', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, auc],
            prefix='Val: ')

        # switch to evaluate mode
        model.eval()

        # Collect all outputs and targets
        all_outputs = []
        all_targets = []

        end = time.time()
        iters_per_epoch = len(val_loader)
        for i, data in enumerate(val_loader):
            c2_c0_image = data["C2-C0_image"] # N1DHW
            c2_image = data["C2_image"] # N1DHW
            c5_c2_image = data["C5-C2_image"] # N1DHW
            mask = data["mask"]
            bm_label = data.get(args.get("bm_key", "is_malignant"), None)
            bg_bm_label = torch.zeros_like(bm_label)

            if args.gpu is not None:
                c2_c0_image = c2_c0_image.to(args.gpu, non_blocking=True)
                c2_image = c2_image.to(args.gpu, non_blocking=True)
                c5_c2_image = c5_c2_image.to(args.gpu, non_blocking=True)
                mask = mask.to(args.gpu, non_blocking=True)
                bm_label = bm_label.to(args.gpu, non_blocking=True)
                bg_bm_label = bg_bm_label.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                c025_image = torch.cat([c2_c0_image, c2_image, c5_c2_image], dim=1) # (N, 3, D, H, W)
                # c025_image = c025_image * mask
                _ = model(c025_image, mask)
                bm_logits = _.get("bm_logits", None)
                bi_logits = _.get("bi_logits", None)
                bg_bm_logits = _.get("bg_bm_logits", None)
                bg_bi_logits = _.get("bg_bi_logits", None)
                bm_logits_group = _.get("bm_logits_group", None)

                loss = .0
                if bm_logits is not None:
                    if args.num_classes == 1:
                        bm_loss = torch.nn.functional.binary_cross_entropy_with_logits(bm_logits.flatten(), bm_label.float())
                    else:
                        bm_loss = torch.nn.functional.cross_entropy(bm_logits, bm_label.long())
                    loss += bm_loss
                
                
                if bg_bm_logits is not None:
                    if args.get("bgbm_weight", 1.0) > 0:
                        bg_bm_loss = torch.nn.functional.binary_cross_entropy_with_logits(bg_bm_logits.flatten(), bg_bm_label.float())
                        loss += args.get("bgbm_weight", 1.0) * bg_bm_loss


                if bm_logits_group is not None:
                    for group in bm_logits_group:
                        if group is not None:
                            loss += torch.nn.functional.pairwise_distance(group, group)
                
                if loss.isnan().any():
                    print("loss nan detected")

                all_outputs.append(bm_logits)
                all_targets.append(bm_label)
            
            batch_size = c2_c0_image.size(0)
            losses.update(loss.item(), batch_size)
            this_auc = get_metric("AUROC", bm_logits, bm_label, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
            this_ap = get_metric("AveragePrecision", bm_logits, bm_label, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
            auc.update(this_auc, batch_size)
            ap.update(this_ap, batch_size)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.val_print_freq > 0 and i % args.val_print_freq == 0:
                progress.display(i)
            
            if args.rank == 0:
                args.summary_writer.add_scalar("val_loss" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("val_auc" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', this_auc, epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("val_ap" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', this_ap, epoch * iters_per_epoch + i)

        # print(' * Val: AUC {auc.avg:.3f}'
        #     .format(auc=auc))
        # return auc.avg
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        _o = all_outputs.detach().cpu().float()
        _t = all_targets.detach().cpu().int()
        real_auc = get_metric("AUROC", _o, _t, num_classes=args.num_classes)
        print(' * Val: AUC {auc:.3f}'
            .format(auc=real_auc))
        return real_auc
        
    @torch.no_grad()
    def test(self):
        args = self.args
        model = self.wrapped_model
        test_loader = self.test_dataloader

        # meters
        batch_time = AverageMeter('Time', ':6.3f')
        metrics = ["AUROC", "AveragePrecision", "Accuracy", "Precision", "Recall", "Specificity", "F1Score"]
        metrics_progress = [AverageMeter(metric, f':6.2f') for metric in metrics]
        # auc = AverageMeter('AUC', ':6.2f')
        # ap = AverageMeter('AP', ':6.2f')

        progress = ProgressMeter(
            len(test_loader),
            [batch_time, ] + metrics_progress,
            # [batch_time, auc, ap],
            prefix='Test: ')

        # Collect all outputs and targets
        all_outputs = []
        all_targets = []
        all_filename = []
        all_center = []

        # switch to evaluate mode
        model.eval()
        end = time.time()
        for i, data in enumerate(test_loader):
            c2_c0_image = data["C2-C0_image"] # N1DHW
            c2_image = data["C2_image"] # N1DHW
            c5_c2_image = data["C5-C2_image"] # N1DHW
            mask = data["mask"]
            bm_label = data.get(args.get("bm_key", "is_malignant"), None)
            bg_bm_label = torch.zeros_like(bm_label)

            if args.gpu is not None:
                c2_c0_image = c2_c0_image.to(args.gpu, non_blocking=True)
                c2_image = c2_image.to(args.gpu, non_blocking=True)
                c5_c2_image = c5_c2_image.to(args.gpu, non_blocking=True)
                mask = mask.to(args.gpu, non_blocking=True)
                bm_label = bm_label.to(args.gpu, non_blocking=True)
                bg_bm_label = bg_bm_label.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                c025_image = torch.cat([c2_c0_image, c2_image, c5_c2_image], dim=1) # (N, 3, D, H, W)
                # c025_image = c025_image * mask
                _ = model(c025_image, mask)
                bm_logits = _.get("bm_logits", None)
                bi_logits = _.get("bi_logits", None)
                bg_bm_logits = _.get("bg_bm_logits", None)
                bg_bi_logits = _.get("bg_bi_logits", None)
                bm_logits_group = _.get("bm_logits_group", None)

            all_outputs.append(bm_logits)
            all_targets.append(bm_label)
            all_filename.extend(data.get("filename", None))
            all_center.extend(data.get("center", "Unknown"))

            batch_size = c2_c0_image.size(0)
            for metric, mprog in zip(metrics, metrics_progress):
                _o = bm_logits.detach().cpu().float()
                _t = bm_label.detach().cpu().int()
                # torch.save(_o, "/mnt/tmp/output.pkl")
                # torch.save(_t, "/mnt/tmp/target.pkl")
                this_metric = get_metric(metric, _o, _t, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
                mprog.update(this_metric, batch_size)

            # this_auc = AUROC(task="multiclass", num_classes=args.num_classes)(output, target)
            # this_ap = AveragePrecision(task="multiclass", num_classes=args.num_classes)(output, target)
            # auc.update(this_auc, batch_size)
            # ap.update(this_ap, batch_size)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.test_print_freq > 0 and i % args.test_print_freq == 0:
                progress.display(i)

            # Additional logging or operations can be done here

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        import pickle
        pkl_dir = Path(args.ckpt_dir).parent / "pkl"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(all_outputs, open(pkl_dir / f"output_{self.fold}.pkl", "wb"))
        pickle.dump(all_targets, open(pkl_dir / f"target_{self.fold}.pkl", "wb"))
        pickle.dump(all_filename, open(pkl_dir / f"filename_{self.fold}.pkl", "wb"))
        pickle.dump(all_center, open(pkl_dir / f"center_{self.fold}.pkl", "wb"))
        # exit()

        # Compute overall AUC and AP
        # total_auc = AUROC(task="multiclass", num_classes=args.num_classes)(all_outputs, all_targets)
        # total_ap = AveragePrecision(task="multiclass", num_classes=args.num_classes)(all_outputs, all_targets)
        # print(' * Test: AUC {:.3f} AP {:.3f}'.format(total_auc, total_ap))
        # if args.rank == 0:
        #     args.summary_writer.add_text("test_auc" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', str(total_auc))
        #     args.summary_writer.add_text("test_ap" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', str(total_ap))
        
        total_metrics = {}
        for metric in metrics:
            _o = all_outputs.detach().cpu().float()
            _t = all_targets.detach().cpu().int()
            total_metrics[metric] = get_metric(metric, _o, _t, num_classes=args.num_classes, threshold=args.get("threshold", 0.5))
            print(f' * Test: {metric} {total_metrics[metric]:.3f}')
            if args.rank == 0:
                args.summary_writer.add_text(f"test_{metric}" + f'{"_fold" + str(args.fold) if self.nfolds > 1 else ""}', str(total_metrics[metric]))

        if "__test_metrics__" not in args:
            args["__test_metrics__"] = []
        args["__test_metrics__"].append(calculate_metrics(all_outputs.reshape(-1).cpu().numpy(), all_targets.reshape(-1).cpu().numpy()))


        # Additional logging or operations can be done here
