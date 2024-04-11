import time

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from utils.others.utils import step_learning_rate, cosine_learning_rate, translate_state_dict
from utils.others.loss_record import print_loss
from core.evaluate import *

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

class Trainer():

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 criterion,
                 loss_record,
                 writer=None,
                 checkpoints_path='',
                 exp_mode='unseen_subject',
                 curr_fold=0,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 len_val_set=0,
                 val_segments=None,
                 len_test_set=0,
                 test_segments=None,
                 test_save_path=None,
                 device='cuda'):

        self.args = args
        self.epoch = 0
        self.epochs = args.epochs
        self.mode = exp_mode
        self.curr_fold = curr_fold
        self.loss_record = loss_record

        self.len_train_loader = 0

        self.train_loader = train_loader
        if self.train_loader is not None:
            self.len_train_loader = len(self.train_loader)

        self.val_loader = val_loader

        if self.mode == 'unseen_group':
            self.test_loader = test_loader

        self.batch_iter = 0
        self.total_batch = self.len_train_loader * self.epochs

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)

        self.writer = writer
        self.device = device

        self.min_val_loss = 1e5
        self.min_loss_dict = {}
        self.checkpoints_path = checkpoints_path
        self.test_save_path = test_save_path

        self.scaler = GradScaler()

        self.val_segments = val_segments
        self.test_segments = test_segments

        self.val_results_record = np.zeros((len_val_set, 6))
        self.val_joints_record = np.zeros((len_val_set, 25, 3))
        if self.mode == 'unseen_group':
            self.test_results_record = np.zeros((len_test_set, 6))
            self.test_joints_record = np.zeros((len_test_set, 25, 3))

        self.test_time_list = []
    def fit(self):

        self.batch_iter = 0
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.train()
            val_loss = self.val('eval')
            if val_loss['mpjpe_wo_align'] < self.min_val_loss:
                self.min_val_loss = val_loss['mpjpe_wo_align']
                val_accel = val_loss['accel']
                self.min_loss_dict = val_loss
                state_dict = translate_state_dict(self.model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(
                    state_dict,
                    self.checkpoints_path + '/' 'hps_' + f'{epoch}_losses_{np.round(self.min_val_loss, 2)}_{np.round(val_accel, 2)}' + '.pth'
                )

            if epoch % 5 == 0:
                # if self.mode == 'unseen_group':
                #     self.val('test')
                self.loss_record.plot(epoch)

        print_loss(0, loss_dict=self.min_loss_dict)


    def train(self):

        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):

            batch_start = time.time()

            batch = {k: v.type(torch.float32).detach().cuda() for k, v in batch.items()}

            if self.args.cosine:
                lr = cosine_learning_rate(
                    self.args, self.epoch, self.batch_iter, self.optimizer, self.len_train_loader
                )
            else:
                lr = step_learning_rate(
                    self.args, self.epoch, self.batch_iter, self.optimizer, self.len_train_loader
                )

            outputs = self.model(batch['images'])[0]

            # import pdb;pdb.set_trace()

            losses, loss_dict = self.criterion(outputs, batch['gt_keypoints_3d'], batch['gt_keypoints_2d'],
                                    batch['betas'], batch['pose'], batch['trans'], batch['pressure_binary'])

            # import pdb;pdb.set_trace()

            # if loss_dict['kp2d'] > 100 and loss_dict['t_mpjpe'] < 0.1:
            #     import pdb;
            #     pdb.set_trace()

            self.optimizer.zero_grad()
            losses.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            # self.optimizer.zero_grad()
            #
            # self.scaler.scale(losses).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            batch_time = time.time() - batch_start

            self.batch_iter += 1

            print(
                "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] {} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    batch_idx + 1,
                    self.len_train_loader,
                    self.batch_iter,
                    self.total_batch,
                    print_loss(losses.item(), loss_dict),
                    lr,
                    batch_time
                ))

            self.writer.info(
                "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] {} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    batch_idx + 1,
                    self.len_train_loader,
                    self.batch_iter,
                    self.total_batch,
                    print_loss(losses.item(), loss_dict),
                    lr,
                    batch_time
                ))

            self.loss_record.update(losses.item(), loss_dict, 'train')

    def val(self, type='eval'):
        self.model.eval()

        if type == 'eval':
            loader = self.val_loader
            descriptor ='Validating'
        else:
            loader = self.test_loader
            descriptor = 'Testing'

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):

                batch = {k: v.type(torch.float32).detach().cuda() for k, v in batch.items()}

                outputs = self.model(batch['images'])[0]

                # import pdb;pdb.set_trace()

                # losses, loss_dict = self.criterion(outputs, batch['gt_keypoints_3d'], batch['gt_keypoints_2d'],
                #                                    batch['betas'], batch['pose'], batch['trans'], batch['pressure_binary'])

                # self.writer.info(
                #     "[{}] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] {}".format(
                #         descriptor,
                #         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #         self.epoch,
                #         self.epochs,
                #         batch_idx + 1,
                #         len(loader),
                #         print_loss(losses.item(), loss_dict),
                #     ))

                self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], outputs['kp_3d'],
                                    batch['verts'], outputs['verts'], type)

                # self.loss_record.update(losses.item(), loss_dict, type)

            self.accel_evaluate(type)

        return self.print_metric(type)
        # return {
        #     'mpjpe_wo_align': 0,
        #     'accel': 0,
        # }

    def test(self):

        self.model.eval()

        if self.mode == 'unseen_subject':
            loader = self.val_loader
            mode = 'eval'
        else:
            loader = self.test_loader
            mode = 'test'

        length = loader.dataset.get_data_len()

        results = {
            'theta': np.zeros((length, 85)),
            # 'verts': np.zeros((length, 6890, 3)),
            'kp_3d': np.zeros((length, 25, 3))
        }

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader)):
                batch = {k: v.type(torch.float32).detach().to(self.device) for k, v in batch.items()}

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                outputs = self.model(batch['images'])[0]

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                self.test_time_list.append(elapsed)

                # losses, loss_dict = self.criterion(outputs, batch['gt_keypoints_3d'], batch['gt_keypoints_2d'],
                #                                    batch['betas'], batch['pose'], batch['trans'], batch['pressure_binary'])

                # self.writer.info(
                #     "[{}] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] {}".format(
                #         descriptor,
                #         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #         self.epoch,
                #         self.epochs,
                #         batch_idx + 1,
                #         len(loader),
                #         print_loss(losses.item(), loss_dict),
                #     ))

                self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], outputs['kp_3d'],
                                    batch['verts'], outputs['verts'], mode)

                index_list = batch['curr_frame_idx'].cpu().reshape(-1).type(torch.long).numpy()
                reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
                for key in results:
                    results[key][index_list] = reduce(outputs[key].cpu().detach().numpy())

                # self.loss_record.update(losses.item(), loss_dict, type)

            self.accel_evaluate(mode)
            if mode == 'eval':
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segmemts': loader.dataset.db_segmemts,
                    'mpjpe_wo_align': self.val_results_record[:, 0],
                    'mpjpe': self.val_results_record[:, 1],
                    'mpjpe_pa': self.val_results_record[:, 2],
                    'mpve': self.val_results_record[:, 3],
                    'accel': self.val_results_record[:, 4] * 1000,
                    'accel_error': self.val_results_record[:, 5]
                })
                print(np.mean(self.val_results_record[:, 0]))
            else:
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segmemts': loader.dataset.db_segmemts,
                    'mpjpe_wo_align': self.test_results_record[:, 0],
                    'mpjpe': self.test_results_record[:, 1],
                    'mpjpe_pa': self.test_results_record[:, 2],
                    'mpve': self.test_results_record[:, 3],
                    'accel': self.test_results_record[:, 4] * 1000,
                    'accel_error': self.test_results_record[:, 5]
                })

                print(np.mean(self.test_results_record[:, 0]))
            np.savez(self.test_save_path + '/test.npz', **results)
            self.print_metric(mode)

        self.print_test_fps()


    def accel_evaluate(self, mode):

        if mode == 'eval':
            segments = self.val_segments
            joints = self.val_joints_record
        else:
            segments = self.test_segments
            joints = self.test_joints_record

        accel = []
        accel_error = []

        # import pdb;pdb.set_trace()

        for segment in segments:
            accel.extend(compute_accel(joints[segment[0]:segment[1]]).tolist())
            # accel_error.extend(compute_error_accel(joints[segment[0]:segment[1]]).tolist())

        if mode == 'eval':
            self.val_results_record[:len(accel), 4] = accel
            # self.val_results_record[:len(accel_error), 5] = torch.tensor(accel_error)
        else:
            self.test_results_record[:len(accel), 4] = accel
            # self.test_results_record[:len(accel_error), 5] = torch.tensor(accel_error)

    def batch_evaluate(self, index, target_j3ds, pred_j3ds, gt_verts, pred_verts, mode):

        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

        # import pdb;
        # pdb.set_trace()

        gt_verts = reduce(gt_verts)
        target_j3ds = reduce(target_j3ds)
        pred_verts = reduce(pred_verts)
        pred_j3ds = reduce(pred_j3ds)
        index = index.reshape(-1).type(torch.long).cpu()

        errors_wo_align = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
        target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1)
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1)

        m2mm = 1000

        pve = compute_error_verts(target_verts=gt_verts, pred_verts=pred_verts) * m2mm
        mpjpe = errors * m2mm
        pa_mpjpe = errors_pa * m2mm

        # import pdb;pdb.set_trace()

        if mode == 'eval':
            self.val_results_record[index, :4] = torch.stack([
                errors_wo_align, mpjpe, pa_mpjpe, pve
            ], dim=1).detach().cpu().numpy()
            self.val_joints_record[index] = pred_j3ds.detach().cpu().numpy()
        else:
            self.test_results_record[index, :4] = torch.stack([
                errors_wo_align, mpjpe, pa_mpjpe, pve
            ], dim=1).detach().cpu().numpy()
            self.test_joints_record[index] = pred_j3ds.detach().cpu().numpy()

    def print_metric(self, mode):
        if mode == 'eval':
            segments = len(self.val_segments)


            loss_dict = {
                'mpjpe_wo_align': np.mean(self.val_results_record[:, 0]),
                'mpjpe': np.mean(self.val_results_record[:, 1]),
                'mpjpe_pa': np.mean(self.val_results_record[:, 2]),
                'mpve': np.mean(self.val_results_record[:, 3]),
                'accel': np.mean(self.val_results_record[:-segments, 4]) * 1000,
                'accel_error': np.mean(self.val_results_record[:-segments * 2, 5])
            }

            print(
                "[Validating] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

            self.writer.info(
                "[Validating] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))
        else:
            segments = len(self.test_segments)

            loss_dict = {
                'mpjpe_wo_align': np.mean(self.test_results_record[:, 0]),
                'mpjpe': np.mean(self.test_results_record[:, 1]),
                'mpjpe_pa': np.mean(self.test_results_record[:, 2]),
                'mpve': np.mean(self.test_results_record[:, 3]),
                'accel': np.mean(self.test_results_record[:-segments, 4]) * 1000,
                'accel_error': np.mean(self.test_results_record[:- segments * 2, 5])
            }

            print(
                "[Testing] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

            self.writer.info(
                "[Testing] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

        self.loss_record.update(None, loss_dict, mode)

        return loss_dict

    def print_test_fps(self):
        fps = (len(self.test_time_list) * 16) / sum(self.test_time_list)
        print(
            f'Done image [{len(self.test_time_list) * 16}], '
            f'fps: {fps:.1f} img / s, '
            f'times per image: {1000 / fps:.1f} ms / img',
            flush=True)



