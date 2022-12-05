import torch
from classify_dvsg import *
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import argparse
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.lib.dl import netx
from lava.magma.core.decorator import implements, requires
import typing as ty
import numpy as np


class TestInput(AbstractProcess):
    """read data from the testset and converts it.
    """
    def __init__(self,
                 num_images: ty.Optional[int] = 288):
        super().__init__()
        shape = (64, 64, 2)
        self.s_out = OutPort(shape=shape)  # Input spikes to the classifier
        self.label_out = OutPort(shape=(1,))  # Ground truth labels to OutputProc
        self.num_images = Var(shape=(1,), init=num_images)
        self.ground_truth_label = Var(shape=(1,))
        self.data = Var(shape=(64, 64, 64, 2))

@implements(proc=TestInput, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    num_images: int = LavaPyType(int, int, precision=32)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=32)
    ground_truth_label: int = LavaPyType(int, int, precision=32)
    data: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.testSet = DVS128Gesture(root='./DVSGnet/datasets/DVSGesture', train=False, data_type='frame', frames_number=64, split_by='number')
        print('load data succeed')
        self.curr_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % 64 == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        frame = np.array([self.testSet[self.curr_img_id][0]]) #[N, T, C, H, W]
        frame = frame.transpose(1,0,2,3,4)# [N, T, C, H, W] -> [T, N, C, H, W]
        frame = encoder(torch.tensor(frame), 64)# [T, N, C, H, W] -> [T, N, C, size, size]
        self.data = frame.permute(1, 0, 4, 3, 2)[0] # [T, N, C, sizeH, sizeW] -> [T, sizeW, sizeH, C] important

        self.ground_truth_label = self.testSet[self.curr_img_id][1]
        self.label_out.send(np.array([self.ground_truth_label]))
        self.curr_img_id += 1

    def run_spk(self) -> None:
        self.s_out.send(np.array(self.data[(self.time_step - 1) % 64]))

class OutputProcess(AbstractProcess):
    def __init__(self):
        super().__init__()
        shape = (11,)
        self.num_images = Var(shape=(1,), init=288)
        self.spikes_in = InPort(shape=shape)
        self.label_in = InPort(shape=(1,))
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_image = Var(shape=(1,), init=64)
        self.pred_labels = Var(shape=(288,))
        self.gt_labels = Var(shape=(288,))

@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    gt_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
        
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % 64 == 0 and self.time_step > 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        gt_label = self.label_in.recv()
        pred_label = np.argmax(self.spikes_accum)
        # print(self.spikes_accum)
        self.gt_labels[self.current_img_id] = gt_label
        self.pred_labels[self.current_img_id] = pred_label
        self.current_img_id += 1
        self.spikes_accum = np.zeros_like(self.spikes_accum)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in


def main():

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=64, type=int, help='simulating time-steps')
    parser.add_argument('-b', default=1, type=int, help='batch size')
    parser.add_argument('-channels', default=32, type=int, help='channels of CSNN')
    parser.add_argument('-model', type=str, default='DVSGNet')
    parser.add_argument('-size', default=64, type=int, help='input size')
    parser.add_argument('-ds', default=3, type=int, help='down sample number')
    args = parser.parse_args()

    net = DVSGNet(size=args.size, channels=args.channels, ds=args.ds)
    checkpoint = torch.load('./DVSGnet/size64_ds_3_T64_64_b16_e128_adam_lr0.005_c32_amp/checkpoint_max.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net_ladl = net.to_lava()
    with torch.no_grad():
        print(net_ladl(torch.rand([args.b, 2, args.size, args.size, args.T])).shape)

    '''
    testSet = DVS128Gesture(root='./DVSGnet/datasets/DVSGesture', train=False, data_type='frame', frames_number=64, split_by='number')
    print('load data succeed')
    test_data_loader = torch.utils.data.DataLoader(
        dataset=testSet,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
        # TODO: pin_memory=True makes my GPU out of memory, change it back before run it on loihi
    )
    test_samples = 0
    test_acc = 0
    ground_truth=list()
    predictions=list()
    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = encoder(frame, args.size) 
            frame = frame.permute(1,0,2,3,4) # [N,T,C,H,W]
            out_fr = net(frame[0]).mean(0).argmax(0)
            test_samples += label.numel()
            test_acc += (out_fr == label).float().sum().item()
            # functional.reset_net(net)
            ground_truth.append(int(label))
            predictions.append(int(out_fr))
            if test_samples==-1:
                break
        test_acc /= test_samples
        print(f"\nGround truth of net_ladl: {ground_truth}\n"
            f"Predictions of net: {predictions}\n"
            f"Test Accuracy of net: {test_acc}")
    exit()
    '''

    export_hdf5(net_ladl, './DVSGnet/net_lava_dl.net')
    net_lava = netx.hdf5.Network(net_config='./DVSGnet/net_lava_dl.net', input_shape=(args.size, args.size, 2))
    
    with torch.no_grad():
        test_input = TestInput()
        test_output = OutputProcess()
        test_input.s_out.connect(net_lava.inp)
        net_lava.out.connect(test_output.spikes_in)
        test_input.label_out.connect(test_output.label_in)
        print(net_lava.out.shape)
        run_condition = RunSteps(num_steps=args.T)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')
        
        for img_id in range(288):
            print(f"\rCurrent image: {img_id+1}", end="\n")
            test_input.run(condition=run_condition, run_cfg=run_config)        
            
        ground_truth = test_output.gt_labels.get().astype(np.int32)
        predictions = test_output.pred_labels.get().astype(np.int32)
        test_input.stop()

        accuracy = np.sum(ground_truth==predictions)/ground_truth.size

        print(f"\nGround truth: {ground_truth}\n"
            f"Predictions : {predictions}\n"
            f"Accuracy    : {accuracy}")


if __name__ == '__main__':
    
    main()
