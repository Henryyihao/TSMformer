import os
import torch
from model import iTransformer, VARformer, FLformer, TimeMixer, Crossformer, DLinear, PatchTST, FEDformer, SegRNN, Informer, TiDE, MTSMixer, MLinear, MLinear2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'VARformer': VARformer,
            'FLformer': FLformer,
            'TimeMixer': TimeMixer,
            'Crossformer': Crossformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'FEDformer': FEDformer,
            'SegRNN': SegRNN,
            'Informer': Informer,
            'TiDE': TiDE,
            'MTSMixer': MTSMixer,
            'MLinear': MLinear,
            'MLinear2': MLinear2,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            elif torch.backends.mps.is_available():
                os.environ["MPS_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('mps:{}'.format(self.args.gpu))
                print('Use GPU: mps:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass