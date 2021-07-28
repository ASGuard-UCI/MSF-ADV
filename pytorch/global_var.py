import yaml
# import tensorboardX
from utils import Dotable
import functools


# -----------------------------------
# Global parameters

param = Dotable()


def param_load_yaml(fname):
    with open(fname, 'r') as f:
        param_dict = yaml.load(f)

    param.clear()
    param.update(param_dict)


# -----------------------------------
# The global summary writer.
# class _SummaryWriter:
#     def __init__(self):
#         self.writer = None


#     def init(self, folder):
#         self.writer = tensorboardX.SummaryWriter(folder)

#     def __getattr__(self, attr):
#         if self.writer is None:
#             raise RuntimeError('Call summary_writer.init() first')

#         return getattr(self.writer, attr)

#     def flush(self):
#         if self.writer is None:
#             raise RuntimeError('Call summary_writer.init() first')

#         self.writer.file_writer.flush()


# summary_writer = _SummaryWriter()
