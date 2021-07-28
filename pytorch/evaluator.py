from utils import registry
from torch.utils import data
import numpy as np
import torch
from utils.metrics import kp_dict
from pdb import set_trace as st
class Evaluator:
    def __call__(self, model, dataset):
        '''
        Evaluate the model using the dataset provided.

        :param model: the model to be evaluated.
        :param dataset: the dataset to be evaluated on.
        :return: a summary dictionary including:
            1. 'scalar': a dictionary of scalar values, such as errors.
            2. 'histogram': a dictionary of histograms, such as distributions.
            3. 'image': a dictionary of images, such as pixelwise error map.
        '''
        raise NotImplementedError


@registry.register('Evaluator', 'ViewpointEstimation')
class ViewpointEstimationEvaluator(Evaluator):
    def __init__(self, gpu, num_classes=12):
        self.gpu = gpu
        self.results_dict = kp_dict(num_classes)

    def __call__(self, model, dataset):
        assert not model.training, 'Model must not be in training mode!'
        if self.gpu:
            model = model.cuda()
        data_loader = data.DataLoader(dataset, batch_size=128, num_workers=0)
        cur_idx = 0
        results_dict = self.results_dict
        # for inputs, target in data_loader:
        for img, azim_label, elev_label, tilt_label, obj_cls, uid in data_loader:
            with torch.no_grad():
                if self.gpu:
                    img = img.cuda()
                    azim_label = azim_label.cuda()
                    elev_label = elev_label.cuda()
                    tilt_label = tilt_label.cuda()

                azim, elev, tilt = model(img)
                # pred = model.prediction_from_output(output)
                # azim, elev, tilt = model.prediction_from_output(output)

                results_dict.update_dict( uid,[azim.data.cpu().numpy(), elev.data.cpu().numpy(), tilt.data.cpu().numpy()], [azim_label.data.cpu().numpy(), elev_label.data.cpu().numpy(), tilt_label.data.cpu().numpy()])                # TODO: Compute the errors here
            print(img.size())
            print(cur_idx)
            cur_idx += 1
        type_accuracy, type_total, type_geo_dist = results_dict.metrics()

        geo_dist_median = [np.median(type_dist) * 180. / np.pi for type_dist in type_geo_dist if type_dist != [] ]
        type_accuracy   = [ type_accuracy[i] * 100. for i in range(0, len(type_accuracy)) if  type_total[i] > 0]
        w_acc           = np.mean(type_accuracy)


        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("Type Acc_pi/6 : ", type_accuracy, " -> ", w_acc, " %")
        print ("Type Median   : ", [ int(1000 * a_type_med) / 1000. for a_type_med in geo_dist_median ], " -> ", int(1000 * np.mean(geo_dist_median)) / 1000., " degrees")
        # print ("Type Loss     : ", [epoch_loss_a/total_step, epoch_loss_e/total_step, epoch_loss_t/total_step], " -> ", (epoch_loss_a + epoch_loss_e + epoch_loss_t ) / total_step)
        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return {
            'scalar': {'MeanAngleError': 0.3},
            'histogram': {'SizeDistribution': [0, 1, 3, 2, 1]},
            'image': {}
        }
