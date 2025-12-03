# import time
# import torch
# import logging
# import datetime
# from collections import OrderedDict
# from contextlib import contextmanager
# from detectron2.utils.comm import is_main_process
# from .calibration_layer import PrototypicalCalibrationBlock


# class DatasetEvaluator:
#     """
#     Base class for a dataset evaluator.

#     The function :func:`inference_on_dataset` runs the model over
#     all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

#     This class will accumulate information of the inputs/outputs (by :meth:`process`),
#     and produce evaluation results in the end (by :meth:`evaluate`).
#     """

#     def reset(self):
#         """
#         Preparation for a new round of evaluation.
#         Should be called before starting a round of evaluation.
#         """
#         pass

#     def process(self, input, output):
#         """
#         Process an input/output pair.

#         Args:
#             input: the input that's used to call the model.
#             output: the return value of `model(output)`
#         """
#         pass

#     def evaluate(self):
#         """
#         Evaluate/summarize the performance, after processing all input/output pairs.

#         Returns:
#             dict:
#                 A new evaluator class can return a dict of arbitrary format
#                 as long as the user can process the results.
#                 In our train_net.py, we expect the following format:

#                 * key: the name of the task (e.g., bbox)
#                 * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
#         """
#         pass


# class DatasetEvaluators(DatasetEvaluator):
#     def __init__(self, evaluators):
#         assert len(evaluators)
#         super().__init__()
#         self._evaluators = evaluators

#     def reset(self):
#         for evaluator in self._evaluators:
#             evaluator.reset()

#     def process(self, input, output):
#         for evaluator in self._evaluators:
#             evaluator.process(input, output)

#     def evaluate(self):
#         results = OrderedDict()
#         for evaluator in self._evaluators:
#             result = evaluator.evaluate()
#             if is_main_process():
#                 for k, v in result.items():
#                     assert (
#                         k not in results
#                     ), "Different evaluators produce results with the same key {}".format(k)
#                     results[k] = v
#         return results


# def inference_on_dataset(model, data_loader, evaluator, cfg=None):

#     num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
#     logger = logging.getLogger(__name__)

#     pcb = None
#     if cfg.TEST.PCB_ENABLE:
#         logger.info("Start initializing PCB module, please wait a seconds...")
#         pcb = PrototypicalCalibrationBlock(cfg)
#         # pcb.visualize_prototypes_tsne(out_path="outputs/pcb_prototypes_tsne.png",
#         #                       perplexity=20,
#         #                       n_iter=1000,
#         #                       pca_components=50)
#         # print('file created successfully ')

#         # # visualize instance features + prototypes
#         # pcb.visualize_featurebank_tsne(out_path="outputs/pcb_featurebank_tsne.png",
#         #                             perplexity=50,
#         #                             n_iter=1000,
#         #                             pca_components=50,
#         #                             max_points=2000)                           
#         # similarities = pcb.compute_pairwise_prototype_similarity()
#     logger.info("Start inference on {} images".format(len(data_loader)))
#     total = len(data_loader)  # inference data loader must have a fixed length
#     evaluator.reset()

#     logging_interval = 50
#     num_warmup = min(5, logging_interval - 1, total - 1)
#     start_time = time.time()
#     total_compute_time = 0
#     with inference_context(model), torch.no_grad():
#         for idx, inputs in enumerate(data_loader):
#             if idx == num_warmup:
#                 start_time = time.time()
#                 total_compute_time = 0

#             start_compute_time = time.time()
#             outputs = model(inputs)
#             if cfg.TEST.PCB_ENABLE:
#                 outputs = pcb.execute_calibration(inputs, outputs)
#             torch.cuda.synchronize()
#             total_compute_time += time.time() - start_compute_time
#             evaluator.process(inputs, outputs)

#             if (idx + 1) % logging_interval == 0:
#                 duration = time.time() - start_time
#                 seconds_per_img = duration / (idx + 1 - num_warmup)
#                 eta = datetime.timedelta(
#                     seconds=int(seconds_per_img * (total - num_warmup) - duration)
#                 )
#                 logger.info(
#                     "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
#                         idx + 1, total, seconds_per_img, str(eta)
#                     )
#                 )

#     # Measure the time only for this worker (before the synchronization barrier)
#     total_time = int(time.time() - start_time)
#     total_time_str = str(datetime.timedelta(seconds=total_time))
#     # NOTE this format is parsed by grep
#     logger.info(
#         "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
#             total_time_str, total_time / (total - num_warmup), num_devices
#         )
#     )
#     total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
#     logger.info(
#         "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
#             total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
#         )
#     )

#     results = evaluator.evaluate()
#     # An evaluator may return None when not in main process.
#     # Replace it by an empty dict instead to make it easier for downstream code to handle
#     if results is None:
#         results = {}
#     return results


# @contextmanager
# def inference_context(model):
#     """
#     A context where the model is temporarily changed to eval mode,
#     and restored to previous mode afterwards.

#     Args:
#         model: a torch Module
#     """
#     training_mode = model.training
#     model.eval()
#     yield
#     model.train(training_mode)


import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)
        # pcb.visualize_prototypes_tsne(out_path="outputs/pcb_prototypes_tsne.png",
        #                              perplexity=20,
        #                              n_iter=1000,
        #                              pca_components=50)
        # print('file created successfully ')

        # # visualize instance features + prototypes
        # pcb.visualize_featurebank_tsne(out_path="outputs/pcb_featurebank_tsne.png",
        #                                perplexity=50,
        #                                n_iter=1000,
        #                                pca_components=50,
        #                                max_points=2000)                                     
        # similarities = pcb.compute_pairwise_prototype_similarity()
    
    # --- TSNE: Enable feature collection ---
    # We enable feature collection on the model instance.
    # We assume 'model' is an instance of GeneralizedRCNN which has 'collect_tsne_features' and 'tsne_feature_buffer'.
    model.tsne_feature_buffer.clear() # Clear any old data
    model.collect_tsne_features = True
    logger.info("TSNE feature collection enabled.")


    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # --- TSNE: Disable collection and run visualization ---
    model.collect_tsne_features = False
    
    # Run the visualization after inference is complete
    # We assume NUM_CLASSES is available in the config (e.g., COCO has 80 classes + 1 background)
    try:
        model.run_tsne_visualization(
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            save_path="outputs/tsne_visualization_post_inference.png"
        )
    except Exception as e:
        logger.error(f"Failed to run t-SNE visualization: {e}")


    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)