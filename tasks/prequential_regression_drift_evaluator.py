import copy
import random

import numpy as numpy
from pympler import asizeof

from archiver.archiver import Archiver
from evaluators.classifier_evaluator import PredictionEvaluator
from evaluators.detector_evaluator import DriftDetectionEvaluator
from plotter.performance_plotter import *
from filters.attribute_handlers import *
from streams.readers.arff_reader import *

import pylab
from matplotlib import pyplot as plt

class PrequentialRegressionDriftEvaluator:
    def __init__(self, learner, drift_dector, actual_drift_points, drift_acceptance_interval, project, memory_check_stop=-1):
        self.learner = learner
        self.drift_detector = drift_dector

        self.__instance_counter = 0
        
        self.__learner_error_rate_array = []
        self.__learner_memory_usage = []
        self.__learner_runtime = []

        self.__actual_drift_points = actual_drift_points
        self.__drift_acceptance_interval = drift_acceptance_interval

        self.__located_drift_points = []
        self.__drift_points_boolean = []
        self.__drift_detection_memory_usage = []
        self.__drift_detection_runtime = []

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()

    def run(self, stream, random_seed=1):
        random.seed(random_seed)
        stream_length = len(stream)
        x = []
        y = []
        for s in stream:
            x.append(float(s[0]))
            y.append(float(s[1]))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.plot(x, y, 'bo', markersize=2)
        plt.savefig('plot.png', dpi=150)

        for record in stream:
            self.__instance_counter += 1

            percentage = (self.__instance_counter / stream_length) * 100
            print("%0.2f" % percentage + "% of instances are prequentially processed!", end="\r")

            r = copy.copy(record)
            # for k in range(len(r) -1):
            #     r = Normalizer.normalize(r)

            # Prequential Learning
            if self.learner.is_ready():
                real_prediction = float(r[-1])
                prediction = self.learner.do_testing(r)

                prediction_status = True
                if prediction > real_prediction*1.1 or prediction < real_prediction*0.9:
                    prediction_status = False

                # Drift Detected?
                warning_status, drift_status = self.drift_detector.detect(prediction_status)
                if drift_status:
                    self.__drift_points_boolean.append(1)
                    self.__located_drift_points.append(self.__instance_counter)
                
                    print("\n ->>> " + self.learner.LEARNER_NAME.title() + " faced a drift at instance " + str(self.__instance_counter) + ".")
                    print("%0.2f" % percentage, " of instances are prequentially processed!", end="\r")

                    self.__learner_memory_usage.append(asizeof.asizeof(self.learner, limit=20))
                    self.__learner_runtime.append(self.learner.get_running_time())

                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))
                    self.__drift_detection_runtime.append(self.drift_detector.RUNTIME)

                    self.learner.reset()
                    self.drift_detector.reset()

                    continue
                learner_error_rate = self.learner.do_training(r)
            else:
                learner_error_rate = self.learner.do_training(r)
                if learner_error_rate == -1:
                    continue
            # learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE, self.learner.get_confusion_matrix())
            
            learner_error_rate = round(learner_error_rate, 4)
            self.__learner_error_rate_array.append(learner_error_rate)

            # if self.__memory_check_step != -1:
            #     if self.__instance_counter % self.__memory_check_step == 0:
            #         self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))

            self.__drift_points_boolean.append(0)

        print("\n" + "The stream is completely processed.")
        # self.__store_stats()
        self.__plot()
        print("\n\r" + "THE END!")
        print("\a")

    def __plot(self):
        learner_name = TornadoDic.get_short_names(self.learner.LEARNER_NAME)
        detector_name = self.drift_detector.DETECTOR_NAME
        detector_setting = self.drift_detector.get_settings()
        file_name = learner_name + "_" + detector_name + "." + detector_setting[0]

        up_range = numpy.max(self.__learner_error_rate_array)
        up_range = 1 if up_range > 0.75 else round(up_range, 1) + 0.25

        pair_name = learner_name + ' + ' + detector_name + "(" + detector_setting[1] + ")"
        
        Plotter.plot_single(pair_name, self.__learner_error_rate_array, "Error-rate", self.__project_name, self.__project_path, file_name, [0, up_range], 'upper right', 200)
        
        Archiver.archive_single(pair_name, self.__learner_error_rate_array, self.__project_path, self.__project_name, 'Error-rate')
        
        Plotter.plot_single_ddm_points(pair_name, self.__drift_points_boolean, self.__project_name, self.__project_path, file_name)

    @staticmethod
    def normalize(record):
        for i in range(len(record)):
            pass
