import time

class SuperRegression():

    #THE _ACTIVE ATTRIBUTE IS USED TO SHOW WHETHER A CLASSIFIER IS SUSPENDED OR NOT
    def __init__(self):
        self._ACTIVE = True
        self._IS_READY = False
        self.NUMBER_OF_INSTANCES_OBSERVED = 0
        self._TRAINING_TIME = 0
        self._TESTING_TIME = 0
        self._TOTAL_TRAINING_TIME = 0
        self._TOTAL_TESTING_TIME = 0

    def is_ready(self):
        return self._IS_READY

    def set_ready(self):
        self._IS_READY = True

    def is_active(self):
        return self._ACTIVE

    def deactivate(self):
        self._ACTIVE = False

    def activate(self):
        self._ACTIVE = True

    def get_training_time(self):
        return self._TRAINING_TIME

    def get_testing_time(self):
        return self._TESTING_TIME

    def get_running_time(self):
        return self._TRAINING_TIME + self._TESTING_TIME

    def get_total_running_time(self):
        return self._TOTAL_TRAINING_TIME + self._TOTAL_TESTING_TIME

    def getError(self):
        raise NotImplementedError
    pass


    def _reset_stats(self):
        # HERE I NEED TO MAKE SOME MODIFICATIONS
        # FOR CONSIDERING CONCEPT DRIFTS.
        self.NUMBER_OF_INSTANCES_OBSERVED = 0
        self._TRAINING_TIME = 0
        self._TESTING_TIME = 0
        self._IS_READY = False
        self._ACTIVE = True

    def do_training(self, record):
        t1 = time.perf_counter()
        error = self.train(record)
        t2 = time.perf_counter()
        delta = (t2 - t1) * 1000  # in milliseconds
        self._TRAINING_TIME += delta
        self._TOTAL_TRAINING_TIME += delta
        return error

    def do_testing(self, record):
        t1 = time.perf_counter()
        error = self.test(record)
        t2 = time.perf_counter()
        delta = (t2 - t1) * 1000  # in milliseconds
        self._TESTING_TIME += delta
        self._TOTAL_TESTING_TIME += delta
        return error