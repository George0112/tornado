import random
from math import *
import math

from streams.generators.tools.transition_functions import Transition

class EXPO:
    def __init__(self, concept_length=2000, transition_length=50, noise_rate=0.1, random_seed=1):
        self.__INSTANCES_NUM = 5 * concept_length
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = 4
        self.__W = transition_length
        self.__RECORDS = []

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__INSTANCES_NUM), int(self.__INSTANCES_NUM * noise_rate))

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")
            
    @staticmethod
    def get_class_name():
        return EXPO.__name__

    def generate(self, output_path="EXPO"):
        random.seed(self.__RANDOM_SEED)

        # 1. CREATE RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            dist_id = int(concept_sec % 2) # if drift occurs
            record = self.create_record(dist_id)
            self.__RECORDS.append(list(record))

        # 2. ADD NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, dist_id):
        x, y = self.create_attribute_value()
        if dist_id == 1:
            y = 2*y+100
        return x, y

    @staticmethod
    def create_attribute_value():
        x = random.uniform(0, 10)
        y = math.pow(math.e, x) * random.uniform(0.9, 1.1)
        return x,y

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            y = self.__RECORDS[noise_spot][1]
            self.__RECORDS[noise_spot][1] = y*random.random()

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation EXPO" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n\n" )
        arff_writer.write("@data" + "\n")
        for i in range(0, len(self.__RECORDS)):
            arff_writer.write(str("%0.3f" % self.__RECORDS[i][0]) + "," +
                              str("%0.3f" % self.__RECORDS[i][1]) + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")