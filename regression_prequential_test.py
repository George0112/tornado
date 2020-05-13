from classifier.__init__ import *
from drift_detection.__init__ import *
from filters.project_creator import Project
from streams.readers.arff_reader import ARFFReader
from tasks.__init__ import *
from regression.__init__ import *


# 1. Creating a project
project = Project("projects/single", "expo")

# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read("data_streams/_synthetic/EXPO/EXPO.arff")
# attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
# learner = NaiveBayes(labels, attributes_scheme['nominal'])
learner = Expo()

# 4. Initializing a drift detector
detector = FHDDMS(n=100)
# actual_drift_points = [20000, 40000, 60000, 80000]
actual_drift_points = [500, 1000, 1500, 2000]
drift_acceptance_interval = 100

# 5. Creating a Prequential Evaluation Process
prequential = PrequentialRegressionDriftEvaluator(learner, detector,  actual_drift_points, drift_acceptance_interval, project)

prequential.run(stream_records, 1)
