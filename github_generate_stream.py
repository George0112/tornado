"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import os

from streams.generators.__init__ import *

stream_name = "SINE1"

project_path = "data_streams/_synthetic/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)

file_path = project_path + stream_name

stream_generator = EXPO(concept_length=500)
stream_generator.generate(file_path)
