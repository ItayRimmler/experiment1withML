# The experiment is using sequential API.

from imageCreation import CreateDataForML, cv2, np, tf
from dataPrep import PrepData
from trainModel import LetsGo

# Setting the length of our dataset (number of images):
leng = 100

# Creating the data-label pairs:
shapes, labels = CreateDataForML(leng)

# Preparing it further:
train, test = PrepData(shapes, labels, leng)

# Doing what we came to do:
LetsGo(train, test, leng)

# Th-th-th-th-that's all folks!
