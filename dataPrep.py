# We divide the data into training and test:
def PrepData(shapes, labels, leng):
    trainS = shapes[0:int(0.8*leng)]
    trainL = labels[0:int(0.8*leng)]
    testS = shapes[int(0.8*leng):]
    testL = labels[int(0.8*leng):]
    train = [trainS, trainL]
    test = [testS, testL]
    return train, test
