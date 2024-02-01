from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)



def calculateStats(array, print_output=False, dataName=None, perceptronCount=None):
    arrayMax = max(array)
    arrayMin = min(array)
    mean = sum(array) / len(array)
    squaredDifferences = [(x - mean)**2 for x in array]
    meanOfSquaredDifferences = sum(squaredDifferences) / len(squaredDifferences)
    standardDeviation = sqrt(meanOfSquaredDifferences)

    if print_output:
        print("******************************************")
        if dataName:
            print("This data is from", dataName + ".")
        if perceptronCount:
            print("PerceptronCount:", perceptronCount)
        print(f"Accuracy for iterations 1-5: {array}")
        print(f"\tThe maximum is {arrayMax:.2f}")
        print(f"\tThe minimum is {arrayMin:.2f}")
        print(f"\tThe mean is {mean:.2f}")
        print(f"\tThe standard deviation is {standardDeviation:.2f}")
        print("******************************************") 

    return [arrayMax, arrayMin, mean, standardDeviation]


def question5():

    testCarArray = []
    for i in range(5):
        output = testCarData()
        # print("The accuracy for iteration", i + 1, "was", output[1])
        testCarArray.append(output[1])

    testPenArray = []
    for i in range(5):
        output = testPenData()
        # print("The accuracy for iteration", i + 1, "was", output[1])
        testPenArray.append(output[1])

    calculateStats(testPenArray, True, "testPenData")
    calculateStats(testCarArray, True, "testCarArray")



def question6(testData, hiddenLayerRange=range(0, 41, 5), numTrials=5):
    if not callable(testData):
        raise TypeError("The first argument must be a method")

    results = []
    
    for hiddenLayerSize in hiddenLayerRange:
        trialAccuracies = []
        for _ in range(numTrials):
            _, accuracy = testData(hiddenLayers=[hiddenLayerSize])
            trialAccuracies.append(accuracy)

        maxAcc = max(trialAccuracies)
        avgAcc = average(trialAccuracies)
        stdAcc = stDeviation(trialAccuracies)
        
        results.append({
            'hidden_layer_size': hiddenLayerSize,
            'max_accuracy': maxAcc,
            'average_accuracy': avgAcc,
            'std_deviation': stdAcc
        })
    
    # Print or return the results
    for result in results:
        print(f"Hidden Layer Size: {result['hidden_layer_size']}")
        print(f"\tMax Accuracy: {result['max_accuracy']:.4f}")
        print(f"\tAverage Accuracy: {result['average_accuracy']:.4f}")
        print(f"\tStandard Deviation: {result['std_deviation']:.4f}")
    
    return results

question6(testPenData)
question6(testCarData)


