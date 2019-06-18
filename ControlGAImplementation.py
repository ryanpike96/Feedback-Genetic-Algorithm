# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from matplotlib import pyplot as plt
import time
import random
import copy

# initialize inputs
population_size = 200
features = 30

# read in data
data = pd.read_csv('wdbc.data')
data = np.array(data)

# remove id
data = data[:, 1:32]

# replace Ms at start of data with 1s at end and Bs with 0s
for row in range(0, data.shape[0]):
    tmp = data[row]
    if tmp[0] =='M': # M=1, B=0
        tmp = np.append(tmp, 1)
    else:
        tmp = np.append(tmp, 0)
    tmp = tmp[1:32]
    data[row] = tmp

# normalize
data = data / data.max(axis=0)


# define network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, activationFunction):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.af = activationFunction
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.af(out)
        out = self.fc2(out)
        out = self.af(out)
        return out


def cross_train(encoding, runs):
    # input solution encoding and number of runs
    # return average accuracy of solution using 4-fold cross validation for the number of runs given

    total_test = 0
    correct_test = 0
    for n in range(0, runs):

        # cross validation
        for x in range(0, 4): # 4):
            # split data into training set (426) and testing set (142)
            msk = np.ones((data.shape[0]), dtype=bool)
            batchSize = 142
            msk[x*batchSize:x*batchSize+batchSize] = False
            train_data = data[msk]
            test_data = data[~msk]

            # remove features specified corresponding to 0s in 'encoding.featuresUsed'
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
            train_data = np.delete(train_data,np.s_[get_indexes(0, encoding.featuresUsed)], axis=1)
            test_data = np.delete(test_data, np.s_[get_indexes(0, encoding.featuresUsed)], axis=1)

            # split training data into input and target
            train_data = train_data.astype(np.float32)
            test_data = test_data.astype(np.float32)
            train_input = torch.tensor(train_data[:, :int(np.sum(encoding.featuresUsed))]).float()
            train_target = torch.tensor(train_data[:, int(np.sum(encoding.featuresUsed))]).long()

            # split training data into input and target
            test_input = torch.tensor(test_data[:, :int(np.sum(encoding.featuresUsed))]).float()
            test_target = torch.tensor(test_data[:, int(np.sum(encoding.featuresUsed))]).long()

            # initialize nn, loss function and optimizer
            net = Net(np.sum(encoding.featuresUsed), encoding.layerSizes, encoding.activationFunction)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=encoding.learningRate, momentum=encoding.momentum)

            # train
            for epoch in range(0, encoding.epochs):

                # forward + backward + update
                outputs = net(train_input)
                loss = criterion(outputs, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test the neural network using testing data and get prediction
            test_output = net(test_input)
            _, predicted_test = torch.max(test_output, 1)

            # calculate accuracy
            total_test = predicted_test.size(0) + total_test
            correct_test = np.sum(predicted_test.data.numpy() == test_target.data.numpy()) + correct_test

    # print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
    return correct_test / total_test


class MyEncoding():
    # stores an individual solution
    def __init__(self, identity, lr, epochs, momentum, featuresUsed, layerSizes, activationFunction):
        self.identity = identity
        self.learningRate = lr
        self.epochs = epochs
        self.momentum = momentum
        self.featuresUsed = featuresUsed
        self.layerSizes = layerSizes
        self.activationFunction = activationFunction
        self.accuracy = 0

    def mutate(self, id):
        # randomly select a type of mutation then return an instance of MyEncoding with only the mutated parameter
        # different from the current instance. Mutation types are : learning rate, none, number of epochs,
        # number of hidden neurons in a layer, features used, momentum, activation functions.
        # consider implementing: alter number of layers, add feature ftx/fty, loss function
        mutationType = np.random.randint(6, size=1)
        d = {
            0: self.mutateLearningRate(id),
            1: self.mutateEverything(id),
            2: self.mutateEpochs(id),
            3: self.mutateLayerSizes(id),
            4: self.mutateMomentum(id),
            5: self.mutateActivationFunction(id)
        }
        return d[mutationType[0]]

    def mutateLearningRate(self, id):
        # change learning rate by small factor 0.5-2
        factor = 2 ** random.uniform(-1.0, 1.0)
        new_lr = self.learningRate * factor
        return MyEncoding(id, new_lr, self.epochs, self.momentum, self.featuresUsed,
                          self.layerSizes, self.activationFunction)

    def mutateEverything(self, id):
        # re-initialize everything

        # activationFunction options
        activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                                 nn.Tanhshrink(), nn.Hardtanh()]
        return MyEncoding(id, 10**(-1*np.random.uniform(1, 6)), int(np.random.uniform(1, 500)), np.random.uniform(0, 0.99),
                          self.featuresUsed, 2+np.random.randint(79, size=(1, 1))[0][0], random.choice(activationFunctionList))

    def mutateEpochs(self, id):
        # change number of epochs +/- 1
        new_epochs = int(np.random.randint(3, size=1)-1 + self.epochs)
        if new_epochs < 1:
            new_epochs = 1
        elif new_epochs > 100:
            new_epochs = 100
        return MyEncoding(id, self.learningRate, new_epochs, self.momentum, self.featuresUsed,
                          self.layerSizes, self.activationFunction)

    def mutateLayerSizes(self, id):
        # change number of hidden neurons +/- 1
        # assuming there's one hidden layer for now
        new_layerSizes = np.random.randint(3, size=1) - 1 + self.layerSizes
        new_layerSizes = new_layerSizes[0]

        # ensure bounds are not exceeded
        if new_layerSizes < 2:
            new_layerSizes = 2
        elif new_layerSizes > 100:
            new_layerSizes = 100

        return MyEncoding(id, self.learningRate, self.epochs, self.momentum, self.featuresUsed,
                          new_layerSizes, self.activationFunction)

    def mutateFeaturesUsed(self, n):
        # change features used, add or remove n
        new_featuresUsed = self.featuresUsed
        ind = random.sample(range(0, 30), n)[0]
        new_featuresUsed[ind] = (new_featuresUsed[ind]+1)%2

        return new_featuresUsed

    def mutateMomentum(self, id):
        # change learning rate by small factor 1/sqrt(2)-sqrt(2)
        factor = 2 ** random.uniform(-.5, .5)
        new_momentum = self.momentum * factor
        return MyEncoding(id, self.learningRate, self.epochs, new_momentum, self.featuresUsed,
                          self.layerSizes, self.activationFunction)

    def mutateActivationFunction(self, id):
        # randomly re-select an activation function
        activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                                  nn.Tanhshrink(), nn.Hardtanh()]
        new_activationFunction = random.choice(activationFunctionList)
        return MyEncoding(id, self.learningRate, self.epochs, self.momentum, self.featuresUsed,
                          self.layerSizes, new_activationFunction)

def featurePlots (populationDict, i):
    # make and save plots for success of, success without and occurances of features throughout the population
    featureArray = np.zeros([1, 30])[0]
    missingFeatureArray = np.zeros([1, 30])[0]
    numFeatureArray = np.zeros([1, 30])[0]
    # check accuracy for encodings which contain each feature
    for n in range(0, 30):
        sum1 = 0
        sum2 = 0

        number1 = 0
        number2 = 0

        for enc in range(0, population_size):
            if (populationDict[enc].featuresUsed[n] == 1):
                number1 = number1 + 1
                sum1 = sum1 + populationDict[enc].accuracy
            else:
                number2 = number2 + 1
                sum2 = sum2 + populationDict[enc].accuracy

        if (number1 > 0):
            featureArray[n] = sum1 / number1 * 100
        if (number2 > 0):
            missingFeatureArray[n] = sum2 / number2 * 100
        numFeatureArray[n] = number1

    # make and save plot of the success of features
    x_vals = np.arange(30) + 1
    fig = plt.figure(i)
    ax = plt.axes(xlim=(0, 31), ylim=(0, 100))
    ax.bar(x_vals, featureArray, color='b')
    ax.set_ylabel('Average Accuracy (%)')
    str1 = 'Feature Success with Population size of ' + \
           str(population_size) + ' after ' + str(i) + ' Generations'
    ax.set_title(str1)
    outputFile = str1
    fig.savefig(outputFile)
    plt.close()

    # make and save plot of the success without features
    x_vals = np.arange(30) + 1
    fig = plt.figure(i + 1)
    ax = plt.axes(xlim=(0, 31), ylim=(0, 100))
    ax.bar(x_vals, missingFeatureArray, color='b')
    ax.set_ylabel('Average Accuracy (%)')
    str1 = 'Success without Feature with Population size of ' + \
           str(population_size) + ' after ' + str(i) + ' Generations'
    ax.set_title(str1)
    outputFile = str1
    fig.savefig(outputFile)
    plt.close()

    # make and save plot of the occurances of features
    x_vals = np.arange(30) + 1
    fig = plt.figure(i + 2)
    ax = plt.axes(xlim=(0, 31), ylim=(0, 1 + max(numFeatureArray)))
    ax.bar(x_vals, numFeatureArray, color='b')
    ax.set_ylabel('Number of Occurrences')
    str1 = 'Feature Occurrences with Population size of ' + \
           str(population_size) + ' after ' + str(i) + ' Generations'
    ax.set_title(str1)
    # plt.show()
    outputFile = str1
    fig.savefig(outputFile)
    plt.close()


def initializePopulation(population_size, runs, featuresUsed):
    # create initial population and store in dictionary for competition
    populationDict = {}
    for i in range(0, population_size):
        # initialize learning rate - range [10^-1, 10^-6]
        learning_rate = 10**(-1*np.random.uniform(1, 6))

        # initialize epochs - range [1, 100]
        epochs = int(np.random.uniform(1, 100))

        # initialize momentum - range [0, 0.99]
        momentum = np.random.uniform(0, 0.99)

        # initialize layerSizes - range [2, 80]
        layerSizes = 2+np.random.randint(79, size=(1, 1))[0][0]

        # initialize activationFunction
        activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                                 nn.Tanhshrink(), nn.Hardtanh()]
        activationFunction = random.choice(activationFunctionList)

        # initialize encoding of solution and find accuracy
        populationDict[i] = MyEncoding(i, learning_rate,epochs, momentum, featuresUsed, layerSizes,
                                    activationFunction)
        populationDict[i].accuracy = cross_train(populationDict[i], runs)

    return populationDict


def selection(populationDict, w):
    # given population dictionary return indices selected
    # to continue to next generation based on roulette wheel approach
    # include best solution if elitism is enabled

    newDict = copy.deepcopy(populationDict)
    # fitness = w*accuracy + (1-w)*features not used
    sumFit = sum([w * enc.accuracy + (1 - w) * (1 - np.sum(enc.featuresUsed) / 30) for enc in populationDict.values()])
    # max = sum([c.fitness for c in population])
    selectionProb = [(w * enc.accuracy + (1 - w) * (1 - np.sum(enc.featuresUsed) / 30)) / sumFit for enc in populationDict.values()]
    # return population[npr.choice(len(population), p=selection_probs)]
    i = 0
    for enc in populationDict.values():
        choice = np.random.choice(len(populationDict.values()), 1, p=selectionProb)[0]
        newDict[enc.identity] = copy.deepcopy(populationDict)[choice]
        newDict[enc.identity].identity = i
        i += 1

    return newDict

def repopulate(populationDict, mutations, runs):
    # given population dictionary, liveList and mutations return a population dictionary for next generation where
    # individuals which are not in the liveList are replaced by new individuals. The new individual's features used
    # is created with crossover at a randomly selected point, using two randomly selected parents from the liveList.
    # The resulting features used is then has some number of it's entries, based on mutations input, XORed with 1

    for encoding in populationDict.values():

        # determine parents and cross over point
        parents = random.sample(list(range(population_size)), 2)
        crossoverP = random.randint(1, 28)

        # create new individual's features used
        encoding.featuresUsed = list(populationDict[parents[0]].featuresUsed[:crossoverP]) + \
                                     list(populationDict[parents[1]].featuresUsed[crossoverP:])
        encoding.featuresUsed = np.asarray(encoding.featuresUsed)

        # add mutations
        encoding.featuresUsed = encoding.mutateFeaturesUsed(mutations)

        # determine fitness
        encoding.accuracy = cross_train(encoding, runs)

    return populationDict

def displayBest(populationDict):
    # find the most accurate solution and display it
    mostAccurate = 0
    maxAccuracy = 0
    for j in range(0, population_size):
        if populationDict[j].accuracy > maxAccuracy:
            maxAccuracy = populationDict[j].accuracy
            mostAccurate = j
    print("Most Accurate: ", populationDict[mostAccurate].accuracy*100)
    attrs = vars(populationDict[mostAccurate])
    print(', '.join("%s: %s" % item for item in attrs.items()))
    return mostAccurate

def featureGA(populationDict, generations, maxTime, runs, w):
    # run feature genetic algorithm, inputs dictionary containing population max numer of generations, maximum run time
    # and w value which specifies the weight which the fitness function places on the accuracy compared to the number
    # of features used

    # specify number of mutations
    mutations = 2
    startTime = time.time()

    # iterate through generations
    for i in range(generations):
        print("Feature GA Generation: ", i)
        # select parents of next generation
        selectedPop = selection(populationDict, w)
        # create next generation
        populationDict = repopulate(selectedPop, mutations, runs)


        # plot feature info
        featurePlots(populationDict, i)
        currentTime = time.time()
        if (currentTime - startTime) > maxTime:
            break
        _ = displayBest(populationDict)

    return populationDict


def hyperparameterGA(populationDict, fights, maxTime, runs):
    # run hyperparameter genetic algorithm, inputs dictionary containing population max number of fights, maximum
    # runtime and number of runs which fitness funciton averages over

    startTime = time.time()
    # each fight represents picking 2 solutions, remove weaker one and replace with a mutation of stronger one
    for i in range(0, fights):
        currentTime = time.time()
        if (currentTime - startTime) > maxTime:
            break

        # select fighters
        fighters = random.sample(range(0, population_size), 2)

        # compare fitness, where fitness = accuracy
        if populationDict[fighters[0]].accuracy > populationDict[fighters[1]].accuracy:
            parent = fighters[0]
            child = fighters[1]
        else:
            parent = fighters[1]
            child = fighters[0]

        # replace weaker solution with mutation of stronger solution
        populationDict[child] = populationDict[parent].mutate(child)
        populationDict[child].accuracy = cross_train(populationDict[child], runs)

        # occasionally display accuracy reached and best solution parameters
        if i % 100 == 0:
            print("HyperparameterGA: ", i, " fights")
            mostAccurate = displayBest(populationDict)

            # re-evaluate most accurate to prevent fluke repeatedly killing competitors
            populationDict[mostAccurate].accuracy = cross_train(populationDict[mostAccurate], runs*2)

    return populationDict

def setHyperparameters(populationDict, w):
    # set all individuals hyperparameters equal to those of the most accurate solution, randomize features used with
    # probability any features is used equal to w

    # find most accurate
    mostAccurate = 0
    maxAccuracy = 0
    for j in range(0, population_size):
        if populationDict[j].accuracy > maxAccuracy:
            maxAccuracy = populationDict[j].accuracy
            mostAccurate = j

    # iterate through population setting values
    for enc in populationDict.values():
        enc.learningRate = populationDict[mostAccurate].learningRate
        enc.epochs = populationDict[mostAccurate].epochs
        enc.momentum = populationDict[mostAccurate].momentum
        enc.layerSizes = populationDict[mostAccurate].layerSizes
        enc.activationFunction = populationDict[mostAccurate].activationFunction
        enc.featuresUsed = np.random.choice([0, 1], size=(30,), p=[(1-w), w])

    return populationDict

def setFeatures(populationDict):
    # find best solution and reinitialize the population so that every individual has the same features used as those
    # in the best solution
    bestFeatures = 0
    maxAccuracy = 0
    for j in range(0, population_size):
        if populationDict[j].accuracy > maxAccuracy:
            maxAccuracy = populationDict[j].accuracy
            bestFeatures = j
    bestFeatures = populationDict[bestFeatures].featuresUsed

    populationDict = initializePopulation(populationSize, runs, bestFeatures)

    return populationDict


# specify number of runs, generations and fights aswell as maximum time each GA will run for, number of iterations and
# w value
runs = 2
generations = 100000000000000
fights = 100000000000000
hyperParameterMaxTime = 60*60/2 # in seconds
featureGAMaxTime = 60*60/2 # in seconds
populationSize = population_size
iterations = 2
w = 0.5

# initialize features used (all)
featuresUsed = np.ones([1, 30])
featuresUsed = featuresUsed[0]
featuresUsed = featuresUsed.astype(int)

start = time.time()
# initialize initial population
populationDict = initializePopulation(populationSize, runs, featuresUsed)

# iterate throuh hyperparameterGA, setHyperparameters, featureGA and setFeatures to find optimal hyperparameters while
# minimizing the features used
for i in range(iterations):
    print("\nIteration: ", i)

    populationDict = hyperparameterGA(populationDict, fights, hyperParameterMaxTime, runs)
    populationDict = setHyperparameters(populationDict, w)

    populationDict = featureGA(populationDict, generations, featureGAMaxTime, runs, w)
    if i == (iterations-1):
        continue
    populationDict = setFeatures(populationDict)
    featureGAMaxTime *= 2
    hyperParameterMaxTime *= 2

# find and display all solutions in final population and finally specify the best solution
best = 0
bestSol = 0
for solution in populationDict.values():
    attrs = vars(solution)
    print(', '.join("%s: %s" % item for item in attrs.items()))
    if best < (w * solution.accuracy + (1 - w) * (1 - np.sum(solution.featuresUsed) / 30)):
        best = (w * solution.accuracy + (1 - w) * (1 - np.sum(solution.featuresUsed) / 30))
        bestSol = solution

end = time.time()
print("Time: ", end - start)

print("Best Solution is: ")
attrs = vars(bestSol)
print(', '.join("%s: %s" % item for item in attrs.items()))


