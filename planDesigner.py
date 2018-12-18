import random,util,math,copy,numpy
from qlearningDesigner import QLearningDesigner
from unitSpaceCFD import unitSpaceAgent


class planDraftEnvironment:
    def __init__(self):
        self.occupiedCell = []
        self.occupiedBoundary = []

    def clear(self):
        self.occupiedCell = []
        self.occupiedBoundary = []

    def update(self, location, newBoundaries):
        x,y = location
        newCells = [(x+p,y+j) for p in range(3) for j in range(3)]
        for c in newCells:
            if not c in self.occupiedCell:
                self.occupiedCell.append(c)
        #newBoundaries = [self.convertOutposToBoundary((x,y),k) for k in range(12)]
        for b in newBoundaries:
            if not b in self.occupiedBoundary:
                self.occupiedBoundary.append(b)

def runEpisode(agent, library, env, spaceAmount, episode, outfile):

    currentEpisode = 0
    sampleRecord = util.Counter()

    f = open(outfile, 'w')
    f.write('')
    f.close()

    while currentEpisode < episode:
        if currentEpisode > 0.7*episode and agent.epsilon > 0.05:
            agent.epsilon -= 5.0/episode
        agent.reset(library) #self.state is now the initial state, self.location is the initial location
        env.clear()
        newBoundaries = [agent.convertOutposToBoundary(agent.location, k) for k in range(12)]
        env.update(agent.location, newBoundaries)
        # the initial inlet should not be blocked by further unitspaces
        inletCells = [(agent.state[0],q) for q in range(-1,-10,-1)]
        env.occupiedCell.extend(inletCells)

        recordState = [agent.state]
        recordLocation = [agent.location]
        recordAction = []
        recordReward = \
            [[float(library[(str(agent.state[0]),str(agent.state[1]),str(agent.state[2]))][j+4]) for j in range(9)]]

        #Run an episode without updating the Q-values, but record all the information of this episode.
        for step in range(spaceAmount-1):
            action = agent.getAction(agent.state,agent.location,env.occupiedCell,env.occupiedBoundary)
            nextstate = agent.nextstate(agent.state,action,library)
            agent.doAction(agent.location,action,nextstate)

            recordState.append(agent.state)
            recordLocation.append(agent.location)
            recordAction.append(action)
            if agent.state == (-1,-1,-1,-1):
                recordReward.append([0.0 for _ in range(9)])
            else:
                recordReward.append([float(library[(str(agent.state[0]),str(agent.state[1]),str(agent.state[2]))][k+4]) for k in range(9)])

            newLocation = agent.location
            newBoundaries = [agent.convertOutposToBoundary(newLocation, k) for k in range(12)]
            env.update(newLocation, newBoundaries)

        #Update the Q-values after recording all the steps and all the rewards.
        agent.state = recordState[0]
        agent.location = recordLocation[0]
        env.clear()
        newBoundaries = [agent.convertOutposToBoundary(agent.location, k) for k in range(12)]
        env.update(agent.location, newBoundaries)
        # the initial inlet should not be blocked by further unitspaces
        inletCells = [(agent.state[0], k) for k in range(-1, -10, -1)]
        env.occupiedCell.extend(inletCells)

        reward = avgWindVelocity(recordReward)
        print(recordState)
        print(reward)

        if tuple(recordState) in sampleRecord:
            sampleRecord[tuple(recordState)] += 1
        else:
            sampleRecord[tuple(recordState)] = 1

        #Record the episode in the output file
        recordTheResult(outfile, recordState)

        for step in range(spaceAmount-1):
            action = recordAction[step]
            nextstate = recordState[step+1]
            #update the Q-Value, reward is based on all the unit space
            agent.update(agent.state, agent.location, action, nextstate, env.occupiedCell, env.occupiedBoundary, reward)
            print("update",step)
            agent.state = recordState[step+1]
            agent.location = recordLocation[step+1]

            newLocation = agent.location
            newBoundaries = [agent.convertOutposToBoundary(newLocation, k) for k in range(12)]
            env.update(newLocation, newBoundaries)

        currentEpisode += 1

    return sampleRecord

def avgWindVelocity(recordReward):
    flattenList = [item for sublist in recordReward for item in sublist]
    return numpy.mean(flattenList)

def stdWindVelocity(recordReward):
    std = []
    for l in recordReward:
        std.append(1-numpy.std(l))
    return numpy.mean(std)

def readCFDLibrary(infile):
    fp = open(infile, 'r').read().split('\n')
    d = util.Counter()
    for entry in fp:
        if entry:
            num = entry.split(',')
            d[(num[0], num[1], num[2])] = num
    return d

def recordTheResult(outfile, newLine):
    wf = open(outfile, 'a')
    for k in newLine:
        for j in k:
            wf.write(str(j)+',')
    wf.write('\n')
    wf.close()

if __name__ == '__main__':
    #from qlearningDesigner import QLearningDesigner
    #from unitSpaceCFD import unitSpaceAgent

    CFDLibrary = readCFDLibrary('2640_full_library.csv')
    #spaceAgent = unitSpaceAgent
    #print(int(CFDLibrary[(str(11),str(9),str(3))][3]))
    designer = QLearningDesigner()
    environment = planDraftEnvironment()
    sample = runEpisode(designer, CFDLibrary, environment, spaceAmount=4, episode=100000, outfile='outputFile.csv')
    print("Finish")
    print("Total explored compositions: ", len(sample))
    print([sample.sortedKeys()[i] for i in range(5)])
    print([sample[sample.sortedKeys()[i]] for i in range(5)])

    topN = 50
    Nmostsamples = [sample.sortedKeys()[i] for i in range(topN)]

    import matplotlib.pyplot as plt
    #n, bins, patches = plt.hist([sample[i] for i in Nmostsamples], bins=topN)

    for i in range(topN):
        plt.plot([i,i], [sample[Nmostsamples[i]],0], label=str(Nmostsamples[i]))

    plt.grid(True)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

