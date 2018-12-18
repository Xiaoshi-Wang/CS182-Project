import random,util,math,copy

class unitSpaceAgent:
    def __init__(self):
        self.state = None
        self.location = None

    def getLegalActions(self, state, location, occupiedCell, occupiedBoundary):
        """
        return the legal action based on the current state, occupiedCell and boundary,
        challenge is that the environment here, comparing to the gridworld and Pacman,
        is changing according to the previous state, so we need to track the current
        world configuration so as to know where we cannot go for the next step.

        :param state: the current unit space, which is a tuple of four int, e.g. (2,4,0,3),
        meaning (inlet position, outlet position, inlet type, outlet type).

        Note: In the improved version (currently implimented), there is the 5th parameter in
        state which is the sequence index of the unitspace in the sequence, e.g. 0 means 1st
        space, 1 means 2nd space...

        :param location: location of the current state, as a tuple (x,y).

        :param occupiedCell: the list of all the occupied cells in the gridworld, each cell
        is a 1*1 square represented by its bottom left vertex (a two element tuple (x,y)).

        :param occupiedBoundary: the list of all the occupied boundaries in the gridworld, each boundary
        is a 1 length line segment represented by its middle point (a two element tuple (x,y)),
        e.g. (0.5, 1.0).
        :return: the list of legal actions of the current state, which is a tuple of three elements:
        (outlet position of the current state, inlet position of the next state, outlet position
        of the next state), e.g. (4,11,9).
        """
        actionList = []
        if state == (-1,-1,-1,-1):
            return actionList
        else:
            posibleInpos = self.posibleInposForNextState(state[1]) #self.state[1] is the outpos of the previous state (unit space)
            for inpos in posibleInpos:
                tempAction = (state[1], inpos, 0) #0 is a placeholder that will not be used in here
                newLocation = self.newLocationAfterAction(location,tempAction)
                newCells = [(newLocation[0]+i,newLocation[1]+j) for j in range(3) for i in range(3)]
                overlap = set(newCells).intersection(set(occupiedCell))
                if len(overlap) == 0:
                    for outpos in range(12):
                        if not outpos == inpos and not self.convertOutposToBoundary(newLocation, outpos) in occupiedBoundary:
                            action = (state[1], inpos, outpos)
                            actionList.append(action)
            return actionList

    def posibleInposForNextState(self, outpos):
        nextInpos = []
        if 0<= outpos <= 2:
            nextInpos = [6, 7, 8]
        elif 3<= outpos <= 5:
            nextInpos = [9, 10, 11]
        elif 6<= outpos <= 8:
            nextInpos = [0, 1, 2]
        elif 9<= outpos <= 11:
            nextInpos = [3, 4, 5]
        return nextInpos

    def convertOutposToBoundary(self, location, outpos):
        boundary = None
        if 0<= outpos <= 2:
            boundary = (outpos+0.5+location[0], 0.0+location[1])
        elif 3<= outpos <= 5:
            boundary = (3.0+location[0], outpos-2.5+location[1])
        elif 6<= outpos <= 8:
            boundary = (8.5-outpos+location[0], 3.0+location[1])
        elif 9<= outpos <= 11:
            boundary = (0.0+location[0], 11.5-outpos+location[1])
        return boundary

    def nextstate(self, state, action, library):
        if action is None:
            nextstate = (-1,-1,-1,-1)
        else:
            nextstate = (action[1], action[2], state[3], int(library[(str(action[1]), str(action[2]), str(state[3]))][3]), state[4]+1)
        return nextstate

    """
    def nextstate(self, state, action, library):
        if action is None:
            nextstate = (-1,-1,-1,-1)
        else:
            nextstate = (action[1], action[2], state[3], int(library[(str(action[1]), str(action[2]), str(state[3]))][3]))
        return nextstate
    """

    def doAction(self, location, action, nextstate):
        self.state = nextstate
        if nextstate == (-1,-1,-1,-1):
            self.location = (float('inf'), float('inf'))
        else:
            self.location = self.newLocationAfterAction(location, action)

    def newLocationAfterAction(self, location, action):
        newLocation = None
        if action is None:
            newLocation = (float('inf'), float('inf'))
        elif 0 <= action[0] <= 2:
            newLocation = (location[0]+action[0]+action[1]-8,location[1]-3)
        elif 3 <= action[0] <= 5:
            newLocation = (location[0]+3,location[1]+action[0]+action[1]-14)
        elif 6 <= action[0] <= 8:
            newLocation = (location[0]-action[0]-action[1]+8,location[1]+3)
        elif 9 <= action[0] <= 11:
            newLocation = (location[0]-3,location[1]-action[0]-action[1]+14)
        return newLocation

    def getCurrentState(self):
        return self.state

    def getStartState(self, library):
        """
        self.state must include the sequence order of the current state,
        e.g. 1st or 2nd or 3rd or 4th of the sequence
        """
        """
        #Here is a fixed starting state
        startInpos = 1
        startOutpos = 5
        """
        startInpos = random.randint(0,2)
        startOutpos = startInpos
        while startOutpos == startInpos:
            startOutpos = random.randint(0,11)

        startState = (startInpos, startOutpos, 0, int(library[(str(startInpos),str(startOutpos),str(0))][3]), 0)
        startLocation = (0,0)
        return startState, startLocation

    """
    def getStartState(self, library):
        startInpos = random.randint(0,2)
        startOutpos = startInpos
        while startOutpos == startInpos:
            startOutpos = random.randint(0,11)
        startState = (startInpos, startOutpos, 0, int(library[(str(startInpos),str(startOutpos),str(0))][3]))
        startLocation = (0,0)
        return startState, startLocation
    """

    def reset(self, library):
        self.state, self.location= self.getStartState(library)