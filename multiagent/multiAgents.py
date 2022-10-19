# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numbers
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currFood = currentGameState.getFood()
        currPos = currentGameState.getPacmanPosition()

        currCloseFood = (99,99)
        newCloseFood = (99,99)

        currFoodPos = currFood.asList()
        newFoodPos = newFood.asList()

        currCloseFoodDist = 9999
        newCloseFoodDist = 9999

        newGhostDist = 9999

        score = successorGameState.getScore() - currentGameState.getScore()

        if len(currFoodPos) == 1:
            currCloseFood = currFoodPos[0]
        else:
            for f in currFoodPos:
                temp = currCloseFoodDist
                currCloseFoodDist = min(currCloseFoodDist, manhattanDistance(f, currPos))
                if temp != currCloseFoodDist:
                    currCloseFood = f

        if len(newFoodPos) == 1:
            newCloseFood = newFoodPos[0]
        else:
            for f in newFoodPos:
                temp = newCloseFoodDist
                newCloseFoodDist = min(newCloseFoodDist, manhattanDistance(f, newPos))
                if temp != newCloseFoodDist:
                    newCloseFood = f

        for g in newGhostStates:
            newGhostDist = min(newGhostDist, manhattanDistance(newPos, g.getPosition()))

        # print("CloseFood: ", currCloseFoodDist)
        eval = 0
        if currCloseFoodDist <= newCloseFoodDist:
            eval += 0
        else:
            eval += 20
        
        if abs(score) == 1:
            eval += 0
        else:
            eval += score
        # if manhattanDistance(newPos, closestGhost) > manhattanDistance(currPos, ghost):
        #         eval += 3
        if newGhostDist < 2:
            #check if newGhostDist > currGhostDist
            eval -= 99
        else:
            eval += 0
        # print(eval)
        return eval

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        numAgents = gameState.getNumAgents()
        def maxDepth(successors):  
            return self.evaluationFunction(successors)


        def maxValue(index, successors):
            if index == numAgents * self.depth:
                return maxDepth(successors)
            valList = [] 
            possibleActions = successors.getLegalActions(index % numAgents)
            for a in possibleActions:
                valList.append(value(index + 1, successors.generateSuccessor(index % numAgents, a)))
            return max(valList)
        def minValue(index, successors):
            if index == numAgents * self.depth:
                return maxDepth(successors)
            valList = []
            possibleActions = successors.getLegalActions(index % numAgents)
            # v = float('inf')
            for a in possibleActions:
                valList.append(value(index + 1, successors.generateSuccessor(index % numAgents, a)))
            return min(valList)
        # index indicates which agent (Pacman or Ghost)
        # index = 0 is Pacman, index > 0 should be Ghosts
        # Sucessors is a MultiAgent
        def value(index, successors):
            #need to be checking the terminal state
            if successors.isWin() or successors.isLose():
                return successors.getScore()
            elif index % numAgents == 0: #need to be adding 1 to index to "treverse" the tree depths
                return maxValue(index, successors)
            else:
                return minValue(index, successors)



        retVals = []
        legalAction = gameState.getLegalActions(0)
        for action in legalAction:
            # print(gameState.generateSuccessor(0, action))
            retVals.append((action, value(1, gameState.generateSuccessor(0, action))))
        maxMove = -1 * float('inf')
        move = 0
        for v in retVals:
            temp = maxMove
            maxMove = max(maxMove,v[1])
            if temp != maxMove:
                move = v
        return move[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        global count
        count = 0
        numAgents = gameState.getNumAgents()
        def maxDepth(successors):  
            return self.evaluationFunction(successors)


        def maxValue(index, successors, alpha, beta):
            if index == numAgents * self.depth:
                return maxDepth(successors)
            v = -1 * float('inf')
            possibleActions = successors.getLegalActions(index % numAgents)
            for a in possibleActions:
                global count
                count += 1
                v = max(v,value(index + 1, successors.generateSuccessor(index % numAgents, a), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        def minValue(index, successors, alpha, beta):
            if index == numAgents * self.depth:
                return maxDepth(successors)
            v = float('inf')
            possibleActions = successors.getLegalActions(index % numAgents)
            for a in possibleActions:
                global count
                count += 1
                v = min(v, value(index + 1, successors.generateSuccessor(index % numAgents, a), alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        # index indicates which agent (Pacman or Ghost)
        # index = 0 is Pacman, index > 0 should be Ghosts
        # Sucessors is a MultiAgent
        def value(index, successors, alpha, beta):
            #need to be checking the terminal state
            if successors.isWin() or successors.isLose():
                return successors.getScore()
            elif index % numAgents == 0: #need to be adding 1 to index to "treverse" the tree depths
                return maxValue(index, successors, alpha, beta)
            else:
                return minValue(index, successors, alpha, beta)


        alpha = -1 * float('inf')
        beta = float('inf')

        retVals = []
        returnAction = -1
        v = -1 * float('inf')
        legalAction = gameState.getLegalActions(0)
        for action in legalAction:
            count += 1
            temp = v
            v = max(v, value(1, gameState.generateSuccessor(0, action), alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
            if temp != v:
                returnAction = action
        return returnAction
       
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        " YOUR CODE HERE "
        return self.value(gameState, 0, self.depth)[1]

    def value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
        elif index == 0:
            return self.maxValue(gameState, index, depth)
        else:
            return self.expectVal(gameState, index, depth)

    def expectVal(self, gameState, index, depth):
        if index == gameState.getNumAgents() - 1:
            newAgent = 0
            depth -= 1
        else:
            newAgent = index + 1
        exp = 0
        possibleActions = gameState.getLegalActions(index)
        totActions = len(possibleActions)
        for a in possibleActions:
            successorState = gameState.generateSuccessor(index, a)
            exp += self.value(successorState, newAgent, depth)[0]
        return exp/totActions, None 

    def maxValue(self, gameState, index, depth):
        if index == gameState.getNumAgents() - 1:
            newAgent = 0
            depth -= 1
        else:
            newAgent = index + 1
        possibleActions = gameState.getLegalActions(index)
        alpha = -999 
        returnAction = None
        for a in possibleActions:
            successorState = gameState.generateSuccessor(index, a)
            score = self.value(successorState, newAgent, depth)[0]
            if score > alpha:
                returnAction = a
                alpha = score
        return alpha, returnAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    My heuristic funciton is the manhattanDistance and I apply the heuristic function 
    to find the closest food and closest ghost to pacMan. I care if the ghost scared time 
    is 0 or not because a close ghost is bad and at this point, I would want to run away. 
    """

    " YOUR CODE HERE "
    def findClosestFood(currentGameState: GameState, foodList):
        closestFood = 999
        if len(foodList) == 0:
            closestFood = 0
        else:
            for food in foodList:
                    mhFoodDistance = manhattanDistance(food, currentGameState.getPacmanPosition())
                    closestFood = min(mhFoodDistance, closestFood)
        return closestFood

    def findClosestGhost(currentGameState: GameState, ghostList):
        closestGhost = 999
        for ghost in ghostList:
            xPos, yPos = ghost.getPosition()
            if ghost.scaredTimer == 0: #find dist from closesest ghost then run away from dot
                mhGhostDistance = manhattanDistance((int(xPos), int(yPos)), currentGameState.getPacmanPosition())
                closestGhost = min(mhGhostDistance, closestGhost)
            else:
                closestGhost = -99
        return closestGhost

    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostStates()
    minFoodDistance = findClosestFood(currentGameState, foodList)
    minGhostDistance = findClosestGhost(currentGameState, ghostList)
    ghostDist = 10 / (minGhostDistance+1)
    foodDist = minFoodDistance / 3
    return currentGameState.getScore() - ghostDist - foodDist

# Abbreviation
better = betterEvaluationFunction
