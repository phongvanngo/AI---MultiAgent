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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        # util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            legalActions = state.getLegalActions(0)

            value = []
            for action in legalActions:
                # value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            if (bestAction == "Stop"):
                stopIndex = legalActions.index('Stop')
                i = 0
                while i < len(value):
                    if ((value[stopIndex] == value[i]) and (legalActions[i] != "Stop")):
                        bestAction = legalActions[i]
                    i += 1

            print(legalActions)
            print("value", value)
            print(bestAction)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(
                    agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(
                    agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        PACMAN = 0

        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)

            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP

            valueArray = []

            for action in actions:
                score = min_agent(state.generateSuccessor(
                    PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                valueArray.append(score)
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score

            if depth == 0:
                print(actions)
                print(valueArray)
                if (best_action == "Stop"):
                    stopIndex = actions.index('Stop')
                    i = 0
                    while i < len(valueArray):
                        if ((valueArray[stopIndex] == valueArray[i]) and (actions[i] != "Stop")):
                            best_action = actions[i]
                        i += 1
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(
                            state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(
                            ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(
                        ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return max_agent(gameState, 0, float("-inf"), float("inf"))


class ExpectimaxAgent0(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return expectimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (expectimax_search(state.generateSuccessor(
                    agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return expectimax values
                else:
                    l = list(next)
                    return sum(l) / len(l)
        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0), key=lambda x: expectimax_search(
            gameState.generateSuccessor(0, x), 1, 1))

        return result


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Get the action and score for pacman (agent_index=0)
        action, score = self.expectimax(0, 0, gameState)
        return action  # Return the action to be done as per minimax algorithm

    def expectimax(self, curr_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the expectimax algorithm. For max player (agent_index=0), the best
        score is the maximum score among its successor states and for the min player (agent_index!=0), the best
        score is the average of all its successor states. Recursion ends if there are no successor states
        available or curr_depth equals the max depth to be searched until.
        :param curr_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        # Roll over agent index and increase current depth if all agents have finished playing their turn in a move
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        # Return the value of evaluationFunction if max depth is reached
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        # Initialize best_score and best_action with None
        best_score, best_action = None, None
        if agent_index == 0:  # If it is max player's (pacman) turn
            # For each legal action of pacman
            for action in gameState.getLegalActions(agent_index):
                # Get the expectimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost) turn now
                # Pass the new game state generated by pacman's `action`
                next_game_state = gameState.generateSuccessor(
                    agent_index, action)
                _, score = self.expectimax(
                    curr_depth, agent_index + 1, next_game_state)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:  # If it is min player's (ghost) turn
            ghostActions = gameState.getLegalActions(agent_index)
            if len(ghostActions) is not 0:
                prob = 1.0 / len(ghostActions)
            # For each legal action of ghost agent
            for action in gameState.getLegalActions(agent_index):
                # Get the expectimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost or pacman) turn now
                # Pass the new game state generated by ghost's `action`
                next_game_state = gameState.generateSuccessor(
                    agent_index, action)
                _, score = self.expectimax(
                    curr_depth, agent_index + 1, next_game_state)

                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action
        # If it is a leaf state with no successor states, return the value of evaluationFunction
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  # Return the best_action and best_score


class MyGhost:
    scaredTime = 0
    position = 0
    mahhattan = 0


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # gán trọng số
    foodCoeff, powerCoeff, ghostCoeff, gameStateScoreCoeff = 2, 3, 1, 1
    currPos, currFood = currentGameState.getPacmanPosition(), currentGameState.getFood()
    currPower = currentGameState.getCapsules()
    scaredTimes = [
        ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    # find the closest ghost
    theClosestGhost = MyGhost()
    closest_distance = 1000000000
    for ghostState in currentGameState.getGhostStates():
        ghostPos = ghostState.getPosition()
        ghostScareTimer = ghostState.scaredTimer
        distance = util.manhattanDistance(ghostPos, currPos)

        if ((distance) < closest_distance) or (distance == closest_distance and ghostScareTimer < theClosestGhost.scaredTime):
            closest_distance = distance
            theClosestGhost.position = ghostPos
            theClosestGhost.mahhattan = distance
            theClosestGhost.scaredTime = ghostScareTimer
    # tính điểm ghost score

    if (theClosestGhost.mahhattan < 1 and theClosestGhost.scaredTime == 0):
        ghostScore = 100
        ghostCoeff = -100
    else:
        ghostScore = 1 / theClosestGhost.mahhattan
    if (theClosestGhost.scaredTime > 0):
        ghostCoeff = 100
        ghostScore = abs(ghostScore)

    # if isScared and closestGhost < max(scaredTimes):
    #     ghostCoeff, ghostScore = 100, abs(ghostScore)

    # tính khoảng cách pacman -> chấm thức ăn gần nhất, trường hợp không còn chấm thức ăn nào sẽ lấy 999
    foodDistance = [util.manhattanDistance(
        currPos, foodPos) for foodPos in currFood.asList()] + [999]
    closestFood = float(min(foodDistance))

    #   tính khoảng cách đến chấm thức ăn To
    powerDistance = [
        999] + [util.manhattanDistance(powerPos, currPos) for powerPos in currPower]
    closestPower = float(min(powerDistance))

    #   Tính evaluation
    foodScore = 1 if len(currFood.asList()) == 0 else 1 / closestFood
    powerScore = 1 if len(currPower) == 0 else 1 / closestPower

    total_score = foodCoeff*foodScore + powerCoeff*powerScore + ghostCoeff * \
        ghostScore + gameStateScoreCoeff*currentGameState.getScore()

    return total_score


# Abbreviation
better = betterEvaluationFunction
