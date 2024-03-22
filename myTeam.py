# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint, PriorityQueue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        #globale variablen kunnen othouden worden in bv de Init
        #naive implementatie (als laatste 4 states gelijk zijn aan nieuwe 4 states dan stoppen we met bewegen)
        #betere implementatie (als we aan de grens komen van ons veld en we worden achtervolgt
        #dan rennen we verder in ons veld om uit de vision te komen van enemy agent en zijn natuurlijk pad te nemen)
        #Dit werkt wel niet bij enemy agents die aan borders blijven
        #laatste implementatie is ga naar het noorden/midden of zuiden om weg te gaan van het defending border ghost
        #self.global_variable = 0
        self.last_states = []
        self.successor = 0
        self.forced_movement = []
        self.check = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height - 1
        self.width = game_state.data.layout.width

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        hasfood = my_state.num_carrying
        my_pos = my_state.get_position()
        walls = game_state.get_walls().as_list()
        teamcolored_red = game_state.is_on_red_team(self.index)
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        features['retreat'] = 0
        old_state = game_state.get_agent_state(self.index)
        old_food = old_state.num_carrying
        old_pos = old_state.get_position()
        last_states = self.last_states
        forced_movement = self.forced_movement
        retreat_locations = [(0,0),(0,0)]

        x_start = self.start[0]
        y_start = self.start[1]

        first_retreat_x = 17
        second_retreat_x = 17
        first_retreat_y = 1
        if teamcolored_red:
            while (first_retreat_x, y_start) in walls:
                first_retreat_x -= 1
            while (second_retreat_x, first_retreat_y) in walls:
                second_retreat_x -= 1
            retreat_locations[0] = (first_retreat_x, y_start)
            retreat_locations[1] = (second_retreat_x, first_retreat_y)
        else:
            while (first_retreat_x, y_start) in walls:
                first_retreat_x += 1
            while (second_retreat_x, first_retreat_y) in walls:
                second_retreat_x += 1
            retreat_locations[0] = (first_retreat_x, y_start)
            retreat_locations[1] = (second_retreat_x, first_retreat_y)

        if len(last_states) < 8 and self.successor == 0:
            self.last_states.append(old_pos)
            self.successor += 1
        elif self.successor == 0:
            self.last_states.pop()
            self.last_states.insert(0, old_pos)
            self.successor += 1
        elif self.successor == 4:
            self.successor = 0
        else:
            self.successor += 1

        all_unique_last_states = []
        for state in self.last_states:
            if state not in all_unique_last_states:
                all_unique_last_states.append(state)

        def give_surrounding_cords(coordinate):
            x = coordinate[0]
            y = coordinate[1]
            return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        def euclidean_distance(coordinate1, coordinate2):
            return ((coordinate1[0]-coordinate2[0])**2 + (coordinate1[1]-coordinate2[1])**2)**(1/2)

        exit_position = [0]

        def a_star_exit(old_pos, goal_state, wanted_distance):
            agenda = util.PriorityQueue()
            agenda.push(item=(old_pos, [], 0), priority=0)
            visited = []
            while True:
                if not agenda.isEmpty():
                    current_pos, path, UCS = agenda.pop()
                    if not current_pos in visited:
                        visited.append(current_pos)
                        if current_pos == goal_state:
                            exit_position[0] = path[0:wanted_distance + 1]
                            break
                        for new_coordinate in give_surrounding_cords(current_pos):
                            if not new_coordinate in walls:
                                UCS = UCS + 1
                                new_path = path + [new_coordinate]
                                heuristic = euclidean_distance(new_coordinate, goal_state)
                                total_cost = UCS + heuristic
                                agenda.push(item=(new_coordinate, new_path, UCS), priority=total_cost)

        if len(all_unique_last_states) < 5 and len(last_states) == 8 and 12 < old_pos[0] < 20:
            if old_pos[1] < 5:
                a_star_exit(old_pos, retreat_locations[0], 16)
                self.forced_movement = exit_position[0]
                forced_movement = self.forced_movement
                self.last_states = []
            elif old_pos[1] > 10:
                a_star_exit(old_pos, retreat_locations[1], 16)
                self.forced_movement = exit_position[0]
                forced_movement = self.forced_movement
                self.last_states = []
            else:
                a_star_exit(old_pos, retreat_locations[0], 16)
                self.forced_movement = exit_position[0]
                forced_movement = self.forced_movement
                self.last_states = []


        #Berekend de afstand naar het dichtste eten (onveranderd tegenover baseline team)
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #Deze berekent de afstand naar ghost en zal beslissen tussen een retreat of food gathering
        #De ghost_distance string waarde beslist of we in dead ends lopen
        features['num_defenders'] = len(defenders)
        ghost_distance = "safe"
        if len(defenders) > 0 and my_state.is_pacman:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            if min(dists) <= 3:
                features['defender_distance'] = min(dists)
                features['distance_to_food'] = 0
                features['retreat'] = self.get_maze_distance(self.start, my_pos)
                ghost_distance = "dangerous"
            elif min(dists) <= 7:
                ghost_distance = "dangerous"
            else:
                features['defender_distance'] = 0

        #De volgende 2 functies geven de coordinaten terug en berekend daarvan of het een dead end is of niet
        x_p = my_pos[0]
        y_p = my_pos[1]
        old_x_p = old_pos[0]
        old_y_p = old_pos[1]


        total_exits = [0]

        def bfs_entry(check_list, n, visited_list):
            if n > 0:
                new_check_list = ([], [], [], [])
                new_visited_list = visited_list
                for idx, direction in enumerate(check_list):
                    for coordinate in direction:
                        if coordinate not in walls:
                            for new_coordinate in give_surrounding_cords(coordinate):
                                if new_coordinate not in visited_list:
                                    new_visited_list.append(new_coordinate)
                                    new_check_list[idx].append(new_coordinate)
                bfs_entry(new_check_list, n - 1, new_visited_list)
            else:
                list_length = 0
                for direction in check_list:
                    if not direction == []:
                        list_length = list_length + 1
                total_exits[0] = list_length

        #De volgende 2 if testen checken of we moeten teruggaan omdat we genoeg food hebben
        #en ook om te forceren om terug in uw eigen maze te gaan als je meer dan 1 food hebt
        features["risk"] = 0
        features["flat"] = 0
        if hasfood >= 3:
            features["flat"] = 100
            features["retreat"] = self.get_maze_distance(self.start, my_pos)
            features['distance_to_food'] = 0
        if old_food >= 1 and hasfood == 0:
            features['flat'] = 10000

        #Dit is de lijst die we checken of er dan een exit is
        if not old_pos == my_pos:
            Check_List = [[(x_p + 1, y_p)], [(x_p - 1, y_p)], [(x_p, y_p + 1)], [(x_p, y_p - 1)]]
            for idx, coordinate_array in enumerate(Check_List):
                coordinate = coordinate_array[0]
                if old_x_p == coordinate[0] and old_y_p == coordinate[1]:
                    Check_List[idx] = []
            bfs_entry(Check_List, 6, [])

        #Hier gaan we nooit in een dead end als er een ghost dichtbij is
        if total_exits[0] == 0 and not ghost_distance == "safe" and not old_pos == my_pos:
            features["risk"] = 1

        # Hier komt code die eigenlijk de pacman uit dead ends duwt als de ghost te dicht is (want het wilt anders blijven)

        if features['risk'] == 1:
            a_star_exit(old_pos, self.start, 0)
            if my_pos == exit_position[0]:
                features["risk"] = 0.1


        #Ik ben gegaan met het idee dat je als een offensieve agent nooit echt wilt stilstaan
        if old_pos == my_pos:
            features['dontstay'] = 1

        if len(self.forced_movement) > 0:
            if my_pos in self.forced_movement:
                features["flat"] = 100000
                features["distance_to_food"] = 0
                features["defender_distance"] = 0
                features["retreat"] = 0
                features["risk"] = 0
                forced_movement.remove(my_pos)
                self.forced_movement = forced_movement
                self.check = 0
            else:
                features["flat"] = -100000
                features["distance_to_food"] = 0
                features["defender_distance"] = 0
                features["retreat"] = 0
                features["risk"] = 0
                self.check += 1
                if self.check > 10:
                    self.forced_movement = []

        return features
    #to do, verander bfs_exit naar line checker (check verschil
    #op x,y van oude en nieuwe state en doe die operatie herhaaldelijk)
    #om te zien of er links, rechts een muur is doen we door de operatie die op x en y gebeuren
    #te laten toegepast worden op y en x en dan ook het invers van die operatie (dus + en -)
    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'defender_distance': 10, 'retreat': -1, 'risk': -10000
            , "flat": 1, 'dontstay': -10000}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        teamcolored_red = game_state.is_on_red_team(self.index)
        Top_border_y = self.height
        Bottom_border_y = 0
        walls = game_state.get_walls()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        left_half = int(self.midWidth - 1)
        right_half = int(self.midWidth + 1)

        # Euclidean distance calculation
        def euclidean_distance(coordinate1, coordinate2):
            return ((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2) ** (1 / 2)

        # Generate surrounding coordinates
        def give_surrounding_cords(coordinate):
            x = coordinate[0]
            y = coordinate[1]
            return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Determine the border based on team color
        def border_find(is_red):
            if is_red:
                return left_half
            else:
                return right_half

        border_x = border_find(teamcolored_red)

        # Find position without wall
        def avoid_top_wall(x, y):
            if (x, y) not in walls:
                return (x, y)
            else:
                return avoid_top_wall(x, y - 1)

        # Find position without wall
        def avoid_bottom_wall(x, y):
            if (x, y) not in walls:
                return (x, y)
            else:
                return avoid_bottom_wall(x, y + 1)

        top_border = avoid_top_wall(border_x, Top_border_y)
        bottom_border = avoid_bottom_wall(border_x, Bottom_border_y)

        wanted_position = top_border
        if my_pos[1] == top_border[1]:
            wanted_position = top_border
        if my_pos[1] == bottom_border[1]:
            wanted_position = bottom_border

        # A* search for the patrol path position
        def a_star_patrol(old_pos, goal_state):
            agenda = util.PriorityQueue()
            agenda.push(item=(old_pos, [], 0), priority=0)
            visited = []
            while not agenda.isEmpty():
                    current_pos, path, UCS = agenda.pop()
                    if  current_pos not in visited:
                        visited.append(current_pos)
                        if current_pos == goal_state:
                            return path
                        for new_coordinate in give_surrounding_cords(current_pos):
                            if new_coordinate not in walls:
                                UCS = UCS + 1
                                new_path = path + [new_coordinate]
                                heuristic = euclidean_distance(new_coordinate, goal_state)
                                total_cost = UCS + heuristic
                                agenda.push(item=(new_coordinate, new_path, UCS), priority=total_cost)

        #patrole_path = a_star_exit(my_pos, wanted_position, 0)
        patrole_path = a_star_patrol(my_pos, wanted_position)
        features['on_border_patrol'] = 1 if my_pos in patrole_path else 0

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1 if my_state.is_pacman else 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0

        # Attack mode
        features['on_hunt'] = 1 if len(invaders) > 0 else 0
        return features

    def get_weights(self, game_state, action):
        if game_state.get_agent_state(self.index).scared_timer > 0:
            return {'num_invaders': 0,
                    'on_defense': 200,
                    'invader_distance': 0,
                    'stop': -10,
                    'reverse': -2,
                    'on_border_patrol': 4000,
                    'on_hunt': 0}
        else:
            return {'num_invaders': -10000,
                    'on_defense': 300,
                    'invader_distance': -10,
                    'stop': -10,
                    'reverse': -2,
                    'on_border_patrol': 4000,
                    'on_hunt': 10000}