from typing import Dict

from gym.spaces import MultiBinary
from dijkstar import Graph, find_path
from .TabularQAgent import Tabular_Q_Agent
class RoutingAgent:
        # Agent interface
        def __init__(self, state_space, action_space):
            number_connections = action_space.n//3
            self.state_space = state_space
            self.action_space = action_space
            self.agents = []
            for i in range(number_connections):
                agent = Tabular_Q_Agent(state_space[0], MultiBinary(3),  )
                self.agents.append(agent)
            self.next_node_index = [None for i in range(len(state_space.nvec))]
            self.edges = [(0, 1), (1, 2), (2, 6),(0,3), (3, 4), (4,3), (3,5), (5,6)]


        # returns  an action for the given state.
        # Must also return extras, None is ok if the alg does not use them.
        def act(self, states, connection_costs):
            # determine which connections to use via djikstras
            self._update_path(connection_costs)

            # fetch action from decentralized agent. If not being used, is all 0s
            actions = [0 for i in range(self.action_space.n)]
            for i in range(len(self.next_node_index)):
                # skip nodes not in path
                next_node = self.next_node_index[i]
                if next_node is None:
                    continue

                # else get action
                active_edge = (i, next_node)
                index_of_edge =  self.edges.index(active_edge)
                channel_actions = self.agents[index_of_edge].act(states[i])

                # assign to global actions
                actions[3*index_of_edge:3*index_of_edge + 3] = channel_actions
            return actions


        def _update_path(self, costs):
            # create graph
            graph = Graph()
            for e, c in zip(self.edges, costs):
                graph.add_edge(e[0], e[1], c)

            # run djikstras
            path = find_path(graph, 0, 6)

            # keep track of next node for each node.
            for i in range(len(path.nodes)-1):
                self.next_node_index[path.nodes[i]] = path.nodes[i+1]



        # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
        # ocassionally updates the policy, but always stores transition

        def learn(self, states, actions, rewards, next_states, done, truncated, info, ):
            # Update all agents
            for i in range(len(self.next_node_index)):
                next_node = self.next_node_index[i]
                if next_node is None:
                    continue

                edge = (i, next_node)
                index = self.edges.index(edge)
                state = states[i]
                reward = rewards[index]
                action = actions[3*index: 3*index + 3]
                next_state = next_states[i]

                self.agents[i].learn(state, action, reward, next_state, done, truncated, info)

        def plot(self):
            raise Exception("TODO")