import gym
from enum import Enum
import numpy as np
import pygame

class Result(Enum):
    LOST = 0
    DELAYED = 1
    RECIEVED = 2


class Node:
    def __init__(self, name, render_x, render_y, is_destination=False, max=4) -> None:
        self.messages_to_send = 0
        self.messages_to_send_limit = max
        self.is_destination = is_destination
        self.render_x = render_x
        self.render_y = render_y
        self.name = name

    def add_message(self):
        if self.messages_to_send < self.messages_to_send_limit:
            self.messages_to_send += 1
            return Result.RECIEVED
        else:
            return Result.LOST
    
    def render(self, canvas, size):
        pygame.draw.circle(
                canvas,
                (128,128,128),
                (self.render_x * size, self.render_y * size),
                0.1 * size,
            )
        s = "{} ({})".format(self.name, self.messages_to_send)
        my_font = pygame.font.SysFont('Comic Sans MS', 25)
        text_surface = my_font.render(s, False, 'black')
        canvas.blit(text_surface, (self.render_x * size - 5 * len(s), self.render_y * size - 5 ))


class Channel:
    def __init__(self, start, end, packet_loss=0.01, delay_percentage=0.1) -> None:
        self.packet_loss = packet_loss
        self.delay_percentage = delay_percentage
        self.start = start
        self.end = end

    def send_message(self):
        # message lost,retry
        if np.random.rand() < self.packet_loss:
            return Result.LOST
        
        ack = self.end.add_message()
        if ack == Result.RECIEVED:
            self.start.messages_to_send -= 1
        return ack

    

class Connection:
    def __init__(self, channels, start, end, color=(0,0,0)) -> None:
        self.channels = channels
        self.start = start
        self.end = end
        self.cost = 0
        self.color = color

    def render(self, canvas, size):
        pygame.draw.line(
                canvas,
                self.color,
                (self.start.render_x * size, self.start.render_y * size),
                (self.end.render_x * size, self.end.render_y * size),
                width=3,
            )

    def update_cost(self, actions):
        if all([a == 0 for a in actions]):
            return
        self.cost = 0
        for c, a in zip(self.channels, actions):
            self.cost += a * c.packet_loss




class RoutingEnv(gym.Env):

    def __init__(self, render_mode="human") -> None:
        self.nodes = [Node("Start", 0.3, 0.1), 
                      Node("B", 0.2, .4), 
                      Node("C", 0.2, .7), 
                      Node("D", .6, 0.1), 
                      Node("E", .5, .5), 
                      Node("F", 0.9, .4), 
                      Node("Goal", 0.5, 0.9, is_destination=True, max=10000)]


        # connect all nodes with 3 channels of different risks
        self.connections = []

        #  add high risk connections
        pairs = [(0,1), (1,2), (2,6),]
        for p in pairs:
            first = self.nodes[p[0]]
            second = self.nodes[p[1]]
            self.connections.append(Connection([
                Channel(first, second, packet_loss=0.1),
                Channel(first, second, packet_loss=0.2),
                Channel(first, second, packet_loss=0.5),
                ],
                start=first,
                end=second,
                color=(255,0,0)
                )
            )

        #  add low risk connections
        pairs = [(0,3), (3, 4), (4,3), (3,5), (5,6)]
        for p in pairs:
            first = self.nodes[p[0]]
            second = self.nodes[p[1]]
            self.connections.append(Connection([
                Channel(first, second, packet_loss=0.001),
                Channel(first, second, packet_loss=0.01),
                Channel(first, second, packet_loss=0.1),
                ],
                start=first,
                end=second
                )
            )

        # gym stuff
        self.observation_space = gym.spaces.MultiDiscrete([5 for i in range(len(self.nodes))])
        self.action_space = gym.spaces.MultiBinary(len(self.connections) * 3)
        self.time = 0
        self.time_limit = 100

        # render stuf
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.metadata = {"render_fps":15}

    def _observation(self):
        return [min(n.messages_to_send, n.messages_to_send_limit) for n in self.nodes]
        
    def _rewards(self):
        rewards = []
        for i in range(len(self.connections)):
            rewards.append(-self.connections[i].start.messages_to_send)
        return rewards
    

    def _update_costs(self, action):
        for i in range(len(self.connections)):
            self.connections[i].update_cost(action[3*i: 3*i+3])

    def get_costs(self):
        return [c.cost for c in self.connections]

    def step(self, action):
        rewards = [0 for n in self.connections]
        for i in range(len(self.connections)-1, -1, -1):
            for j in range(3):
                send_packet = action[i*3 + j]
                if send_packet:
                    if self.connections[i].start.messages_to_send > 0:
                        result = self.connections[i].channels[j].send_message()
                        if result == Result.LOST:
                            rewards[i] -= 0.1

        obs = self._observation()
        filled_penalty = self._rewards()
        rewards = [a+b for a,b in zip(rewards, filled_penalty)]

        # update channel costs
        self._update_costs(action)

        # add new messages
        self.nodes[0].messages_to_send = self.nodes[0].messages_to_send + 1

        self.time += 1
        return obs, rewards, False, self.time > self.time_limit, {}

    def reset(self):
        self.time = 0
        for n in self.nodes:
            n.messages_to_send = 0
        return self._observation(), {}


    def render(self):
        # init rendering
        if self.window is None and self.render_mode is not None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
        if self.clock is None and self.render_mode is not None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # render all nodes and connections
        for c in self.connections:
            c.render(canvas, self.window_size)

        for n in self.nodes:
            n.render(canvas, self.window_size)

        # finish up
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )