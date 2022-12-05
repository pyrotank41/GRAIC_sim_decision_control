import rospy
import rospkg
import numpy as np
import argparse
import time
from graic_msgs.msg import ObstacleList, ObstacleInfo
from graic_msgs.msg import LocationInfo, WaypointInfo
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaEgoVehicleControl
from graic_msgs.msg import LaneList
from graic_msgs.msg import LaneInfo

from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import math
import threading



def prune_path(path):
        def point(p):
            return np.array([p[0], p[1], 1.]).reshape(1, -1)

        def collinearity_check(p1, p2, p3, epsilon=0.1):
            m = np.concatenate((p1, p2, p3), 0)
            det = np.linalg.det(m)
            return abs(det) < epsilon
        pruned_path = []
        # TODO: prune the path!
        p1 = path[0]
        p2 = path[1]
        pruned_path.append(p1)
        for i in range(2,len(path)):
            p3 = path[i]
            if collinearity_check(point(p1),point(p2),point(p3)):
                p2 = p3
            else:
                pruned_path.append(p2)
                p1 = p2
                p2 = p3
        pruned_path.append(p3)

        return np.array(pruned_path)


plot_shown = False
def plot(path, grid, start, goal, tvec, scale):
    global plot_shown
    path = np.array(path)/scale + tvec
    grid_x_y = np.array(np.where(grid == 1)).astype(np.int16)
    print(grid_x_y)
    grid_x_y = grid_x_y/scale + tvec.reshape(2,1).astype(np.int16)
    print(grid_x_y)
    
    print(f"start {start}")
    start = np.array(start)/scale + tvec
    print(f"current {start}")
    print(f"goal {goal}")
    goal = np.array(goal)/scale + tvec
    print(f"waypoint {goal}")

    plt.scatter(grid_x_y[0], grid_x_y[1], s=5, color='k', alpha=.1)
    plt.plot(start[0], start[1], 'x')
    plt.plot(goal[0], goal[1], 'xr')
    pp = np.array(path)
    plt.plot(pp[:, 0], pp[:, 1], 'g', alpha=0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    # if not plot_shown:
    plt.show()
        # plot_shown = True
    

def scaled_points_on_grid(points, scale): 
    return np.floor(points * scale).astype(int)
   

def get_transform_min(left_lane_location, right_lane_location, current_position, waypoint):
    # min and max values for each parameter
    min_x = min(left_lane_location[:, 0].min(), right_lane_location[:, 0].min(), current_position[0], waypoint[0])
    min_y = min(left_lane_location[:, 1].min(), right_lane_location[:, 1].min(), current_position[1], waypoint[1])
    return (min_x, min_y)

def get_transform_max(left_lane_location, right_lane_location, current_position, waypoint):
    # min and max values for each parameter
    max_x = max(left_lane_location[:, 0].max(), right_lane_location[:, 0].max(), current_position[0], waypoint[0])
    max_y = max(left_lane_location[:, 1].max(), right_lane_location[:, 1].max(), current_position[1], waypoint[1])
    
    return (max_x, max_y)




def plan(current_state, waypoint, left_lane, right_lane, obstacles):
    global plot_shown
    current_position = np.array(current_state[0])
    waypoint = np.array([waypoint.x, waypoint.y])
    
    scale = 1
    padding = 2 # padding around the occupancy grid, in meters (unit of measurement value)


    # transform to make current position the origin
    left_lane_t = left_lane - current_position
    right_lane_t = right_lane - current_position 
    current_position_t = current_position - current_position
    waypoint_t  =  waypoint - current_position 
    
    # Getting the transform to shift all the values to positive values as grid index cannot have negative values
    min_transform = get_transform_min(left_lane_t, right_lane_t, current_position_t, waypoint_t)
    
    left_lane_t = left_lane_t - min_transform
    right_lane_t = right_lane_t - min_transform
    current_position_t = current_position_t - min_transform
    waypoint_t = waypoint_t - min_transform

    # tvec container all the transform without roation values untill now, we will use this to regain our actual values
    tvec = min_transform + current_position

    # scaling the grid to make it smaller or bigger depending the resolution we want, 
    # lower the better for calculatioins, but too low will lead to no path solution
    grid_left_lane_points  = scaled_points_on_grid(left_lane_t, scale)
    grid_right_lane_points = scaled_points_on_grid(right_lane_t, scale)
    grid_waypoint          = scaled_points_on_grid(waypoint_t, scale)
    grid_current_position  = scaled_points_on_grid(current_position_t, scale)
    

    # Since all the values are positive due to previous transformations, 
    # we can use the max values to get the size of the grid.
    # This allows us to make dynamic grid size based on the obsticles, goal, and start positions
    max = get_transform_max(left_lane_t, right_lane_t, current_position_t, waypoint_t)

    # creating the grid and populating it with obsticles
    grid_shape = (np.ceil(np.array(max)*scale)).astype(int)
    grid = np.zeros(grid_shape)
    grid[grid_left_lane_points[:, 0], grid_left_lane_points[:, 1]] = 1
    grid[grid_right_lane_points[:, 0], grid_right_lane_points[:, 1]] = 1
    grid[grid_waypoint[0], grid_waypoint[1]] = 6 # any value other than 1 is considered not an obsticle, this is purely for visualization
    grid[grid_current_position[0], grid_current_position[1]] = 2 # same as above
    
    # print(grid)
    # print(grid_waypoint)
    # print(grid_current_position)
    
    grid  = add_padding(grid, padding*scale)
    print(grid)

    path, cost = a_star(grid, heuristic, 
                            (grid_current_position[0], grid_current_position[1]) , 
                            (grid_waypoint[0], grid_waypoint[1]))
    if path is not None:
        path = np.array(path)
        pruned_path = prune_path(path)

        #transforming path to the original coordinate system
        pruned_path_t = (pruned_path / scale) + min_transform + current_position
        t = threading.Thread(target=plot, args=(pruned_path, grid, grid_current_position, grid_waypoint, tvec, scale))
        t.start()
        # t.join()
        return pruned_path_t[1]

    else: 
        return None

def add_padding(grid, padding):
    """makes grids adjacent to the occupied grid to be occupied based on the padding value"""
    occupanct_grids_xy = np.array(np.where(grid == 1)).astype(np.int16)
    padding = np.floor(padding).astype(np.int16)
    
    step = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    for i in range(occupanct_grids_xy.shape[1]):
        for s in step:
            try:
                if grid[occupanct_grids_xy[0, i] + s[0], occupanct_grids_xy[1, i] + s[1]] == 0:
                    grid[occupanct_grids_xy[0, i] + s[0], occupanct_grids_xy[1, i] + s[1]] = 1
            except IndexError:
                pass
    
    return grid


class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """


    UP        = (0, 1, 1)
    DOWN      = (0, -1, 1)
    LEFT      = (-1, 0, 1)
    RIGHT     = (1, 0, 1)
    UP_LEFT   = (-1, 1, 1.41421)
    UP_RIGHT  = (1, 1, 1.41421)
    DOWN_LEFT = (-1, -1, 1.41421)
    DOWN_RIGHT= (1, -1, 1.41421)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

    def __str__(self):
        return str(self.name)

def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    if x -1 < 0: valid_actions.remove(Action.LEFT)
    if x + 1 > n: valid_actions.remove(Action.RIGHT)
        
    if y - 1 < 0: valid_actions.remove(Action.DOWN)
    if y + 1 > m: valid_actions.remove(Action.UP)

    if x - 1 < 0 or y - 1 < 0: valid_actions.remove(Action.DOWN_LEFT)
    if x - 1 < 0 or y + 1 > m: valid_actions.remove(Action.UP_LEFT)
    if x + 1 > n or y - 1 < 0: valid_actions.remove(Action.DOWN_RIGHT)
    if x + 1 > n or y + 1 > m: valid_actions.remove(Action.UP_RIGHT)

    
    for actions in valid_actions:
        if grid[x + actions.delta[0], y + actions.delta[1]] == 1:
            # print(f"removeing {actions}")
            valid_actions.remove(actions)

    return valid_actions

def a_star(grid, h, start, goal):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        
        
        if current_node == goal:
            # print('Found a path.')
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node):
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])
                new_cost = current_cost + a.cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        return None, None

    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

class VehicleDecision():
    def __init__(self):
        self.vehicle_state = 'straight'
        self.lane_state = 0
        self.counter = 0
        self.lane_marker = None
        self.target_x = None
        self.target_y = None
        self.change_lane = False
        self.change_lane_wp_idx = 0
        self.detect_dist = 30
        self.speed = 20

        self.reachEnd = False

    def get_ref_state(self, currState, obstacleList, lane_marker, waypoint, boundary_lane_markers):
        """
            Get the reference state for the vehicle according to the current state and result from perception module
            Inputs:
                currState: [Loaction, Rotation, Velocity] the current state of vehicle
                obstacleList: List of obstacles
            Outputs: reference state position and velocity of the vehicle
        """
        self.lane_marker = lane_marker.lane_markers_center.location[-1]
        self.lane_state = lane_marker.lane_state

        # left_lane_location = np.array([[location.x, location.y] for location in left_lane.location])
        # right_lane_location = np.array([[location.x, location.y] for location in right_lane.location])

        # print(obstacleList)
        # [obstacle_name: "vehicle.bmw.isetta"
        # obstacle_id: 92
        # location: 
        # x: 165.4092559814453
        # y: -11.015679359436035
        # z: -0.01039806380867958
        # vertices_locations: 
        # - 
        #     vertex_location: 
        #     x: 166.48597717285156
        #     y: -10.19422435760498
        #     z: -0.648704469203949
        # - 
        #     vertex_location: 
        #     x: 166.44442749023438
        #     y: -10.226252555847168
        #     z: 0.7290442585945129
        # - 
        #     vertex_location: 
        #     x: 166.57626342773438
        #     y: -11.672050476074219
        #     z: -0.6803363561630249
        # - 
        #     vertex_location: 
        #     x: 166.5347137451172
        #     y: -11.704078674316406
        #     z: 0.697412371635437
        # - 
        #     vertex_location: 
        #     x: 164.28379821777344
        #     y: -10.327280044555664
        #     z: -0.7182085514068604
        # - 
        #     vertex_location: 
        #     x: 164.24224853515625
        #     y: -10.359308242797852
        #     z: 0.6595401763916016
        # - 
        #     vertex_location: 
        #     x: 164.37408447265625
        #     y: -11.805106163024902
        #     z: -0.7498404383659363
        # - 
        #     vertex_location: 
        #     x: 164.33253479003906
        #     y: -11.83713436126709
        #     z: 0.6279082894325256]
        
           
        if self.reachEnd:
            print("reached the finish line")
            return None
        if boundary_lane_markers.left_lane is not None and boundary_lane_markers.right_lane is not None:
            resp = plan(currState, 
                        waypoint.location,
                        boundary_lane_markers.left_lane, 
                        boundary_lane_markers.right_lane, 
                        obstacleList)

            if resp is not None:
                self.target_x = resp[0]
                self.target_y = resp[1]
            else:
                self.target_x = self.lane_marker.x
                self.target_y = self.lane_marker.y
        
        else:
            self.target_x = self.lane_marker.x
            self.target_y = self.lane_marker.y


        

        return [self.target_x, self.target_y, self.speed]


class VehicleController():
    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
        return newAckermannCmd

    def execute(self, currentPose, targetPose):
        """
            This function takes the current state of the vehicle and
            the target state to compute low-level control input to the vehicle
            Inputs:
                currentPose: ModelState, the current state of vehicle
                targetPose: The desired state of the vehicle
        """

        currentEuler = currentPose[1]
        curr_x = currentPose[0][0]
        curr_y = currentPose[0][1]

        target_x = targetPose[0]
        target_y = targetPose[1]
        target_v = targetPose[2]

        k_s = 0.1
        k_ds = 1
        k_n = 0.1
        k_theta = 1

        # compute errors
        dx = target_x - curr_x
        dy = target_y - curr_y
        xError = (target_x - curr_x) * np.cos(
            currentEuler[2]) + (target_y - curr_y) * np.sin(currentEuler[2])
        yError = -(target_x - curr_x) * np.sin(
            currentEuler[2]) + (target_y - curr_y) * np.cos(currentEuler[2])
        curr_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)
        vError = target_v - curr_v

        delta = k_n * yError
        # Checking if the vehicle need to stop
        if target_v > 0:
            v = xError * k_s + vError * k_ds
            #Send computed control input to vehicle
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = v
            newAckermannCmd.steering_angle = delta
            return newAckermannCmd
        else:
            return self.stop()

class LaneMarkers():
    def __init__(self, role_name="ego_vehicle") -> None:
        rospy.Subscriber("/carla/%s/left_lane_markers" % role_name, LaneList, self.left_lane_calback)
        rospy.Subscriber("/carla/%s/right_lane_markers" % role_name, LaneList, self.right_lane_calback)

        self.left_lane_markers = None
        self.right_lane_markers = None 

    def left_lane_calback(self, msg):
        self.left_lane_markers = np.array([[location.x, location.y] for location in msg.location])
    
    def right_lane_calback(self, msg):
        self.right_lane_markers = np.array([[location.x, location.y] for location in msg.location])
    
    @property
    def right_lane(self):
        return self.right_lane_markers
    
    @property
    def left_lane(self):
        return self.left_lane_markers


class Controller(object):
    """docstring for Controller"""
    def __init__(self):
        super(Controller, self).__init__()
        self.decisionModule = VehicleDecision()
        self.controlModule = VehicleController()
        self.boundary_lane_markers = LaneMarkers()


    def stop(self):
        return self.controlModule.stop()

    def execute(self, currState, obstacleList, lane_marker, waypoint):
        # Get the target state from decision module
        
        refState = self.decisionModule.get_ref_state(currState, obstacleList,
                                                     lane_marker, waypoint, self.boundary_lane_markers)

        if not refState:
            return None
        return self.controlModule.execute(currState, refState)
