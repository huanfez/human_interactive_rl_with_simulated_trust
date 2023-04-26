#!/usr/bin/env python
import rospy
from tf.transformations import quaternion_from_euler
import actionlib
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseWithCovarianceStamped
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellResult
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from src.trust_rl_mrs.src.env_abstraction import setting_environment as env
import numpy as np


# Select the point with the best LoS in next cell as the target
def gen_space_target2(next_cell_x, next_cell_y, dsm_map=env.dsm_img, margin_width=10, margin_height=10):
    """
    Given a target cell space, select the position that has the lowest height in the dsm image.
    For example, (1) locate a size of 5 * 5, (2) remove the 'x' which is the margin area, (3) locate the '0'
    which has the lowest value in dsm
        X X X X X
        X - - - X
        X - o - X
        X - - - X
        X X X X X
    """

    # locate the pixels in the next cell
    start_window_x, start_window_y = next_cell_x * env.cell_width, next_cell_y * env.cell_height
    end_window_x, end_window_y = start_window_x + env.cell_width, start_window_y + env.cell_height

    # remove the boundary area with width margin_width, margin_height
    start_window_x_, end_window_x_ = start_window_x + margin_width, end_window_x - margin_width
    start_window_y_, end_window_y_ = start_window_y + margin_height, end_window_y - margin_height

    # Obtain the target area dsm information
    dsm_img = np.asarray(dsm_map)
    node_dict = {(px, py): dsm_img[py][px] for px in range(start_window_x_, end_window_x_)
                 for py in range(start_window_y_, end_window_y_)}

    key_min = min(node_dict, key=node_dict.get)  # select the minimum dsm one as the target position
    target_env_pos = env.imgPos2envPos(np.array([[key_min[0]], [key_min[1]]]))  # change it into geological position
    print "target pixel:", target_env_pos

    return target_env_pos


def movebase_client(targetx, targety, target_ori, robot_name='husky_alpha'):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'odom'
    goal.target_pose.header.stamp = rospy.Time.now()
    # Move 0.5 meters forward along the x axis of the "map" coordinate frame
    goal.target_pose.pose.position.x = targetx
    goal.target_pose.pose.position.y = targety
    # No rotation of the mobile base frame w.r.t. map frame
    goal.target_pose.pose.orientation.x = target_ori[0]
    goal.target_pose.pose.orientation.y = target_ori[1]
    goal.target_pose.pose.orientation.z = target_ori[2]
    goal.target_pose.pose.orientation.w = target_ori[3]

    # Sends the goal to the action server.
    client.send_goal(goal)
    # Waits for the server to finish performing the action.
    wait = client.wait_for_result()
    # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        # Result of executing the action
        return client.get_result()


def update_odom():
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_husky_base_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        husky_base_link_state = get_husky_base_link_state("/::base_link", "world")

        set_huksy_odom_state = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=10)
        huksy_reset_state = PoseWithCovarianceStamped()
        huksy_reset_state.header.frame_id = 'odom'
        huksy_reset_state.pose.pose = husky_base_link_state.link_state.pose
        set_huksy_odom_state.publish(huksy_reset_state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def execExplore(goal_cells):
    next_cell_x, next_cell_y = goal_cells.to_cell_x, goal_cells.to_cell_y
    next_cell_target_pos = gen_space_target2(next_cell_x, next_cell_y)
    target_ori = quaternion_from_euler(0.0, 0.0, np.arctan2(-(next_cell_y - goal_cells.in_cell_y),
                                                            next_cell_x - goal_cells.in_cell_x))
    # print "alpha's goal cell:", next_cell_x, next_cell_y, "target ori:", target_ori
    update_odom()
    move_base_result = movebase_client(next_cell_target_pos[0][0],
                                       next_cell_target_pos[1][0],
                                       target_ori, robot_name='')
    # - common_parameters.husky_alpha_init.x - common_parameters.husky_alpha_init.y

    result = NeighborCellResult()
    result.at_cell_x = next_cell_x
    result.at_cell_y = next_cell_y
    print "!!! REMINDER: autonomous robots reached their temporary goal. " \
          "Please provide TRUST change with the HCI !!! ->->->\n"
    cellExploreServer.set_succeeded(result, "temporary target reached")


if __name__ == '__main__':
    try:
        rospy.init_node('robot_alpha_path_executor', anonymous=True)
        update_odom()
        cellExploreServer = actionlib.SimpleActionServer('/server1_localcells', NeighborCellAction, execExplore, False)
        cellExploreServer.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
