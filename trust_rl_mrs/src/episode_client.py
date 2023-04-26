#!/usr/bin/env python

import rospy
from trust_rl_mrs.srv import episode, episodeResponse


if __name__ == "__main__":
    rospy.init_node("episode_client", anonymous=True)
    rate = rospy.Rate(2)

    rospy.wait_for_service('/Episode')
    send_episode = rospy.ServiceProxy('/Episode', episode)

    iteration = 0
    while iteration < 5:
        resp = send_episode(iteration)
        print resp.response
        iteration += 1
        rate.sleep()

    rospy.spin()
