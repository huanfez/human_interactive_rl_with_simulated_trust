#!/usr/bin/env python

import rospy
from trust_rl_mrs.srv import episode, episodeResponse


def get_episode(req):
    curr_episode = req.val
    print "current episode:", curr_episode
    return "received"


if __name__ == "__main__":
    rospy.init_node("epidoe_server", anonymous=True)

    rospy.Service('/Episode', episode, get_episode)
    print "processes can go simultaneously"
    rospy.spin()
