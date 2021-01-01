import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 
import robots, locomotion_tools, plotting_tools, measurement 


if __name__=='__main__':
    robot = robots.load_ankled_biped_pinocchio()
    contact_names = ['LF_heel', 'LF_toe', 'RF_heel', 'RF_toe']
    x0 = np.hstack([np.resize(
        robot.model.referenceConfigurations["reference"], robot.nq), np.zeros(robot.nv)])
    robot.initViewer(loadModel=True)
    cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
    robot.viewer.gui.setCameraTransform(0, cameraTF)
    backgroundColor = [1., 1., 1., 1.]
    floorColor = [0.7, 0.7, 0.7, 1.]
    #   
    window_id = robot.viz.viewer.gui.getWindowID("python-pinocchio")
    robot.viz.viewer.gui.setBackgroundColor1(window_id, backgroundColor)
    robot.viz.viewer.gui.setBackgroundColor2(window_id, backgroundColor)
    robot.display(x0[:robot.nq, None])




