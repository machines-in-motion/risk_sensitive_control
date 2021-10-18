
# Risk Sensitive Optimal Control Library


## Dependencies

1. [pinocchio](https://github.com/stack-of-tasks/pinocchio)
2. [crocoddyl](https://github.com/loco-3d/crocoddyl)
3. [simple_simulator](https://github.com/hammoudbilal/simple_simulator)
4. [robot_properties_solo](https://github.com/open-dynamic-robot-initiative/robot_properties_solo)


Note: simple_simulator could be replaced by any other simulation environment


## Reference Trajectories

Reference trajectories for the quadruped were optimized using the [Kinodynamic Opimization Library](https://github.com/machines-in-motion/kino_dynamic_opt), the references are added to this repository but if you wish to generate your own feel free to modify the yaml files in `python/risc/demos/solo12/planner/config`


## Running Demos

A good starting point might be at `risk_sensitive_control/python/risc/demos/cliff`. In the cliff demo, the optimal control problem is that of a point mass crossing a cliff. This problem is adopted from [Farshidian](https://arxiv.org/pdf/1512.07173.pdf) for its simplicity. More complicated demos involving locomotion can be found in `risk_sensitive_control/python/risc/demos/solo`.  We setup optimal control problems accroding to `crocoddyl` convention while adding noise models. The main solver code is in `risk_sensitive_control/python/risc/solvers/risc.py`. 

## Author 

1. Bilal Hammoud (<bah436@nyu.edu>)