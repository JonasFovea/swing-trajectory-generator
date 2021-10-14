# Idea Collection

## On path planning methods

+ Elliptical path *(see [2] p.103f)*

*see [1] p.203f*
+ Cubic or higher order polynomials
  - Position and speed of a joint can be defined for each point
  - Inverse Kinematics is used to convert from cartesian to angular coordinates
  - Velocities can also be chosen automatically by using a simple heuristic or setting a fixed linear acceleration
+ Linear paths with blends
+ Pseudo via points (like bezier curves)

All of these methods are generating joint angle functions $\Theta(t)$ describing a path between two points in space.



# Bibliography
1. Craig, J. J., 2005. Introduction to robotics : mechanics and control / John J. Craig. 3. ed. Pearson Prentice Hall. 
2. Paskarbeit, J., 2017. Consider the robot - Abstraction of bioinspired leg coordination and its application to a hexapod robot under consideration of technical constraints, Bielefeld: Universität Bielefeld.
