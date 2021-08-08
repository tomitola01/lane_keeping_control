import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

v = 5
L = 2.3
u = 2*np.pi/180

def dynamics(t, z):
    theta = z[2]
    return [v*np.cos(theta), v*np.sin(theta), v*np.tan(u)/L]

num_points= 100
t_final = 2
z_initial = [0,0.3,5]
sol = solve_ivp(dynamics,
                [0, t_final],
                z_initial,
                t_eval=np.linspace(0, t_final, num_points))

x_points = sol.y[0]
y_points = sol.y[1]
theta_points = sol.y[2]
t_points = sol.t

# plt.plot(t_points, x_points)
# plt.plot(t_points, y_points)
plt.plot(x_points, y_points)
# plt.plot(t_points, theta_points)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()