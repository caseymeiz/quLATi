import numpy as np
import trimesh
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from qulati import gpmi, eigensolver
import matplotlib.pyplot as plt

# Gaussian function
def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    exponent = -((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2))
    return np.exp(exponent)

grid_size = 10
x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
z = gaussian(x, y, mu_x=0.5, mu_y=0.5, sigma_x=0.2, sigma_y=0.2)
vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
tri = Delaunay(vertices[:, :2])
faces = tri.simplices

# Call the eigensolver function
num_eigenpairs = 2**8
Q, V, gradV, centroids = eigensolver(vertices, faces, holes=0, layers=1, num=10)

model = gpmi.Matern(vertices, faces, Q, V, gradV, JAX=False)
model.kernelSetup(smoothness=3.5)

# fit model to data
vert_val = np.array([4, 2])
vert = np.array([25, 75])
model.set_data(vert_val, vert)
model.optimize(nugget=0.123, restarts=5)
pred_mean, pred_stdev = model.posterior(pointwise=True)

# Create a Plotly Mesh3d object for the mesh
mesh = go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    intensity=pred_mean,
    intensitymode="vertex",
    colorscale="Viridis",
    flatshading=True,
)
# Create a Plotly Scatter3D object for vert_val points using vert as an index
data = go.Scatter3d(
    x=vertices[vert, 0],
    y=vertices[vert, 1],
    z=vertices[vert, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="black",
    ),
)

# Create a 3D figure
fig = go.Figure(data=[mesh, data])

# Set the axis labels
fig.update_layout(
    scene=dict(
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        zaxis_title="Z-axis",
    ),
)

# Show the plot
fig.show()
