import numpy as np
import trimesh
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from qulati import gpmi, eigensolver
import matplotlib.pyplot as plt

# Sine function
def sine_wave(x, y, amplitude, frequency_x, frequency_y):
    return amplitude * np.sin(2 * np.pi * (frequency_x * x + frequency_y * y))

grid_size = 100
x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
z = sine_wave(x, y, amplitude=0.1, frequency_x=4, frequency_y=4)
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
vert = np.array([7277, 6312])
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
    # text index of vertices
    text=pred_mean,
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
