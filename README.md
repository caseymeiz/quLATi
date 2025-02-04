# quLATi

This package is for *Quantifying Uncertainty for Local Activation Time Interpolation*.

[![DOI](https://zenodo.org/badge/239559025.svg)](https://zenodo.org/badge/latestdoi/239559025)

It implements *Gaussian Process Manifold Interpolation (GPMI)* for doing Gaussian process regression on a manifold represented by a triangle mesh. See the paper [Gaussian process manifold interpolation for probabilistic atrial activation maps and uncertain conduction velocity](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2019.0345)

Since the code is focussed on GP regression on atrial meshes, the eigenfunction calculation routine makes use of specialized algorithms for mesh extension, subdivision, and gradient calculations. These could easily be modified for other use cases.


## How to install

It is probably best to set up a specialized anaconda environment for using quLATi:

We set up such an environment in this order:

```bash

conda create --name pylat

source activate pylat

conda install -c conda-forge numpy scipy matplotlib numba trimesh

```

The dependencies can be installed without anaconda using pip.

Then run the following command from the top directory of quLATi:

```bash

python setup.py install

```

My stuff

```shell
python3.11 -m  venv ./venv
source ./venv/bin/activate
pip install wheel
python setup.py bdist_wheel
```


## How to use

The workflow for using quLATi is:

* solve the eigenproblem for the mesh
* set the data and kernel
* optimize the hyperparameters
* make predictions

The atrial mesh must be defined by numpy arrays X (vertices/nodes) and Tri (triangles/faces, indexed from zero). The mesh must be manifold. (It may be desirable to create a new mesh - see the end of this README file for some useful routines).

```python
from qulati import gpmi, eigensolver
```

### Solve the eigenproblem

For a mesh with 5 holes (4 pulmonary veins and 1 mitral valve), where 10 layers of mesh elements are to be appeneded to the edges representing these holes, the Laplacian eigenvalue problem can be solved for the 256 smallest eigenvalues with:

```python
Q, V, gradV, centroids = eigensolver(X, Tri, holes = 5, layers = 10, num = 256)
```

`Q` are the 256 smallest eigenvalues, and `V` are the corresponding eigenfunction values at vertex and centroid locations. The gradient of the eigenfunctions at face `centroids` are given by `gradV`.

Note that this problem only needs solving a single time given a specific mesh, so results ought to be saved for future use.

The class for doing the interpolation can then be intialized with

```python
model = gpmi.Matern(X, Tri, Q, V, gradV, JAX = False)
```

If using `JAX = True` then optimization uses gradients of the loglikelihood, which are calculated using the JAX library (a soft dependency for quLATi).

NOTE: A fairly recent issue in JAX (which may now be solved) may cause issues, solveable by installing an earlier version of JAX. See here: https://github.com/google-research/long-range-arena/issues/7


### Set the data

For observations defined at vertices and centroids, data can be set using

```python
model.set_data(obs, vertices, obs_err_stdev)
```

where `vertices` is a zero indexed integer array referencing which vertex or face centroid an observation belongs to. Observations can be assigned to vertices with indices `0:X.shape[0]` and assigned to face centroids with indices `X.shape[0] + Tri_index` (where `Tri_index` refers to faces defined in `Tri`).

For the Matern kernel class, the kernel smoothness must be set:

```python
model.kernelSetup(smoothness = 3./2.)
```

### Optimization

To optimize the kernel hyperparameters:

```python
# optimize the nugget
model.optimize(nugget = None, restarts = 5)

# fix the nugget to 0.123
#model.optimize(nugget = 0.123, restarts = 5)
```


### Predictions

Predictions at vertices and centroids can be obtained with:

```python
pred_mean, pred_stdev = model.posterior(pointwise = True)
```

where the posterior mean and standard deviation are returned (vertex predictions are indexed `0:X.shape[0]`, centroid predictions are indexed `X.shape[0]:(X.shape[0] + Tri.shape[0]`).

To calculate the posterior for a subset of indices, such as the vertices only, use:

```python
pred_mean, pred_stdev = model.posterior(indices = range(X.shape[0]), pointwise = True)
```

If `pointwise = False` is used, then the full posterior covariance is calculated and stored (not returned from this function). Then `num` posterior samples can be generated using:

```python
samples = model.posteriorSamples(num, nugget = 1e-10)
```

where `nugget` is used to stabilize calculations with the posterior variance matrix (in other words, it is not used to simulate noisy observations).


The posterior mean of the gradient for face centroids, calculated from the gradients of the eigenfunctions, can be obtained with

```python
grad_pred = model.posteriorGradient()
```

Statistics on the magnitude of the posterior gradient can be obtained with

```python
# for all face centroids
mag_stats = model.gradientStatistics()

# for specific centroids given by centroid_index
#mag_stats = model.gradientStatistics(centroid_index)
```


## Remeshing

quLATi also has some useful remeshing utilities, designed to create a lower resolution mesh from a higher resolution mesh.

Use simulated annealing to choose a subset of well-spaced mesh vertices, and then triangulate these new points.

```python
from qulati.meshutils import subset_anneal, subset_triangulate, extendMesh

# simulated annealing (suggestion: continue until low percentage of successful new moves)
NUM = 2000
RUNS = 25000

cont = True
first_run = True
while cont:

    if first_run:
        choice = subset_anneal(X, Tri, num = NUM, runs = RUNS)
        first_run = False
    else:
        choice = subset_anneal(X, Tri, num = NUM, runs = RUNS, choice = choice)

    inp = input("Type Y or y to do more annealing: ")
    if (inp != "Y") and (inp != "y"):
        cont = False
        
# create triangulation with new points (note: mesh extension helps remeshing)
newX, newTri = subset_triangulate(X, Tri, choice, layers = 5, holes = 5)

```

