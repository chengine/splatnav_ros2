#%%
import numpy as np
import clarabel
from scipy import sparse
import scipy
import time

qhull_pts = np.random.rand(100, 3)

convex_hull = scipy.spatial.ConvexHull(qhull_pts)
Ab = convex_hull.equations

A = Ab[:, :-1]
b = -Ab[:, -1]

test_point = np.random.rand(3) + 1.

n_constraints = A.shape[0]

tnow = time.time()
# Setup workspace
P = sparse.eye(3, format='csc')
A = sparse.csc_matrix(A)    
q = -2*test_point

settings = clarabel.DefaultSettings()
settings.verbose = False

solver = clarabel.DefaultSolver(P, q, A, b, [clarabel.NonnegativeConeT(n_constraints)], settings)
sol = solver.solve()
print('Time to solve:', time.time() - tnow)

# Check solver status
if str(sol.status) != 'Solved':
    print(f"Solver status: {sol.status}")
    print(f"Number of iterations: {sol.iterations}")
    print('Clarabel did not solve the problem!')
    solver_success = False
    solution = None
else:
    solver_success = True
    solution = sol.x

print('Closest point:', solution, 'Success?', solver_success)

# %%
