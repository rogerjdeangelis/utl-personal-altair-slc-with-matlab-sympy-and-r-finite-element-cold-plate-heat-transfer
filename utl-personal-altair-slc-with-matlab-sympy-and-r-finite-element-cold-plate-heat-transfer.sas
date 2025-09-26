%let pgm=utl-personal-altair-slc-with-matlab-sympy-and-r-finite-element-cold-plate-heat-transfer;

%stop_submission;

Personal altair slc with matlab sympy and r finite element cold plate heat transfer

Too log to post here, see github

see github
https://tinyurl.com/3f9w97t2
https://github.com/rogerjdeangelis/utl-personal-altair-slc-with-matlab-sympy-and-r-finite-element-cold-plate-heat-transfer

Personal altair slc with matlab sympy and r finite element cold plate heat transfer

 CONTENTS (very simplistic examples - out of my comfort zone on mach of this)

   1 slc python matlab (via open source Octave)
     This is how you execute matlab code inside the altair slc.
     oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")

   2 slc smypy SLC, Exact solution to the one dimensional heat transfer problem,
     I do realize very few heat transfer problems have closed form solution
     but intial estimates bases on a closed form can be useful?

   3 slc r interative solution

     graphics
     https://tinyurl.com/ms3thwep

     unfortunately Altair SLC dos not support ascii contour plots
     heatmap https://tinyurl.com/5n97pucx

     options ls=64 ps=32;
     proc plot data=wantxpox ;

     plot y*x=z / contour=5  box;

     ERROR: Option "contour" is not known for the PLOT statement
     In the spirit of John Tukey, manual editable ascii graphis provides insights to data.
      Maybe Seimens will bring ascii grahics up to the sas level.

      Also parmcards4 is critical for integration, maybe add parmcards to the personal SLC.

   4 slc r finite element solution cold plate

     Graphics

     https://tinyurl.com/yeyjwtk6
     https://tinyurl.com/ywxmp4yb
     https://tinyurl.com/3n6zsy8f


   5 Repository categories (although many of the categories involve sas, it should be easy to convert to the SLC)

       SYPY
       AI
       MATLAB
       POWERSHELL
       SPSS
       POSTGREQL
       MYSQL
       SQLITE
       EXCEL

3 slc r interative solution
---------------------------
https://tinyurl.com/5n97pucx
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/heatequ.png

github
https://tinyurl.com/2dxceary
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python


4 slc r finite element solution cold plate
------------------------------------------
https://tinyurl.com/yeyjwtk6
https://github.com/rogerjdeangelis/utl-personal-altair-slc-with-matlab-sympy-and-r-finite-element-cold-plate-heat-transfer/blob/main/fea1.png

https://tinyurl.com/ywxmp4yb
https://github.com/rogerjdeangelis/utl-personal-altair-slc-with-matlab-sympy-and-r-finite-element-cold-plate-heat-transfer/blob/main/fea2.png

https://tinyurl.com/3n6zsy8f
https://github.com/rogerjdeangelis/utl-personal-altair-slc-with-matlab-sympy-and-r-finite-element-cold-plate-heat-transfer/blob/main/fea3.png

/*       _                    _   _                                  _   _       _
/ |  ___| | ___   _ __  _   _| |_| |__   ___  _ __   _ __ ___   __ _| |_| | __ _| |__
| | / __| |/ __| | `_ \| | | | __| `_ \ / _ \| `_ \ | `_ ` _ \ / _` | __| |/ _` | `_ \
| | \__ \ | (__  | |_) | |_| | |_| | | | (_) | | | || | | | | | (_| | |_| | (_| | |_) |
|_| |___/_|\___| | .__/ \__, |\__|_| |_|\___/|_| |_||_| |_| |_|\__,_|\__|_|\__,_|_.__/
                 |_|    |___/
*/

PROBLEM:
   Minimize yjr objective exfression
   (x(1)-3)^2 + (x(2)-5)^2
   Obviously the minimum is 3 and 5 for x(1) and x(2)

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Run the entire optimization as an Octave command
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")
oc.eval("result = fminsearch(objfun, [0, 0]);  ")

# Retrieve the result
result = oc.pull("result")
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;

OUTPUT
======

The PYTHON Procedure

Optimal solution found by Octave:

[[2.99995267 4.99983461]]

/*___                                      _     _  _                _          __
|___ \   ___ _   _ _ __ ___  _ __  _   _  / | __| || |__   ___  __ _| |_ __  __/ _| ___ _ __
  __) | / __| | | | `_ ` _ \| `_ \| | | | | |/ _` || `_ \ / _ \/ _` | __|\ \/ / |_ / _ \ `__|
 / __/  \__ \ |_| | | | | | | |_) | |_| | | | (_| || | | |  __/ (_| | |_  >  <|  _|  __/ |
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_|\__,_||_| |_|\___|\__,_|\__|/_/\_\_|  \___|_|
             |___/          |_|    |___/
*/

   EXPLANATION

   One dimensional heat distribution along a length of coldplate

           /    -T_in + T(x) \
         P*|1 - -------------|
   dT      \    -T_in + T_out/
   -- =  ---------------------
   dx         L*c_p*m_dot

   Initial Condition

   T_in
   T_out
   L
   m_dot
   c_p
   P

   EASY TO SOLVE BECAUSE
   ======================

      dt
      -- = t(x)   (everthing else in the heat equation are constants)
      dx
           1
    Then  --- dt = dx
           t

    Integrate both sides

       / 1       /
       \ - dt  = \ dx
       / t       /

      ln(t)    =  x + c (c=constent of integration)

      Lets exponentiate both sides

                      x
         t(x)  = c * e

   LETS LET SYMPY DO THE MESSY ALGEBRA
   ====================================

                                      P*x
                           --------------------------
                           L*c_p*m_dot*(T_in - T_out)
   T_out + (T_in - T_out)*e



&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
import sympy as sp
from sympy import symbols, exp, pi, sqrt, integrate, diff, simplify, pprint, erf

# Define symbols
x = sp.Symbol('x')
T = sp.Function('T')(x)
T_in, T_out, L, m_dot, c_p, P, h = sp.symbols('T_in T_out L m_dot c_p P h')

# Define the differential equation
dT_dx = (P / (m_dot * c_p * L)) * (1 - (T - T_in) / (T_out - T_in))
sp,pprint(dT_dx)
eq = sp.Eq(T.diff(x), dT_dx)

# Solve the differential equation
solution = sp.dsolve(eq, T, ics={T.subs(x, 0): T_in})

# Simplify the solution
simplified_solution = sp.simplify(solution.rhs)

print("Solution:")
sp.pprint(simplified_solution)
endsubmit;
;quit;run;

OUTPUT
======

Altair SLC

Solution:
                                   P*x
                        --------------------------
                        L*c_p*m_dot*(T_in - T_out)
T_out + (T_in - T_out)*e


/*____       _               ____     _             _     _        _       _
|___ /   ___| | ___   _ __  |___ \ __| |   ___ ___ | | __| | _ __ | | __ _| |_ ___
  |_ \  / __| |/ __| | `__|   __) / _` |  / __/ _ \| |/ _` || `_ \| |/ _` | __/ _ \
 ___) | \__ \ | (__  | |     / __/ (_| | | (_| (_) | | (_| || |_) | | (_| | ||  __/
|____/  |___/_|\___| |_|    |_____\__,_|  \___\___/|_|\__,_|| .__/|_|\__,_|\__\___|
                                                            |_|
*/

&_init_;
options noerrorabend;
options set=RHOME "D:\d451";
%utl_fkil(d:/png/heatequ.png);
proc r;
submit;
# Load necessary library
library(haven)
# Parameters
Lx <- 1       # Length of the plate in x-direction
Ly <- 1       # Length of the plate in y-direction
Nx <- 20      # Number of grid points in x-direction
Ny <- 20      # Number of grid points in y-direction
dx <- Lx / (Nx - 1)
dy <- Ly / (Ny - 1)
alpha <- 0.01 # Thermal diffusivity
dt <- 0.001   # Time step
Nt <- 100     # Number of time steps

# Initialize temperature grid
u <- matrix(0, nrow = Nx, ncol = Ny)

# Initial condition: some function f(x,y)
f <- function(x, y) {
  return(sin(pi * x / Lx) * sin(pi * y / Ly))
}

# Apply initial condition
for (i in seq_len(Nx)) {
  for (j in seq_len(Ny)) {
    x <- (i - 1) * dx
    y <- (j - 1) * dy
    u[i, j] <- f(x, y)
  }
}
# Time-stepping loop
for (n in seq_len(Nt)) {
  u_new <- u
  for (i in 2:(Nx-1)) {
    for (j in 2:(Ny-1)) {
      u_new[i, j] <- u[i, j] + alpha * dt * (
        (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 +
        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2
      )
    }
  }
  u <- u_new
}
str(u)
u <- as.matrix(u)
png("d:/png/heatequ.png")
# Plot the final temperature distribution
image(u, main="Temperature Distribution", xlab="X", ylab="Y", col=heat.colors(100))
dev.off()
rwant=as.data.frame(u)
rwant$id <- 1:nrow(rwant)
head(rwant)
endsubmit;
import data=rwant r=rwant;
run;quit;

&_init_;
proc transpose data=rwant out=wantxpo;
by id;
run;quit;

data wantxpox;
 set wantxpo;
 y=input(substr(_name_,2),3.);
 x=id;
 z=col1;
 drop _name_ col1;
run;quit;

proc print data=rwant;
format _numeric_ 4.2;
run;quit

/*--- does not work
options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;

/*  _         _         __ _       _ _             _                           _
| || |    ___| | ___   / _(_)_ __ (_) |_ ___   ___| | ___ _ __ ___   ___ _ __ | |_
| || |_  / __| |/ __| | |_| | `_ \| | __/ _ \ / _ \ |/ _ \ `_ ` _ \ / _ \ `_ \| __|
|__   _| \__ \ | (__  |  _| | | | | | ||  __/|  __/ |  __/ | | | | |  __/ | | | |_
   |_|   |___/_|\___| |_| |_|_| |_|_|\__\___| \___|_|\___|_| |_| |_|\___|_| |_|\__|

Example of solving a cold plate heat transfer problem using
Finite Element Analysis (FEA) in R. This example models a simple rectangular plate
with a cold boundary condition on one edge and a heat source in the center.
*/

&_init_;
proc r;
submit;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages

library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Shape functions for linear triangles
shape_functions <- function(xi, eta) {
  N <- c(1 - xi - eta, xi, eta)
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  return(list(N = N, dN_dxi = dN_dxi))
}

# Element stiffness matrix for heat conduction
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)
  invJ <- solve(J)

  # Shape function derivatives in global coordinates
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  dN_dx <- invJ %*% dN_dxi

  # Element stiffness matrix
  ke <- matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      ke[i, j] <- k * detJ * t(dN_dx[, i]) %*% dN_dx[, j] / 2
    }
  }

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)
T_ambient <- 20 # Ambient temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh
mesh <- create_rectangular_mesh(Lx, Ly, nx = 30, ny = 15)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = temperature
)


# Contour plot
fea1<-ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_contour_filled(bins = 20) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_d(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()
ggsave(filename="d:/png/fea1.png",plot=fea1)

# 3D surface plot
fea2<-ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_raster(aes(fill = temperature), interpolate = TRUE) +
  geom_contour(color = "white", alpha = 0.5, bins = 15) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution - Surface Plot",
       x = "X (m)", y = "Y (m)") +
  theme_minimal()

# Temperature profile along centerline
centerline_nodes <- which(mesh$nodes$y == Ly/2)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggsave(filename="d:/png/fea2.png",plot=fea2)

fea3<-ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()
ggsave(filename="d:/png/fea3.png",plot=fea3)


# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")

# Display temperature at specific points
cat("\nTemperature at key locations:\n")
cat("Center of plate (x=0.5, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 0.5) + abs(mesh$nodes$y - 0.25))], "°C\n")
cat("Right edge center (x=1.0, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 1.0) + abs(mesh$nodes$y - 0.25))], "°C\n")

endsubmit;
run;quit;

OUTPUT
======

Altair SLC

Assembling stiffness matrix...
Assembling load vector...
Applying boundary conditions...
Solving linear system...
=== SOLUTION SUMMARY ===
Minimum temperature: -0.0181723 °C
Maximum temperature: 0.1462442 °C
Cold boundary nodes: 15
Total nodes: 450
Total elements: 812
Temperature at key locations:
Center of plate (x=0.5, y=0.25): 0.1416964 °C
Right edge center (x=1.0, y=0.25): 8.317972e-08 °C

/*___                                     _                        _
| ___|   _ __ ___ _ __   ___     ___ __ _| |_ ___  __ _  ___  _ __(_) ___  ___
|___ \  | `__/ _ \ `_ \ / _ \   / __/ _` | __/ _ \/ _` |/ _ \| `__| |/ _ \/ __|
 ___) | | | |  __/ |_) | (_) | | (_| (_| | ||  __/ (_| | (_) | |  | |  __/\__ \
|____/  |_|  \___| .__/ \___/   \___\__,_|\__\___|\__, |\___/|_|  |_|\___||___/
                 |_|                              |___/
*/

SYMPY
-----------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-area-between-curves-with-an-intersection-point-adding-negative-and-positive-areas-plot-sympy
https://github.com/rogerjdeangelis/utl-calculating-the-cube-root-of-minus-one-with-drop-down-to-python-symbolic-math-sympy
https://github.com/rogerjdeangelis/utl-closed-form-solution-for-sample-size-in-a-clinical-equlivalence-trial-using-r-and-sas-and-sympy
https://github.com/rogerjdeangelis/utl-distance-between-a-point-and-curve-in-sql-and-wps-pythony-r-sympy
https://github.com/rogerjdeangelis/utl-fun-with-sympy-infinite-series-and-integrals-to-define-common-functions-and-constants
https://github.com/rogerjdeangelis/utl-maximum-likelihood-estimate-of--therate-parameter-lamda-of-a-Poisson-distribution-sympy
https://github.com/rogerjdeangelis/utl-maximum-liklihood-regresssion-wps-python-sympy
https://github.com/rogerjdeangelis/utl-mle-symbolic-solution-for-mu-and-sigma-of-normal-pdf-using-sympy
https://github.com/rogerjdeangelis/utl-python-sympy-projection-of-the-intersection-of-two-parabolic-surfaces-onto-the-xy-plane-AI
https://github.com/rogerjdeangelis/utl-r-python-compute-the-area-between-two-curves-AI-sympy-trapezoid
https://github.com/rogerjdeangelis/utl-roots-of-a-non-linear-function-using-python-sympy
https://github.com/rogerjdeangelis/utl-solve-a-system-of-simutaneous-equations-r-python-sympy
https://github.com/rogerjdeangelis/utl-symbolic-algebraic-simplification-of-a-polynomial-expressions-sympy
https://github.com/rogerjdeangelis/utl-symbolic-solution-for-the-gradient-of-the-cumulative-bivariate-normal-using-erf-and-sympy
https://github.com/rogerjdeangelis/utl-symbolically-solve-for-the-mean-and-variance-of-normal-density-using-expected-values-in-SymPy
https://github.com/rogerjdeangelis/utl-sympy-exact-pdf-and-cdf-for-the-correlation-coefficient-given-bivariate-normals
https://github.com/rogerjdeangelis/utl-sympy-technique-for-symbolic-integration-of-bivariate-density-function
https://github.com/rogerjdeangelis/utl-using-python-sympy-for-mathematical-characterization-of-the-human-face
https://github.com/rogerjdeangelis/utl-vertical-distance-covered-by-a-bouncing-ball-for-infinite-number-of-bounces-using-sympy

AI
-----------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-AI-compute-the-distance-between-objects-in-an-image-python
https://github.com/rogerjdeangelis/utl-AI-computer-vision-fitting-a-circle-to-a-scatter-plot-of-points-wps-r
https://github.com/rogerjdeangelis/utl-AI-first-name-and-birth-date-range-to-gender
https://github.com/rogerjdeangelis/utl-AI-geometry-is-the-figure-a-right-triangle-
https://github.com/rogerjdeangelis/utl-AI-internal-angles-of-polygon-from-vertex-coordinates-in-r
https://github.com/rogerjdeangelis/utl-AI-labeling-centroids-inside-and-within-a-boundary-polygon
https://github.com/rogerjdeangelis/utl-AI-remove-noise-from-an-image-python-opencv
https://github.com/rogerjdeangelis/utl-AI-spelling-corrector-when-word-is-close-to-the-correct-word
https://github.com/rogerjdeangelis/utl-R-AI-igraph-list-connections-in-a-non-directed-graph-for-a-subset-of-vertices
https://github.com/rogerjdeangelis/utl-capturing-old-faithful-before-and-during-an-eruption--AI-visual-analytics
https://github.com/rogerjdeangelis/utl-determinating-gender-from-firstname-AI-sas-r-and-python
https://github.com/rogerjdeangelis/utl-did-shakespeare-write-hamlet-nlp-of-shakespeare-plays-AI-natural-language-processing
https://github.com/rogerjdeangelis/utl-finding-the-syllables-of-words-AI-NLP
https://github.com/rogerjdeangelis/utl-formatting-ai-seacrh-output-in-pdf-rtf-and-excel-format-perplexity-chatGPT-results
https://github.com/rogerjdeangelis/utl-identify-the-outer-most-points-in-a-graph-object-ai-convex-hulls
https://github.com/rogerjdeangelis/utl-python-AI-color-frequencies-in-an-image
https://github.com/rogerjdeangelis/utl-r-compute-the-area--of-an-image-which-is-under-a-curve-AI-image-processing-AI
https://github.com/rogerjdeangelis/utl-r-python-compute-the-area-between-two-curves-AI-sympy-trapezoid
https://github.com/rogerjdeangelis/utl-scraping-AI-results-without-restriction-or-API-with-powershell-and-perplexity
https://github.com/rogerjdeangelis/utl-simple-three-letter-commands-to-format-perplexity-AI-results-for-word-pdf-text-and-excel


MATLAB
-------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-bringing-matlab-into-the-larger-family-of-computer-languages
https://github.com/rogerjdeangelis/utl-convert-a-sqlite-numeric-table-to-a-matrix-for-octave-matlab-processing
https://github.com/rogerjdeangelis/utl-how-to-store-octave-matlab-objects-in-external-files-for-later-use-with-octave-r-and-python
https://github.com/rogerjdeangelis/utl-matlab-combine-every-two-rows-ito-one-and-and-add-a-prefix-to-duplicate-column-names
https://github.com/rogerjdeangelis/utl-matlab-select-all-possible-pairs-and-and-use-octave-sqlwrite-to-return-values-to-sas
https://github.com/rogerjdeangelis/utl-octave-matlab-check-if-all-of-the-rows-of-a-dbtable-are-the-same
https://github.com/rogerjdeangelis/utl-octave-matlab-deleting-rows-where-age-is-zero
https://github.com/rogerjdeangelis/utl-randomly-pick-one-player-from-the-heat-and-suns-for-captains-sql-sas-r-python-matlab
https://github.com/rogerjdeangelis/utl-runing-a-regression-using-matlab-syntax-using-the-open-source-r-octave-package

POWERSHELL
------------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-bringing-matlab-into-the-larger-family-of-computer-languages
https://github.com/rogerjdeangelis/utl-convert-a-sqlite-numeric-table-to-a-matrix-for-octave-matlab-processing
https://github.com/rogerjdeangelis/utl-how-to-store-octave-matlab-objects-in-external-files-for-later-use-with-octave-r-and-python
https://github.com/rogerjdeangelis/utl-matlab-combine-every-two-rows-ito-one-and-and-add-a-prefix-to-duplicate-column-names
https://github.com/rogerjdeangelis/utl-matlab-select-all-possible-pairs-and-and-use-octave-sqlwrite-to-return-values-to-sas
https://github.com/rogerjdeangelis/utl-octave-matlab-check-if-all-of-the-rows-of-a-dbtable-are-the-same
https://github.com/rogerjdeangelis/utl-octave-matlab-deleting-rows-where-age-is-zero
https://github.com/rogerjdeangelis/utl-randomly-pick-one-player-from-the-heat-and-suns-for-captains-sql-sas-r-python-matlab
https://github.com/rogerjdeangelis/utl-runing-a-regression-using-matlab-syntax-using-the-open-source-r-octave-package
https://github.com/rogerjdeangelis/utl-sas-utility-functions-and-the-lack-of-support-for-utility-functions-r-python-and-matlab

SPSS
------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-connecting-spss-pspp-to-postgresql-sample-problem-compute-mean-weight-by-sex
https://github.com/rogerjdeangelis/utl-creating-spss-tables-from-a-sas-datasets-using-sas-r-and-python
https://github.com/rogerjdeangelis/utl-dropping-down-to-spss-using-the-pspp-free-clone-and-running-a-spss-linear-regression
https://github.com/rogerjdeangelis/utl-identifying-the-html-table-and-exporting-to-spss-then-sas-scraping
https://github.com/rogerjdeangelis/utl-import-dbf-dif-ods-xlsx-spss-json-stata-csv-html-xml-tsv-files-without-sas-access-products
https://github.com/rogerjdeangelis/utl-removing-factors-and-preserving-type-and-length-when-importing-spss-sav-tables
https://github.com/rogerjdeangelis/utl-sas-to-and-from-sqllite-excel-ms-access-spss-stata-using-r-packages-without-sas
https://github.com/rogerjdeangelis/utl-using-open-source-pspp-to-convert-spss-programs-to-sas-or-other-languages
https://github.com/rogerjdeangelis/utl-using-sas-compatible-character-and-numeric-missing-values-in-spss-pspp

POSTGRESQL
---------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-clumsily-done-in-sas-sql-and-eligantly-done-in-postgresql-simply-count-distinct-by-groupings
https://github.com/rogerjdeangelis/utl-connecting-spss-pspp-to-postgresql-sample-problem-compute-mean-weight-by-sex
https://github.com/rogerjdeangelis/utl-creating-sqlite-and-postgresql-tables-from-sas-datasets-without-sas-access-and-a-blueprint
https://github.com/rogerjdeangelis/utl-loading-tiny-one-million-row-sas-dataset-into-postgres-db-sql-and-selecting-distinct-values
https://github.com/rogerjdeangelis/utl-partial-key-matching-and-luminosity-in-gene-analysis-sas-r-python-postgresql
https://github.com/rogerjdeangelis/utl-pivot-wide-when-variable-names-contain-values-sql-and-base-r-sas-oython-excel-postgreSQL
https://github.com/rogerjdeangelis/utl-saving-and-creating-r-dataframes-to-and-from-a-postgresql-database-schema

MYSQL
-------------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/mySQL-uml-modeling-to-create-entity-diagrams-for-sas-datasets
https://github.com/rogerjdeangelis/utl-PASSTHRU-to-mysql-and-select-rows-based-on-a-SAS-dataset-without-loading-the-SAS-daatset-into-my
https://github.com/rogerjdeangelis/utl-accessing-a-mysql-database-using-R-without-the-sas-access-product
https://github.com/rogerjdeangelis/utl-mysql-queries-without-sas-using-r-python-and-wps
https://github.com/rogerjdeangelis/utl-rename-and-cast-char-to-numeric-using-do-over-arrays-in-natve-and-mysql-wps-r-python
https://github.com/rogerjdeangelis/utl_examples_SAS_mysql_interface_on_power_workstation
https://github.com/rogerjdeangelis/utl_explicit_pass_through_to_mysql_to_subset_a_table_using_macro_variable_dates
https://github.com/rogerjdeangelis/utl_exporting_a_sas_single_60k_string_to_mysql_and_reading_it_back_into_two_30k_strings
https://github.com/rogerjdeangelis/utl_extract_from_mySQL_add_index
https://github.com/rogerjdeangelis/utl_sql_update_master_using_a_transaction_table_mysql_database
https://github.com/rogerjdeangelis/utl_with_a_press_of_a_function_key_convert_the_highlighted_dataset_to_a_mysql_database_table

SQLITE
-------------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-adding-the-missing-math-stat-and-string-functions-to-python-sql-pandasql-sqllite3
https://github.com/rogerjdeangelis/utl-converting-r-sas-and-excel-serial-dates-to-sqllite-serial-dates-and-posix-iso8601-clinical
https://github.com/rogerjdeangelis/utl-example-of-recursion-in-sqllite-creating-a-file-hierachy-r-and-python-sql-multi-language
https://github.com/rogerjdeangelis/utl-exporting-python-panda-dataframes-to-wps-r-using-a-shared-sqllite-database
https://github.com/rogerjdeangelis/utl-find-area-under-curve-and-compute-regression-slope-and-intercept-using-sqllite-r-python
https://github.com/rogerjdeangelis/utl-importing-nhanes-data-in-raw-and-from-sqllite-database
https://github.com/rogerjdeangelis/utl-key-differences-between-sas-proc-sql-and-sqllite-in-r-sqldf-and-python-pdsql-packages
https://github.com/rogerjdeangelis/utl-passing-r-python-and-sas-macro-vars-to-sqllite-interface-arguments
https://github.com/rogerjdeangelis/utl-python-very-simple-interactive-sqllite-dashboard-to-query-roger-deangelis-repositories
https://github.com/rogerjdeangelis/utl-sas-to-and-from-sqllite-excel-ms-access-spss-stata-using-r-packages-without-sas
https://github.com/rogerjdeangelis/utl-set-up-a-temporary-in-memory-sqllite-database-in-r-for-added-functionality
https://github.com/rogerjdeangelis/utl-sqllite-working-with-dates-in-r-and-python
https://github.com/rogerjdeangelis/sas-macros-to-import-sqlite-tables-with-and-without-data-typing-sas-access-prouct-not-needed
https://github.com/rogerjdeangelis/utl-add-sqlite-windows-functions-to-octave-sqlite-temporary-solition-until-put-in-octave-forge
https://github.com/rogerjdeangelis/utl-better-than-using-csv-files-to-transport-sas-datasets-sqlite-table
https://github.com/rogerjdeangelis/utl-centered-moving-average-of-three-observations-using-sas-r-zoo-package-and-sqlite-sqldf
https://github.com/rogerjdeangelis/utl-convert-a-sqlite-numeric-table-to-a-matrix-for-octave-matlab-processing
https://github.com/rogerjdeangelis/utl-creating-sqlite-and-postgresql-tables-from-sas-datasets-without-sas-access-and-a-blueprint
https://github.com/rogerjdeangelis/utl-drop-down-to-perl-and-summarize-a-sql-table-using-sqlite-file-database
https://github.com/rogerjdeangelis/utl-example-of-a-cartesian-join-in-sqlite-using-cross-join-instead-of-outer-join
https://github.com/rogerjdeangelis/utl-example-of-sqlite-group_concat-and-associated-sas-datastep-solution
https://github.com/rogerjdeangelis/utl-example-of-using-r-sqlite-sqldf-group-concat-function-to-concat-a-hierarchy-of-strings
https://github.com/rogerjdeangelis/utl-identify-changes-in-column-values-using-sas-and-sqlite
https://github.com/rogerjdeangelis/utl-import-a-sqlite-table-with-data-types-without-sas-access-product
https://github.com/rogerjdeangelis/utl-missing-basic-math-and-stat-functions-in-python-sqlite3-sql
https://github.com/rogerjdeangelis/utl-r-python-sas-sqlite-subtracting-the-means-of-a-specific-column-from-other-columns
https://github.com/rogerjdeangelis/utl-rolling-moving-five-year-sum-using-sqlite-window-functiona-r-and-python
https://github.com/rogerjdeangelis/utl-sqlite-processing-in-python-with-added-math-and-stat-functions
https://github.com/rogerjdeangelis/utl-update-a-row-in-a-table-using-sas-sql-update-and-r-sqlite-sqldf-update
https://github.com/rogerjdeangelis/utl-using-r-and-python-sqlite-recursion-to-generate-date-or-number-sequences
https://github.com/rogerjdeangelis/utl-utl-interface-sqlite3-with-any-language-that-supports-host-commands

EXCEL
---------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/excel-how-do-I-remove-troublesome-characters-before-importing
https://github.com/rogerjdeangelis/ods_excel_does_not_always_honor_start_at--bug
https://github.com/rogerjdeangelis/utl-Delete-all-files-in-a-directory-with-a-specified-extension-ie-delete-excel-files
https://github.com/rogerjdeangelis/utl-If-over-sixty-days-between-rx-then-new-rx-else-refill-rx-sas-and-sql-r-sas-python-excel
https://github.com/rogerjdeangelis/utl-Import-excel-sheet-as-character-fixing-truncation-mixed-type-columns-and-appending-issues
https://github.com/rogerjdeangelis/utl-Import-the-datepart-of-an-excel-datetime-formatted-columns
https://github.com/rogerjdeangelis/utl-Password-protect-an-EXCEL-file-in-sas-without-X-command
https://github.com/rogerjdeangelis/utl-a-cursory-comparison-of-excel-alternatives-using-r-python-and-libre-office-calc
https://github.com/rogerjdeangelis/utl-add-a-tab-to-excel-that-autmatically-impot-a-sas-datasets-www.colectica-com
https://github.com/rogerjdeangelis/utl-add-formula-to-an-existing-excel-worksheet-using-powershell
https://github.com/rogerjdeangelis/utl-add-monthly-worksheets-to-an-existing-yearly-excel-workbook
https://github.com/rogerjdeangelis/utl-add-one-to-each-new-occurance-of-a-value-in-a-series-sas-hash-and--sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-add-sequence-number-by-group-sas-sql-with-without-monotonic-and-in-sql-r-python-excel
https://github.com/rogerjdeangelis/utl-add-sequence-numbers-to-each-group-using-sas-and-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-adding-a-password-to-an-existing-excel-workbook
https://github.com/rogerjdeangelis/utl-adding-a-second-ods-excel-created-sheet-to-a-closed-ods-excel-workbook
https://github.com/rogerjdeangelis/utl-adding-a-sheet-to-an-existing-open-and-on-screen-or-saved-excel-workbook
https://github.com/rogerjdeangelis/utl-appending-records-to-an-existing-excel-sheet
https://github.com/rogerjdeangelis/utl-apply-excel-styling-across-multiple-spreadsheets-using-openxlsx-in-r
https://github.com/rogerjdeangelis/utl-apply-transactions-to-master-to-flag-male-and-non-male-patients-sas-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-applying-meta-data-and-importing-data-from-an-excel-named-range
https://github.com/rogerjdeangelis/utl-avoid-storing-numbers-as-text-when-exporting-mixed-type-to-excel
https://github.com/rogerjdeangelis/utl-calculate-percentage-by-group-in-wps-r-python-excel-sql-no-sql
https://github.com/rogerjdeangelis/utl-calculating-three-day-moving-sum-by-city-using-r-python-excel-sql
https://github.com/rogerjdeangelis/utl-casting-and-reformatting-excel-data-before-importing
https://github.com/rogerjdeangelis/utl-classic-pivot-wider-transpose-with-output-compound-column-names-using-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-clear-named-and-unnamed-cell-ranges-in-excel
https://github.com/rogerjdeangelis/utl-combine-name-value-pairs-with-all-other-name-value-pairs-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-combine-text-in-an-excel-column-down-multiple-rows-by-group
https://github.com/rogerjdeangelis/utl-complete-years-using-overall-max-and-min-applied-to-each-group-r-sas-and-sql-r-python-excel
https://github.com/rogerjdeangelis/utl-concatenate-contents-from-the-same-column-base-sas-r-python-excel-sql
https://github.com/rogerjdeangelis/utl-concatenating-thirty-seven-excel-tabs-while-correcting-column-types-and-using-longest-length
https://github.com/rogerjdeangelis/utl-convert-excel-sheet-to-sas-dataset-without-sas-access
https://github.com/rogerjdeangelis/utl-convert-excel-to-csv-by-dropping-down-to-r-or-python
https://github.com/rogerjdeangelis/utl-converting-r-sas-and-excel-serial-dates-to-sqllite-serial-dates-and-posix-iso8601-clinical
https://github.com/rogerjdeangelis/utl-copy-all-sas-datasets-in-work-library-to-tabs-in-one-excel-workbook
https://github.com/rogerjdeangelis/utl-count-employees-that-have-made-at-leat-one-stock-ticker-purchase-r-sql-python-excel
https://github.com/rogerjdeangelis/utl-create-a-pdf-excel-html-proc-report-with-greek-letters
https://github.com/rogerjdeangelis/utl-create-graphs-in-excel-using-excel-chart-templates
https://github.com/rogerjdeangelis/utl-create-new-variable-by-multiplying-corresponding-variables-and-summing-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-create-r-dataframe-and-sas-dataset-from-second-sheet-in-multisheet-excel-workbook-r-openxlsx
https://github.com/rogerjdeangelis/utl-creating-a-two-by-two-grid-of-reports-in-excel
https://github.com/rogerjdeangelis/utl-creating-multiple-odbc-tables-in-a-one-excel-sheet
https://github.com/rogerjdeangelis/utl-creating-variables-that-flag-changes-over-time-for-a-medication-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-cumulative-sum-by-group-in-the-order-of-rank-variable-sas-and-sql-r-python-octave-excel
https://github.com/rogerjdeangelis/utl-cumulative-sums-by-group-using-base-sas-and-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-do-not-add-data-transformations-to-create-csv-files-from-excel-or-any-other-data-structure
https://github.com/rogerjdeangelis/utl-does-the-excel-named-range-table-exist
https://github.com/rogerjdeangelis/utl-does-the-excel-sheet-exist
https://github.com/rogerjdeangelis/utl-drop-down-to-powershell-and-programatically-create-an-odbc-data-source-for-excel-wps-r-rodbc
https://github.com/rogerjdeangelis/utl-example-of-perl-regex-in-sas-r-python-and-excel-sql
https://github.com/rogerjdeangelis/utl-example-rtf-excel-and-pdf-reports-using-all-sas-provided-style-templates
https://github.com/rogerjdeangelis/utl-excel-changing-cell-contents-inside-proc-report
https://github.com/rogerjdeangelis/utl-excel-database-schema-and-tables-as-workbooks-sheets-and-named-ranges-update-existing-workbooks
https://github.com/rogerjdeangelis/utl-excel-fixing-bad-formatting-using-passthru
https://github.com/rogerjdeangelis/utl-excel-grid-of-four-reports-in-one-sheet
https://github.com/rogerjdeangelis/utl-excel-hiding-columns-height-and-weight-in-sheet-class
https://github.com/rogerjdeangelis/utl-excel-hiperlinks-click-on-your-favorite-baseball-player-and-a-google-search-will-pop-up
https://github.com/rogerjdeangelis/utl-excel-import-individual-cells
https://github.com/rogerjdeangelis/utl-excel-import-number-strings-with-spaces
https://github.com/rogerjdeangelis/utl-excel-report-with-two-side-by-side-graphs-below_python
https://github.com/rogerjdeangelis/utl-excel-use-the-name-of-the-last-variable-in-the-pdv-for-sheet-name
https://github.com/rogerjdeangelis/utl-excel-using-proc-report-workarea-columns-to-operate--on-arbitrary-row
https://github.com/rogerjdeangelis/utl-export-excel-sheet-to-sas-table-using-one-line-macro-invocation-without-sas-access-powershell
https://github.com/rogerjdeangelis/utl-extract-sheet-names-from-multiple-excel-versions-using-r
https://github.com/rogerjdeangelis/utl-extracting-hyperlinks-from-an-excel-sheet-python
https://github.com/rogerjdeangelis/utl-find-airlines-that-use-mutiple-vendors-for-in-flight-services-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-find-out-which-excel-columns-are-dates-and-assign-date-type
https://github.com/rogerjdeangelis/utl-fix-excel-columns-with-mutiple-datatypes-on-the-excel-side-using-ms-sql-and-passthru
https://github.com/rogerjdeangelis/utl-fix-excel-date-fields-on-the-excel-side-using-ms-sql-and-passthru
https://github.com/rogerjdeangelis/utl-for-each-row-return-the-column-name-of-the-largest-value-base-and-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-force-excel-to-read-all-the-columns-as-numeric-or-character
https://github.com/rogerjdeangelis/utl-formatting-ai-seacrh-output-in-pdf-rtf-and-excel-format-perplexity-chatGPT-results
https://github.com/rogerjdeangelis/utl-get-the-color-of-a-cell-in-excel-xlsx
https://github.com/rogerjdeangelis/utl-highlight-existing-cells-in-excel-sheet2-that-correspond-to-cells-in-sheet1-with-specified-value
https://github.com/rogerjdeangelis/utl-highlite-sas-dataset-and-view-the-table-in-excel-without-sas-access
https://github.com/rogerjdeangelis/utl-how-to-check-whether-a-student-is-in-the-Excel-sheet-class
https://github.com/rogerjdeangelis/utl-how-to-find-common-values-in-four-columns-r-and-sql-r-sas-python-excel
https://github.com/rogerjdeangelis/utl-how-to-import-the-ms-excel-sheet-names-in-actual-order
https://github.com/rogerjdeangelis/utl-how-to-load-excel-sheets-with-sheet-names-with-31-characters
https://github.com/rogerjdeangelis/utl-identify-missing-drug-presciption-days-by-patient-using-sql-recursion-r-python-and-excel
https://github.com/rogerjdeangelis/utl-identify-sequential-maximums-in-a-series-of-numbers-using-base-sas-and-sql-r-python-excel
https://github.com/rogerjdeangelis/utl-import-a-messy-excel-file
https://github.com/rogerjdeangelis/utl-import-all-excel-columns-as-character
https://github.com/rogerjdeangelis/utl-import-all-excel-columns-as-character-three-solutions
https://github.com/rogerjdeangelis/utl-import-all-excel-dates-as-character-strings-and-convert-back-to-SAS-dates
https://github.com/rogerjdeangelis/utl-import-all-excel-workskkets-and-named-ranges--in-all-workbooks-in-a-directory
https://github.com/rogerjdeangelis/utl-import-csv-file-to-excel-with-leading-zeros
https://github.com/rogerjdeangelis/utl-import-excel-when-column-names-are-excel-dates
https://github.com/rogerjdeangelis/utl-import-excel-workbooks-in-all-folders-and-subfolders
https://github.com/rogerjdeangelis/utl-importing-excel-datetime-values-in-xlsx-and-xlsx-workbooks
https://github.com/rogerjdeangelis/utl-importing-excel-string_of_32-thousand-characters-SAS-XLConnect
https://github.com/rogerjdeangelis/utl-importing-excel-when-sheetname-has-spaces
https://github.com/rogerjdeangelis/utl-importing-inconsistently-formatted-excel-dates-numeric-and-charater-in-the-same-column
https://github.com/rogerjdeangelis/utl-importing-multiple-excel-files-which-names-are-defined-by-state
https://github.com/rogerjdeangelis/utl-importing-multiple-excel-worksheets-without-access-to-pc-files
https://github.com/rogerjdeangelis/utl-in-palce-updates-to-an-existing-shared-excel-workbook
https://github.com/rogerjdeangelis/utl-increment-a-counter-each-time-a-missing-value-is-encountered-usin-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-join-a-sas-table-with-an-excel-table-when-column-names-that-are-dates
https://github.com/rogerjdeangelis/utl-keep-orginal-SAS-table-but-mask-excel-output
https://github.com/rogerjdeangelis/utl-keeping-leading-and-trailing-zeros-in-character-fields-with-ods-excel-output
https://github.com/rogerjdeangelis/utl-layout-ods-excel-reports-in-a-grid
https://github.com/rogerjdeangelis/utl-load-and-extract-ms-excel-document-properties-metadata
https://github.com/rogerjdeangelis/utl-manipulate-excel-directly-using-passthru-microsoft-sql-wps-r-rodbc
https://github.com/rogerjdeangelis/utl-matching-tables-based-on-a-hierachy-of-rules-sas-hash-sql-r-pytho-excel
https://github.com/rogerjdeangelis/utl-merge-cells-in-excel-and-rtf-output-using-proc-report-and-r-openxlsx-package
https://github.com/rogerjdeangelis/utl-move-even-numbered-columns-to-the-end-of-the-sas-pdv-datastep-and-sql-r-excel
https://github.com/rogerjdeangelis/utl-ninetyfifth-percentiles-of-home-runs-for-lf-and-rf-baseball-players-sas-and-sql-r-excel
https://github.com/rogerjdeangelis/utl-no-need-for-sql-or-sort-merge-use-a-elegant-hash-excel-vlookup
https://github.com/rogerjdeangelis/utl-ods-excel-color-code-every-other-column-in-a-specified-row
https://github.com/rogerjdeangelis/utl-ods-excel-hilite-diagonal-cells
https://github.com/rogerjdeangelis/utl-ods-excel-update-excel-sheet-in-place-python
https://github.com/rogerjdeangelis/utl-ods-export-sas-table-to-excel-with-rotated-column-headers
https://github.com/rogerjdeangelis/utl-pivot-excel-columns-and-output-a-database-table
https://github.com/rogerjdeangelis/utl-pivot-long--excel-sheet-and-run-a-regression-in-r-and-python
https://github.com/rogerjdeangelis/utl-pivot-transpose-an-excel-sheet-with-columns-that-are-excel-dates
https://github.com/rogerjdeangelis/utl-pivot-wide-when-variable-names-contain-values-sql-and-base-r-sas-oython-excel-postgreSQL
https://github.com/rogerjdeangelis/utl-place-animal-behavior-into-four-categories-using-sql-sas-r-python-excel-mapping-lookups
https://github.com/rogerjdeangelis/utl-position-a-sas-table-arbitrarily-in-excel-without-column-names-using-r-and-python
https://github.com/rogerjdeangelis/utl-posting-your-problem-with-an-ascii-image-that-looks-just-like-an-excel-shee
https://github.com/rogerjdeangelis/utl-preserving-excel-formatting-when-writing-to-an-existing-worksheet
https://github.com/rogerjdeangelis/utl-programatically-downlaod-an-excel-file-from-the-web
https://github.com/rogerjdeangelis/utl-programatically-execute-excel-vba-macro-using-sas-python
https://github.com/rogerjdeangelis/utl-programatically-search-all-cells-in-an-excel-sheet-for-an-arbitrary-string-python-openxl
https://github.com/rogerjdeangelis/utl-r-open-closed-excel-workbook-an-update-master-sheet-using-transaction-sheet
https://github.com/rogerjdeangelis/utl-remove-sheet-from-excel-workbook
https://github.com/rogerjdeangelis/utl-remove-sheet-from-existing-excel-worksheet-unix-and-windows-R
https://github.com/rogerjdeangelis/utl-rename-excel-columns-to-common-names-before-creating-sas-tables
https://github.com/rogerjdeangelis/utl-renaming-duplicate-excel-column-names-before-importing
https://github.com/rogerjdeangelis/utl-rogers-ods-excel-lineprinter-style-formchar-gridlines-sort-of-sharebuffers
https://github.com/rogerjdeangelis/utl-roll-up-adverse-events-by-patient-and-date-using-sql-groupcat-r-python-and-excel
https://github.com/rogerjdeangelis/utl-round-trip-sas-time-values-to-excel-and-back-sas-and-r-openxlsx
https://github.com/rogerjdeangelis/utl-safe-way-import-excel-time-value
https://github.com/rogerjdeangelis/utl-safely-sending-dates-or-datetimes-back-and-forth-to-excel
https://github.com/rogerjdeangelis/utl-sas-ods-bidirectional-hyperlinked-table-of-contents-in-ods-pdf-html-and-excel
https://github.com/rogerjdeangelis/utl-sas-ods-excel-to-create-excel-report-and-separate-png-graph-finally-r-for-layout-in-excel
https://github.com/rogerjdeangelis/utl-sas-to-and-from-sqllite-excel-ms-access-spss-stata-using-r-packages-without-sas
https://github.com/rogerjdeangelis/utl-select-fruits-purchased-and-not-purchased--from-old-macdonalds-farm-excel-r-sql
https://github.com/rogerjdeangelis/utl-select-phase2-and-phase3-trials-with-both-treatment-and-control-arms-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-select-the-diagonal-values-from-a-dataset-in-excel-r-wps-python
https://github.com/rogerjdeangelis/utl-select-the-top-five-ages-using-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-select-the-top-n-and-bottom-n-by-group-sql-r-python-excel
https://github.com/rogerjdeangelis/utl-select-the-top-ten-rows-from-excel-table-without-importing-to-sas
https://github.com/rogerjdeangelis/utl-select-the-top-two-largest-values-by-group-using-sql-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-select-type-and-length-using-odbc-excel-passthru-query
https://github.com/rogerjdeangelis/utl-send-all-tables-in-a-sas-library-to-excel
https://github.com/rogerjdeangelis/utl-sending-a-formula-to-excel-to-reference-a-cell-in-another-sheet
https://github.com/rogerjdeangelis/utl-seven-algorithms-to-convert-a-sas-dataset-to-an-excel-workbook
https://github.com/rogerjdeangelis/utl-side-by-side-proc-report-output-in-pdf-html-and-excel
https://github.com/rogerjdeangelis/utl-side-by-side-reports-within-arbitrary-positions-in-one-excel-sheet-wps-r
https://github.com/rogerjdeangelis/utl-side-by-side-sas-tables-in-one-excel-sheet
https://github.com/rogerjdeangelis/utl-simple-example-of-a-three-table-full-outer-join-by-ids-sas-r-python-excel-sql
https://github.com/rogerjdeangelis/utl-simple-r-code-to-covert-excel-to-sas-and-sas-to-excel
https://github.com/rogerjdeangelis/utl-simple-three-letter-commands-to-format-perplexity-AI-results-for-word-pdf-text-and-excel
https://github.com/rogerjdeangelis/utl-single-click-and-eight-excel-tabs-are-converted-to-csv-files
https://github.com/rogerjdeangelis/utl-skilled-nursing-cost-reports-2011-2019-in-excel
https://github.com/rogerjdeangelis/utl-splitting-rows-based-on-variable-suffixes-with-and-without-sql-arrays-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-sql-arrays-to-simplfy-summarization-of-many-variables-in-sas-r-python-excel-sql-arrays
https://github.com/rogerjdeangelis/utl-sql-workaround-fix-for-excel-bug-when-counting-distinct-values
https://github.com/rogerjdeangelis/utl-stack-tables-when-tables-are-not-conguent-base-r-and-sql-r-python-excel
https://github.com/rogerjdeangelis/utl-string-functions-to-parse-messy-four-word--string-into-name-grade-and-weight-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-subset-a-database-table-based-on-a-list-of-names-in-excel
https://github.com/rogerjdeangelis/utl-subset-sas-dataset-based-on-the-value-of-a-macro-variable-contiained-in-an-excel-worksheet
https://github.com/rogerjdeangelis/utl-substituting-name-and-label-to-column-headings-in-excel
https://github.com/rogerjdeangelis/utl-sum-all-pairs-of-columns-using-by-row-base-r-sql-sas-r-excel
https://github.com/rogerjdeangelis/utl-sum-purchases-before-and-after-sale-dates-sql-sas-r-python-and-excel
https://github.com/rogerjdeangelis/utl-sum-the-three-largest-yoy-growth-by-company-sql-partitioning-and-rank-sas-r-python-excel
https://github.com/rogerjdeangelis/utl-tables-to-specific-excel-cells
https://github.com/rogerjdeangelis/utl-temporarily-change-the-excel-base-font-to-a-blue-consolas-monospace-font-r-openxlsx
https://github.com/rogerjdeangelis/utl-total-weekly-cost-for-breakfast-lunch-and-dinner-for-mary-jane-and-mike-sas-r-sql-python-excel
https://github.com/rogerjdeangelis/utl-update-a-master-sheet-with-transaction-sheet-using-excel-and-r-openxls-package-and-sqldf
https://github.com/rogerjdeangelis/utl-update-an-excel-workbook-in-place
https://github.com/rogerjdeangelis/utl-update-an-existing-excel-named-range-R-python-sas
https://github.com/rogerjdeangelis/utl-update-existing-excel-sheet-in-place-using-r-dcom-client
https://github.com/rogerjdeangelis/utl-update-in-place-sheet2-by-adding-dinner-costs-from-sheet1-preserving-excel-formatting-r
https://github.com/rogerjdeangelis/utl-using-column-position-instead-of-excel-column-names-due-to-misspellings-sas-r-python
https://github.com/rogerjdeangelis/utl-using-excel-to-get-a-usefull-proc-tabulate-output-table
https://github.com/rogerjdeangelis/utl-using-only-r-openxlsx-to-add-excel-formulas-to-an-existing-sheet
https://github.com/rogerjdeangelis/utl-using-proc-odstext-to-add-documentation-tabs-to-your-excel-workbook
https://github.com/rogerjdeangelis/utl-using-sql-instead-the-excel-formula-language-for-solving-excel-problems-pyodbc
Select excel from d:/git/git_010_repos.sasbdat
https://github.com/rogerjdeangelis/utl-very-fast-summation-of-ll6-columns-in-excel-without-importing-to-sheet
https://github.com/rogerjdeangelis/utl-within-sql-join-a-a-text-or-excel-file-to-a-sas-or-foreign-table-without-proc-import
https://github.com/rogerjdeangelis/utl-wps-create-a-pie-chart-in-excel-using-wps-proc-gchart
https://github.com/rogerjdeangelis/utl_1130pm_batch_SAS_job_that_imports_an_excel_if_modified_that_day
https://github.com/rogerjdeangelis/utl_adding_SAS_graphics_at_an_arbitrary_position_into_existing_excel_sheets
https://github.com/rogerjdeangelis/utl_convert_a_sas_dataset_to_excel_without_sas_or_excel_only_need_R
https://github.com/rogerjdeangelis/utl_create_many_excel_workbooks_for_selected_cities_in_sashelp_zipcode_with_logging
https://github.com/rogerjdeangelis/utl_creating_www_hyperlinks_in_ods_excel
https://github.com/rogerjdeangelis/utl_excel-copying-a-worlbook-with-many-named-ranges-into-sas-tables
https://github.com/rogerjdeangelis/utl_excel_Import_and_transpose_range_A9-Y97_using_only_one_procedure
https://github.com/rogerjdeangelis/utl_excel_add_formula_inplace
https://github.com/rogerjdeangelis/utl_excel_add_formulas
https://github.com/rogerjdeangelis/utl_excel_add_sheet
https://github.com/rogerjdeangelis/utl_excel_add_to_sheet
https://github.com/rogerjdeangelis/utl_excel_combining_sheets_without_common_names_types_lengths
https://github.com/rogerjdeangelis/utl_excel_create_a_sheet_for_each_table_with_variable_name_position_and_label
https://github.com/rogerjdeangelis/utl_excel_create_sql_insert_and_value_statements_to_update_databases
https://github.com/rogerjdeangelis/utl_excel_determine_type_length
https://github.com/rogerjdeangelis/utl_excel_experimenting-with-the-new-ods-excel-destination
https://github.com/rogerjdeangelis/utl_excel_exporting_data_with_leading_zeros
https://github.com/rogerjdeangelis/utl_excel_fix_type_length_on_import
https://github.com/rogerjdeangelis/utl_excel_highlight_individual_cells_based_on_indicator_variables
https://github.com/rogerjdeangelis/utl_excel_import_all_columns_as_character_and_preserve_long_variable_names
https://github.com/rogerjdeangelis/utl_excel_import_data_from_a_xlsx_file_where_first_2_rows_are_header
https://github.com/rogerjdeangelis/utl_excel_import_entire_directory
https://github.com/rogerjdeangelis/utl_excel_import_long_colnames
https://github.com/rogerjdeangelis/utl_excel_import_only_female_students
https://github.com/rogerjdeangelis/utl_excel_import_sas_functions_fail_on_cells_with_mutiple_line_breaks
https://github.com/rogerjdeangelis/utl_excel_import_sub_rectangle
https://github.com/rogerjdeangelis/utl_excel_import_two_excel_ranges_within_one_sheet
https://github.com/rogerjdeangelis/utl_excel_import_xlsm_to_sas_dataset
https://github.com/rogerjdeangelis/utl_excel_importing_unicode_and_other_special_characters_without_changing_sas_encoding
https://github.com/rogerjdeangelis/utl_excel_merge_two-sheets
https://github.com/rogerjdeangelis/utl_excel_reading_a_single_cell
https://github.com/rogerjdeangelis/utl_excel_sas_wps_r_import_xlsx_without_sas_access_to_pc_files
https://github.com/rogerjdeangelis/utl_excel_update_inplace
https://github.com/rogerjdeangelis/utl_excel_update_rectangle
https://github.com/rogerjdeangelis/utl_excel_update_xlsm_workbook_using_SAS_dataset
https://github.com/rogerjdeangelis/utl_excel_updating-named-ranged-cells
https://github.com/rogerjdeangelis/utl_excel_using_a_cell_value_for_the_name_of_sas_dataset
https://github.com/rogerjdeangelis/utl_excel_using_byval_sex_and_sheet_interval_bygroups_to_create_multiple_worksheets
https://github.com/rogerjdeangelis/utl_fix_excel_column_names_before_import
https://github.com/rogerjdeangelis/utl_how_to_get_data_from_excel_file_into_wps_sas_procedure
https://github.com/rogerjdeangelis/utl_import_all_excel_workbooks_created_in_the_previous_seven_days
https://github.com/rogerjdeangelis/utl_import_data_from_excel_sheet_with_headers_and_footers_without_specifying_range_option
https://github.com/rogerjdeangelis/utl_import_excel_column_names_that_contain_a_dollar_sign_and_rename_without
https://github.com/rogerjdeangelis/utl_import_excel_unicode
https://github.com/rogerjdeangelis/utl_importing_three_excel_tables_that_are_in_one_sheet
https://github.com/rogerjdeangelis/utl_joining_and_updating_excel_sheets_without_importing_data
https://github.com/rogerjdeangelis/utl_maintaining_all_significant_digits_when_importing_excel_sheet
https://github.com/rogerjdeangelis/utl_maintaining_numeric_significance_when_exporting_and_importing_excel_workbooks
https://github.com/rogerjdeangelis/utl_ods_excel_conditionaly_higlight_individua1_cells
https://github.com/rogerjdeangelis/utl_ods_excel_create_a_table_of_contents_with_links_to_and_from_each_sheet
https://github.com/rogerjdeangelis/utl_ods_excel_font_size_and_justification_proc_report_titles_formatting
https://github.com/rogerjdeangelis/utl_ods_excel_merging_cells_after_column_header_and_before_column_names
https://github.com/rogerjdeangelis/utl_passthru_to_excel_to_fix_column_names
https://github.com/rogerjdeangelis/utl_proc_import_columns_as_character_from_excel_linux_or_windows
https://github.com/rogerjdeangelis/utl_programatically_execute_excel_macro_using_wps_proc_python
https://github.com/rogerjdeangelis/utl_put_excel_sheetnames_into_sas_macro_variable
https://github.com/rogerjdeangelis/utl_renaming_duplicate_excel_columns_to_avoid_name_collisions_when_importing
https://github.com/rogerjdeangelis/utl_sas_v5_transport_file_to_excel
https://github.com/rogerjdeangelis/utl_side_by_side_excel_reports
https://github.com/rogerjdeangelis/utl_stacking-strings-in-one-excel-cell-using-ods-excel-newline-carriage-return-line-feed-tags
https://github.com/rogerjdeangelis/utl_table_of_contents_with_excel_links_to_sheets


/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/


























































































































































































































































































































%utl_rbeginx;
parmcards4;
&_init_;
proc r;
submit;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages

library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Shape functions for linear triangles
shape_functions <- function(xi, eta) {
  N <- c(1 - xi - eta, xi, eta)
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  return(list(N = N, dN_dxi = dN_dxi))
}

# Element stiffness matrix for heat conduction
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)
  invJ <- solve(J)

  # Shape function derivatives in global coordinates
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  dN_dx <- invJ %*% dN_dxi

  # Element stiffness matrix
  ke <- matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      ke[i, j] <- k * detJ * t(dN_dx[, i]) %*% dN_dx[, j] / 2
    }
  }

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)
T_ambient <- 20 # Ambient temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh
mesh <- create_rectangular_mesh(Lx, Ly, nx = 30, ny = 15)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = temperature
)

png("d;/png/fea.png")
# Contour plot
ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_contour_filled(bins = 20) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_d(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()

# 3D surface plot
ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_raster(aes(fill = temperature), interpolate = TRUE) +
  geom_contour(color = "white", alpha = 0.5, bins = 15) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution - Surface Plot",
       x = "X (m)", y = "Y (m)") +
  theme_minimal()

# Temperature profile along centerline
centerline_nodes <- which(mesh$nodes$y == Ly/2)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()

# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")

# Display temperature at specific points
cat("\nTemperature at key locations:\n")
cat("Center of plate (x=0.5, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 0.5) + abs(mesh$nodes$y - 0.25))], "°C\n")
cat("Right edge center (x=1.0, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 1.0) + abs(mesh$nodes$y - 0.25))], "°C\n")

endsubmit;
run;quit;


 ;;;;%end;%mend;/*'*/ *);*};*];*/;/*"*/;run;quit;%end;end;run;endcomp;%utlfix;

%utl_rbeginx;
parmcards4;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages
library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Element stiffness matrix for heat conduction - CORRECTED VERSION
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)

  # Check for degenerate element
  if (abs(detJ) < 1e-10) {
    return(matrix(0, 3, 3))
  }

  invJ <- solve(J)

  # Shape function derivatives in natural coordinates (xi, eta)
  # For linear triangles: N1 = 1 - xi - eta, N2 = xi, N3 = eta
  dN_dxi <- matrix(c(-1, 1, 0,   # dN/dxi for N1, N2, N3
                     -1, 0, 1),  # dN/deta for N1, N2, N3
                   nrow = 2, ncol = 3, byrow = TRUE)

  # Transform to global coordinates
  dN_dx <- invJ %%*%% dN_dxi

  # Element stiffness matrix: k * ?(?N? · ?N) dO
  # For constant Jacobian: ke = k * detJ * (dN_dx? %%*%% dN_dx) / 2
  ke <- k * detJ * (t(dN_dx) %%*%% dN_dx) / 2

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh (using smaller mesh for testing)
mesh <- create_rectangular_mesh(Lx, Ly, nx = 15, ny = 8)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = as.numeric(temperature)
)

# Contour plot
ggplot(plot_data, aes(x = x, y = y)) +
  geom_tile(aes(fill = temperature)) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()

# Temperature profile along centerline
centerline_nodes <- which(abs(mesh$nodes$y - Ly/2) < 1e-10)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()

# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")
;;;;
%utl_rendx;
























options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;




















































































































/*___                                            _  ____     _             _     _        _       _
|___ \   ___ _   _ _ __ ___  _ __  _   _  / | __| |  ___ ___ | | __| | _ __ | | __ _| |_ ___
  __) | / __| | | | `_ ` _ \| `_ \| | | | | |/ _` | / __/ _ \| |/ _` || `_ \| |/ _` | __/ _ \
 / __/  \__ \ |_| | | | | | | |_) | |_| | | | (_| || (_| (_) | | (_| || |_) | | (_| | ||  __/
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_|\__,_| \___\___/|_|\__,_|| .__/|_|\__,_|\__\___|
             |___/          |_|    |___/                     |_|
*/



import sympy as sp
from sympy import symbols, exp, pi, sqrt, integrate, diff, simplify, pprint, erf

# Define symbols
x = sp.Symbol('x')
T = sp.Function('T')(x)
T_in, T_out, L, m_dot, c_p, P, h = sp.symbols('T_in T_out L m_dot c_p P h')

# Define the differential equation
dT_dx = (P / (m_dot * c_p * L)) * (1 - (T - T_in) / (T_out - T_in))
sp,pprint(dT_dx)
eq = sp.Eq(T.diff(x), dT_dx)

# Solve the differential equation
solution = sp.dsolve(eq, T, ics={T.subs(x, 0): T_in})

# Simplify the solution
simplified_solution = sp.simplify(solution.rhs)

print("Solution:")
sp.pprint(simplified_solution)
;;;;
%utl_pyendx;


          2. Python 1 dimensional closed form solution.
             Temp(x) along the length of the cold plate
             Requires boundary temperatures
             (for symbolic mathematics I like sympy)

             Solution: (given boundary conditions)
                                                      P*x
                                           --------------------------
                                           L*c_p*m_dot*(T_in - T_out)
             T(x)= T_out + (T_in - T_out)*e




















































/*___                                      ____     _             _     _
|___ \   ___ _   _ _ __ ___  _ __  _   _  |___ \ __| |   ___ ___ | | __| |
  __) | / __| | | | `_ ` _ \| `_ \| | | |   __) / _` |  / __/ _ \| |/ _` |
 / __/  \__ \ |_| | | | | | | |_) | |_| |  / __/ (_| | | (_| (_) | | (_| |
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_____\__,_|  \___\___/|_|\__,_|
             |___/          |_|    |___/
 __ _       _
| _| | __ _| |_ ___
| || |/ _` | __/ _ \
| || | (_| | ||  __/
| ||_|\__,_|\__\___|
|__|
*/


2 ALTAIR SLC, SOLVING THE TWOD COLD PLATE PROBLEM
=================================================

Too long to post here






https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python.sas


Graphic output 2 dimensional plate
https://tinyurl.com/5n97pucx
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/heatequ.png

github
https://tinyurl.com/2dxceary
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python











A simple, reproducible real-world optimization example
using oct2py with Python is to call Octave and pass data back and forth seamlessly.

A simple, reproducible real-world optimization example
using simlab interface with Python is to call simlab and pass data back and forth seamlessly.



https://www.perplexity.ai/search/a-simple-reproducible-real-wor-N85SP9ewQ5yQLWyvD7RcGQ

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Define and minimize the objective function in one go
result = oc.fminsearch('@(x) (x(1)-3)**2 + (x(2)-5)**2', [0, 0])

# Pass result back to Python (result is a numpy array)
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;

These are valid matlab statements

@(x) (x(1)-3)^2 + (x(2)-5)^2;
fminsearch(objfun, [0, 0]);

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Run the entire optimization as an Octave command
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")
oc.eval("result = fminsearch(objfun, [0, 0]);  ")

# Retrieve the result
result = oc.pull("result")
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;























# Initialize Oct2Py session
oc = Oct2Py()

# Define the objective function in Octave as an anonymous function
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")

# Use Oct ave's fminsearch to minimize the objective function starting at [0,0]
result = oc.fminsearch('objfun', [0, 0])

# Pass result back to Python (result is a numpy array)
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;



# Example: SimLab Python Automation Script (Conceptual Demonstration)
&_init_;
proc python;
submit;
from hwx import simlab

# Load the simulated model directory/project
project_dir = "ConnectingRodModel"
model_file = f"{project_dir}/ConnectingRod.xmt_txt"

# Example XML template snippet for renaming a body in the model,
# this kind of XML string is extracted/generated by SimLab automation recording.
RenameBodyTemplate = """
<RenameBody CheckBox="ON" UUID="78633e0d-3d2f-4e9a-b075-7bff122772d8">
    <SupportEntities>
        <Entities>
            <Model>{model_file}</Model>
            <Body>"Body 2",</Body>
        </Entities>
    </SupportEntities>
    <NewName Value="{new_name}"/>
    <Output/>
</RenameBody>
"""

# Function to execute a rename operation in SimLab using the template
def rename_body_in_simlab(new_name):
    # Format the template for the desired new name
    command_xml = RenameBodyTemplate.format(model_file=model_file, new_name=new_name)
    # Call the simlab API to execute this command
    simlab.execute(command_xml)
    print(f"Renamed body to: {new_name}")

# Main workflow example: update a parameter, run simulation, get output
def run_simlab_example():
    # Pause recording before injecting parameters (known SimLab limitation)
    simlab.pause_recording()

    # Example: Rename a body (as a proxy to changing a parameter)
    rename_body_in_simlab("CHANGED_BODY_NAME")

    # Resume recording and run the model
    simlab.resume_recording()
    results = simlab.run_model(project_dir)

    # Extract output value(s) from results
    max_stress = results.get("MaxStress", None)
    max_displacement = results.get("MaxDisplacement", None)

    print(f"Max Stress: {max_stress}")
    print(f"Max Displacement: {max_displacement}")

    return max_stress, max_displacement

# Run the example workflow
if __name__ == "__main__":
    run_simlab_example()




# Example: SimLab Python Automation Script (Conceptual Demonstration)

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from hwx import simlab

# Load the simulated model directory/project
project_dir = "ConnectingRodModel"
model_file = f"{project_dir}/ConnectingRod.xmt_txt"

# Example XML template snippet for renaming a body in the model,
# this kind of XML string is extracted/generated by SimLab automation recording.
RenameBodyTemplate = """
<RenameBody CheckBox="ON" UUID="78633e0d-3d2f-4e9a-b075-7bff122772d8">
    <SupportEntities>
        <Entities>
            <Model>{model_file}</Model>
            <Body>"Body 2",</Body>
        </Entities>
    </SupportEntities>
    <NewName Value="{new_name}"/>
    <Output/>
</RenameBody>
"""

# Function to execute a rename operation in SimLab using the template
def rename_body_in_simlab(new_name):
    # Format the template for the desired new name
    command_xml = RenameBodyTemplate.format(model_file=model_file, new_name=new_name)
    # Call the simlab API to execute this command
    simlab.execute(command_xml)
    print(f"Renamed body to: {new_name}")

# Main workflow example: update a parameter, run simulation, get output
def run_simlab_example():
    # Pause recording before injecting parameters (known SimLab limitation)
    simlab.pause_recording()

    # Example: Rename a body (as a proxy to changing a parameter)
    rename_body_in_simlab("CHANGED_BODY_NAME")

    # Resume recording and run the model
    simlab.resume_recording()
    results = simlab.run_model(project_dir)

    # Extract output value(s) from results
    max_stress = results.get("MaxStress", None)
    max_displacement = results.get("MaxDisplacement", None)

    print(f"Max Stress: {max_stress}")
    print(f"Max Displacement: {max_displacement}")

    return max_stress, max_displacement

# Run the example workflow
if __name__ == "__main__":
    run_simlab_example()
endsubmit;
quit;run;


















Must be a problem with the personal edition

My setup does work


&_init_;
options set=PYTHONHOME "D:\python310";
PROC PYTHON;
submit;
x=2;
print(x);
ENDSUBMIT;
RUN;QUIT;

OUTPUT
======

Altair SLC

The PYTHON Procedure

2


ptions set=PYTHONHOME "D:\python310";
PROC PYTHON;
EXPORT DATA=TESTDATA PYTHON=df;
SUBMIT;
df.info()
dfo = df
ENDSUBMIT;
IMPORT DATA=TESTDATA PYTHON=dfo;
RUN;QUIT;

proc print data=dfo;
run;

189       options set=PYTHONHOME "D:\python310";
190       PROC PYTHON;
191       EXPORT DATA=TESTDATA PYTHON=df;
NOTE: Creating Python data frame 'df' from data set 'WORK.TESTDATA'
ERROR: Received unknown response [10]
192       SUBMIT;
193       df.info()
194       dfo = df
195       ENDSUBMIT;

NOTE: Submitting statements to Python:

WARNING: Could not cleanly terminate Python worker: Error writing to pipe : The pipe is being closed.


ERROR: Altair SLC has encountered a problem - please contact Altair Data Analytics Customer Support
196       IMPORT DATA=TESTDATA PYTHON=dfo;
197       quit; run;
198       ODS _ALL_ CLOSE;
199       FILENAME WPSWBHTM CLEAR;
ERROR: Expected a statement keyword : found "IMPORT"
200       RUN;QUIT;
201
202       proc print data=dfo;
                          ^
ERROR: Data set "WORK.dfo" not found
NOTE: Procedure PRINT was not executed because of errors detected
203       run;
NOTE: Procedure print step took :
      real time : 0.002
      cpu time  : 0.000





%utl_rbeginx;
parmcards4;
# Load necessary library
library(pracma)
library(haven)
source("c:/oto/fn_tosas9x.R")
# Parameters
Lx <- 1       # Length of the plate in x-direction
Ly <- 1       # Length of the plate in y-direction
Nx <- 20      # Number of grid points in x-direction
Ny <- 20      # Number of grid points in y-direction
dx <- Lx / (Nx - 1)
dy <- Ly / (Ny - 1)
alpha <- 0.01 # Thermal diffusivity
dt <- 0.001   # Time step
Nt <- 100     # Number of time steps

# Initialize temperature grid
u <- matrix(0, nrow = Nx, ncol = Ny)

# Initial condition: some function f(x,y)
f <- function(x, y) {
  return(sin(pi * x / Lx) * sin(pi * y / Ly))
}

# Apply initial condition
for (i in seq_len(Nx)) {
  for (j in seq_len(Ny)) {
    x <- (i - 1) * dx
    y <- (j - 1) * dy
    u[i, j] <- f(x, y)
  }
}

# Time-stepping loop
for (n in seq_len(Nt)) {
  u_new <- u
  for (i in 2:(Nx-1)) {
    for (j in 2:(Ny-1)) {
      u_new[i, j] <- u[i, j] + alpha * dt * (
        (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 +
        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2
      )
    }
  }
  u <- u_new
}
str(u)
xy=as.data.frame(u)
head(xy)
png("d:/png/heatequ.png")
# Plot the final temperature distribution
image(xy, main="Temperature Distribution", xlab="X", ylab="Y", col=heat.colors(100))
dev.off()
fn_tosas9x(
      inp    = xy
     ,outlib ="d:/sd1/"
     ,outdsn ="want"
     )
;;;;
%utl_rendx;

libname sd1 "d:/sd1";
proc print data=sd1.want;
format _numeric_ 4.2;
run;quit

proc transpose data=sd1.want out=wantxpo;
by rownames;
run;quit;

data wantxpox;
 set wantxpo;
 y=input(substr(_name_,2),3.);
 x=rownames;
 z=col1;
 drop _name_ col1;
run;quit;

options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;








































































































































































































%let pgm=utl-integrating-personal-altairslc-with-matlab-sympy-and-r-pracma-twoD-cold-plate-problem;

%stop_submission;

Integrating Personal Altair SLC with matlab sympy and r twoD cold plate problem

 CONTENTS (very siplistic examples)

   1 slc python matlab (via open source Octave)
     This is how you execute matlab code inside the altair slc.
     oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")

   2 slc smypy SLC, Exact solution to the one dimensional heat transfer problem,
     I do realize very few heat transfer problems have closed form solution
     but intial estimates bases on a closed form can be useful?

   3 slc r interative solution
     unfortunately Altair SLC dos not support ascii contour plots
     heatmap https://tinyurl.com/5n97pucx

     options ls=64 ps=32;
     proc plot data=wantxpox ;
     plot y*x=z / contour=5  box;

     ERROR: Option "contour" is not known for the PLOT statement

   4 slc r finite element solution cold plate


Related
https://tinyurl.com/5n97pucx
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/heatequ.png

github
https://tinyurl.com/2dxceary
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python

/*       _                    _   _                                  _   _       _
/ |  ___| | ___   _ __  _   _| |_| |__   ___  _ __   _ __ ___   __ _| |_| | __ _| |__
| | / __| |/ __| | `_ \| | | | __| `_ \ / _ \| `_ \ | `_ ` _ \ / _` | __| |/ _` | `_ \
| | \__ \ | (__  | |_) | |_| | |_| | | | (_) | | | || | | | | | (_| | |_| | (_| | |_) |
|_| |___/_|\___| | .__/ \__, |\__|_| |_|\___/|_| |_||_| |_| |_|\__,_|\__|_|\__,_|_.__/
                 |_|    |___/
*/

PROBLEM:
   Minimize yjr objective exfression
   (x(1)-3)^2 + (x(2)-5)^2
   Obviously the minimum is 3 and 5 for x(1) and x(2)

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Run the entire optimization as an Octave command
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")
oc.eval("result = fminsearch(objfun, [0, 0]);  ")

# Retrieve the result
result = oc.pull("result")
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;

OUTPUT
======

The PYTHON Procedure

Optimal solution found by Octave:

[[2.99995267 4.99983461]]

/*___                                      _     _  _                _          __
|___ \   ___ _   _ _ __ ___  _ __  _   _  / | __| || |__   ___  __ _| |_ __  __/ _| ___ _ __
  __) | / __| | | | `_ ` _ \| `_ \| | | | | |/ _` || `_ \ / _ \/ _` | __|\ \/ / |_ / _ \ `__|
 / __/  \__ \ |_| | | | | | | |_) | |_| | | | (_| || | | |  __/ (_| | |_  >  <|  _|  __/ |
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_|\__,_||_| |_|\___|\__,_|\__|/_/\_\_|  \___|_|
             |___/          |_|    |___/
*/

   EXPLANATION

   One dimensional heat distribution along a length of coldplate

           /    -T_in + T(x) \
         P*|1 - -------------|
   dT      \    -T_in + T_out/
   -- =  ---------------------
   dx         L*c_p*m_dot

   Initial Condition

   T_in
   T_out
   L
   m_dot
   c_p
   P

   EASY TO SOLVE BECAUSE
   ======================

      dt
      -- = t(x)   (everthing else in the heat equation are constants)
      dx
           1
    Then  --- dt = dx
           t

    Integrate both sides

       / 1       /
       \ - dt  = \ dx
       / t       /

      ln(t)    =  x + c (c=constent of integration)

      Lets exponentiate both sides

                      x
         t(x)  = c * e

   LETS LET SYMPY DO THE MESSY ALGEBRA
   ====================================

                                      P*x
                           --------------------------
                           L*c_p*m_dot*(T_in - T_out)
   T_out + (T_in - T_out)*e



&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
import sympy as sp
from sympy import symbols, exp, pi, sqrt, integrate, diff, simplify, pprint, erf

# Define symbols
x = sp.Symbol('x')
T = sp.Function('T')(x)
T_in, T_out, L, m_dot, c_p, P, h = sp.symbols('T_in T_out L m_dot c_p P h')

# Define the differential equation
dT_dx = (P / (m_dot * c_p * L)) * (1 - (T - T_in) / (T_out - T_in))
sp,pprint(dT_dx)
eq = sp.Eq(T.diff(x), dT_dx)

# Solve the differential equation
solution = sp.dsolve(eq, T, ics={T.subs(x, 0): T_in})

# Simplify the solution
simplified_solution = sp.simplify(solution.rhs)

print("Solution:")
sp.pprint(simplified_solution)
endsubmit;
;quit;run;

OUTPUT
======

Altair SLC

Solution:
                                   P*x
                        --------------------------
                        L*c_p*m_dot*(T_in - T_out)
T_out + (T_in - T_out)*e


/*____       _               ____     _             _     _        _       _
|___ /   ___| | ___   _ __  |___ \ __| |   ___ ___ | | __| | _ __ | | __ _| |_ ___
  |_ \  / __| |/ __| | `__|   __) / _` |  / __/ _ \| |/ _` || `_ \| |/ _` | __/ _ \
 ___) | \__ \ | (__  | |     / __/ (_| | | (_| (_) | | (_| || |_) | | (_| | ||  __/
|____/  |___/_|\___| |_|    |_____\__,_|  \___\___/|_|\__,_|| .__/|_|\__,_|\__\___|
                                                            |_|
*/

&_init_;
options noerrorabend;
options set=RHOME "D:\d451";
%utl_fkil(d:/png/heatequ.png);
proc r;
submit;
# Load necessary library
library(haven)
# Parameters
Lx <- 1       # Length of the plate in x-direction
Ly <- 1       # Length of the plate in y-direction
Nx <- 20      # Number of grid points in x-direction
Ny <- 20      # Number of grid points in y-direction
dx <- Lx / (Nx - 1)
dy <- Ly / (Ny - 1)
alpha <- 0.01 # Thermal diffusivity
dt <- 0.001   # Time step
Nt <- 100     # Number of time steps

# Initialize temperature grid
u <- matrix(0, nrow = Nx, ncol = Ny)

# Initial condition: some function f(x,y)
f <- function(x, y) {
  return(sin(pi * x / Lx) * sin(pi * y / Ly))
}

# Apply initial condition
for (i in seq_len(Nx)) {
  for (j in seq_len(Ny)) {
    x <- (i - 1) * dx
    y <- (j - 1) * dy
    u[i, j] <- f(x, y)
  }
}
# Time-stepping loop
for (n in seq_len(Nt)) {
  u_new <- u
  for (i in 2:(Nx-1)) {
    for (j in 2:(Ny-1)) {
      u_new[i, j] <- u[i, j] + alpha * dt * (
        (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 +
        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2
      )
    }
  }
  u <- u_new
}
str(u)
u <- as.matrix(u)
png("d:/png/heatequ.png")
# Plot the final temperature distribution
image(u, main="Temperature Distribution", xlab="X", ylab="Y", col=heat.colors(100))
dev.off()
rwant=as.data.frame(u)
rwant$id <- 1:nrow(rwant)
head(rwant)
endsubmit;
import data=rwant r=rwant;
run;quit;

&_init_;
proc transpose data=rwant out=wantxpo;
by id;
run;quit;

data wantxpox;
 set wantxpo;
 y=input(substr(_name_,2),3.);
 x=id;
 z=col1;
 drop _name_ col1;
run;quit;

proc print data=rwant;
format _numeric_ 4.2;
run;quit

/*--- does not work
options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;

/*  _         _         __ _       _ _             _                           _
| || |    ___| | ___   / _(_)_ __ (_) |_ ___   ___| | ___ _ __ ___   ___ _ __ | |_
| || |_  / __| |/ __| | |_| | `_ \| | __/ _ \ / _ \ |/ _ \ `_ ` _ \ / _ \ `_ \| __|
|__   _| \__ \ | (__  |  _| | | | | | ||  __/|  __/ |  __/ | | | | |  __/ | | | |_
   |_|   |___/_|\___| |_| |_|_| |_|_|\__\___| \___|_|\___|_| |_| |_|\___|_| |_|\__|

Example of solving a cold plate heat transfer problem using
Finite Element Analysis (FEA) in R. This example models a simple rectangular plate
with a cold boundary condition on one edge and a heat source in the center.
*/

&_init_;
proc r;
submit;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages

library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Shape functions for linear triangles
shape_functions <- function(xi, eta) {
  N <- c(1 - xi - eta, xi, eta)
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  return(list(N = N, dN_dxi = dN_dxi))
}

# Element stiffness matrix for heat conduction
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)
  invJ <- solve(J)

  # Shape function derivatives in global coordinates
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  dN_dx <- invJ %*% dN_dxi

  # Element stiffness matrix
  ke <- matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      ke[i, j] <- k * detJ * t(dN_dx[, i]) %*% dN_dx[, j] / 2
    }
  }

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)
T_ambient <- 20 # Ambient temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh
mesh <- create_rectangular_mesh(Lx, Ly, nx = 30, ny = 15)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = temperature
)


# Contour plot
fea1<-ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_contour_filled(bins = 20) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_d(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()
ggsave(filename="d:/png/fea1.png",plot=fea1)

# 3D surface plot
fea2<-ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_raster(aes(fill = temperature), interpolate = TRUE) +
  geom_contour(color = "white", alpha = 0.5, bins = 15) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution - Surface Plot",
       x = "X (m)", y = "Y (m)") +
  theme_minimal()

# Temperature profile along centerline
centerline_nodes <- which(mesh$nodes$y == Ly/2)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggsave(filename="d:/png/fea2.png",plot=fea2)

fea3<-ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()
ggsave(filename="d:/png/fea3.png",plot=fea3)


# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")

# Display temperature at specific points
cat("\nTemperature at key locations:\n")
cat("Center of plate (x=0.5, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 0.5) + abs(mesh$nodes$y - 0.25))], "°C\n")
cat("Right edge center (x=1.0, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 1.0) + abs(mesh$nodes$y - 0.25))], "°C\n")

endsubmit;
run;quit;

OUTPUT
======

Altair SLC

Assembling stiffness matrix...
Assembling load vector...
Applying boundary conditions...
Solving linear system...
=== SOLUTION SUMMARY ===
Minimum temperature: -0.0181723 °C
Maximum temperature: 0.1462442 °C
Cold boundary nodes: 15
Total nodes: 450
Total elements: 812
Temperature at key locations:
Center of plate (x=0.5, y=0.25): 0.1416964 °C
Right edge center (x=1.0, y=0.25): 8.317972e-08 °C


REPO
-----------------------------------------------------------------------------------------------------------------------------------------
https://github.com/rogerjdeangelis/utl-area-between-curves-with-an-intersection-point-adding-negative-and-positive-areas-plot-sympy
https://github.com/rogerjdeangelis/utl-calculating-the-cube-root-of-minus-one-with-drop-down-to-python-symbolic-math-sympy
https://github.com/rogerjdeangelis/utl-closed-form-solution-for-sample-size-in-a-clinical-equlivalence-trial-using-r-and-sas-and-sympy
https://github.com/rogerjdeangelis/utl-distance-between-a-point-and-curve-in-sql-and-wps-pythony-r-sympy
https://github.com/rogerjdeangelis/utl-fun-with-sympy-infinite-series-and-integrals-to-define-common-functions-and-constants
https://github.com/rogerjdeangelis/utl-maximum-likelihood-estimate-of--therate-parameter-lamda-of-a-Poisson-distribution-sympy
https://github.com/rogerjdeangelis/utl-maximum-liklihood-regresssion-wps-python-sympy
https://github.com/rogerjdeangelis/utl-mle-symbolic-solution-for-mu-and-sigma-of-normal-pdf-using-sympy
https://github.com/rogerjdeangelis/utl-python-sympy-projection-of-the-intersection-of-two-parabolic-surfaces-onto-the-xy-plane-AI
https://github.com/rogerjdeangelis/utl-r-python-compute-the-area-between-two-curves-AI-sympy-trapezoid
https://github.com/rogerjdeangelis/utl-roots-of-a-non-linear-function-using-python-sympy
https://github.com/rogerjdeangelis/utl-solve-a-system-of-simutaneous-equations-r-python-sympy
https://github.com/rogerjdeangelis/utl-symbolic-algebraic-simplification-of-a-polynomial-expressions-sympy
https://github.com/rogerjdeangelis/utl-symbolic-solution-for-the-gradient-of-the-cumulative-bivariate-normal-using-erf-and-sympy
https://github.com/rogerjdeangelis/utl-symbolically-solve-for-the-mean-and-variance-of-normal-density-using-expected-values-in-SymPy
https://github.com/rogerjdeangelis/utl-sympy-exact-pdf-and-cdf-for-the-correlation-coefficient-given-bivariate-normals
https://github.com/rogerjdeangelis/utl-sympy-technique-for-symbolic-integration-of-bivariate-density-function
https://github.com/rogerjdeangelis/utl-using-python-sympy-for-mathematical-characterization-of-the-human-face
https://github.com/rogerjdeangelis/utl-vertical-distance-covered-by-a-bouncing-ball-for-infinite-number-of-bounces-using-sympy



https://github.com/rogerjdeangelis/utl-runing-a-regression-using-matlab-syntax-using-the-open-source-r-octave-package






























































































































































































































































































































































%utl_rbeginx;
parmcards4;
&_init_;
proc r;
submit;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages

library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Shape functions for linear triangles
shape_functions <- function(xi, eta) {
  N <- c(1 - xi - eta, xi, eta)
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  return(list(N = N, dN_dxi = dN_dxi))
}

# Element stiffness matrix for heat conduction
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)
  invJ <- solve(J)

  # Shape function derivatives in global coordinates
  dN_dxi <- matrix(c(-1, -1, 1, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
  dN_dx <- invJ %*% dN_dxi

  # Element stiffness matrix
  ke <- matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      ke[i, j] <- k * detJ * t(dN_dx[, i]) %*% dN_dx[, j] / 2
    }
  }

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)
T_ambient <- 20 # Ambient temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh
mesh <- create_rectangular_mesh(Lx, Ly, nx = 30, ny = 15)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = temperature
)

png("d;/png/fea.png")
# Contour plot
ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_contour_filled(bins = 20) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_d(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()

# 3D surface plot
ggplot(plot_data, aes(x = x, y = y, z = temperature)) +
  geom_raster(aes(fill = temperature), interpolate = TRUE) +
  geom_contour(color = "white", alpha = 0.5, bins = 15) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution - Surface Plot",
       x = "X (m)", y = "Y (m)") +
  theme_minimal()

# Temperature profile along centerline
centerline_nodes <- which(mesh$nodes$y == Ly/2)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()

# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")

# Display temperature at specific points
cat("\nTemperature at key locations:\n")
cat("Center of plate (x=0.5, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 0.5) + abs(mesh$nodes$y - 0.25))], "°C\n")
cat("Right edge center (x=1.0, y=0.25):",
    temperature[which.min(abs(mesh$nodes$x - 1.0) + abs(mesh$nodes$y - 0.25))], "°C\n")

endsubmit;
run;quit;


 ;;;;%end;%mend;/*'*/ *);*};*];*/;/*"*/;run;quit;%end;end;run;endcomp;%utlfix;

%utl_rbeginx;
parmcards4;
# Cold Plate Heat Transfer FEA in R
# Using base R and easily installable packages

# Load required packages
library(Matrix)
library(ggplot2)
library(reshape2)

# ============================================================================
# MESH GENERATION
# ============================================================================

create_rectangular_mesh <- function(Lx = 1.0, Ly = 0.5, nx = 20, ny = 10) {
  # Create node coordinates
  x <- seq(0, Lx, length.out = nx)
  y <- seq(0, Ly, length.out = ny)
  nodes <- expand.grid(x = x, y = y)

  # Create triangular elements
  elements <- matrix(0, nrow = 2*(nx-1)*(ny-1), ncol = 3)
  elem_count <- 1

  for (j in 1:(ny-1)) {
    for (i in 1:(nx-1)) {
      # Lower triangle
      n1 <- (j-1)*nx + i
      n2 <- (j-1)*nx + i + 1
      n3 <- j*nx + i
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1

      # Upper triangle
      n1 <- j*nx + i + 1
      n2 <- j*nx + i
      n3 <- (j-1)*nx + i + 1
      elements[elem_count, ] <- c(n1, n2, n3)
      elem_count <- elem_count + 1
    }
  }

  return(list(nodes = nodes, elements = elements, nx = nx, ny = ny))
}

# ============================================================================
# FEA FUNCTIONS
# ============================================================================

# Element stiffness matrix for heat conduction - CORRECTED VERSION
element_stiffness <- function(nodes, k) {
  # nodes: 3x2 matrix of node coordinates
  x <- nodes[, 1]
  y <- nodes[, 2]

  # Jacobian matrix
  J <- matrix(c(x[2]-x[1], x[3]-x[1],
                y[2]-y[1], y[3]-y[1]), nrow = 2, byrow = TRUE)

  detJ <- det(J)

  # Check for degenerate element
  if (abs(detJ) < 1e-10) {
    return(matrix(0, 3, 3))
  }

  invJ <- solve(J)

  # Shape function derivatives in natural coordinates (xi, eta)
  # For linear triangles: N1 = 1 - xi - eta, N2 = xi, N3 = eta
  dN_dxi <- matrix(c(-1, 1, 0,   # dN/dxi for N1, N2, N3
                     -1, 0, 1),  # dN/deta for N1, N2, N3
                   nrow = 2, ncol = 3, byrow = TRUE)

  # Transform to global coordinates
  dN_dx <- invJ %%*%% dN_dxi

  # Element stiffness matrix: k * ?(?N? · ?N) dO
  # For constant Jacobian: ke = k * detJ * (dN_dx? %%*%% dN_dx) / 2
  ke <- k * detJ * (t(dN_dx) %%*%% dN_dx) / 2

  return(ke)
}

# Assemble global stiffness matrix
assemble_stiffness_matrix <- function(mesh, k) {
  n_nodes <- nrow(mesh$nodes)
  K <- Matrix(0, n_nodes, n_nodes, sparse = TRUE)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    ke <- element_stiffness(elem_nodes, k)

    for (i in 1:3) {
      for (j in 1:3) {
        K[node_indices[i], node_indices[j]] <- K[node_indices[i], node_indices[j]] + ke[i, j]
      }
    }
  }

  return(K)
}

# Assemble load vector for heat generation
assemble_load_vector <- function(mesh, heat_source_func) {
  n_nodes <- nrow(mesh$nodes)
  F <- numeric(n_nodes)

  for (elem in 1:nrow(mesh$elements)) {
    node_indices <- mesh$elements[elem, ]
    elem_nodes <- as.matrix(mesh$nodes[node_indices, ])

    # Element centroid for heat source evaluation
    centroid <- colMeans(elem_nodes)
    q_val <- heat_source_func(centroid[1], centroid[2])

    # Element area
    x <- elem_nodes[, 1]
    y <- elem_nodes[, 2]
    area <- abs(0.5 * (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])))

    # Distribute heat source equally to nodes
    for (i in 1:3) {
      F[node_indices[i]] <- F[node_indices[i]] + q_val * area / 3
    }
  }

  return(F)
}

# Apply Dirichlet boundary conditions
apply_dirichlet_bc <- function(K, F, bc_nodes, bc_values) {
  K_mod <- K
  F_mod <- F

  # Set rows and columns for boundary nodes
  for (i in seq_along(bc_nodes)) {
    node <- bc_nodes[i]
    value <- bc_values[i]

    # Modify stiffness matrix
    K_mod[node, ] <- 0
    K_mod[, node] <- 0
    K_mod[node, node] <- 1

    # Modify load vector
    F_mod[node] <- value
  }

  return(list(K = K_mod, F = F_mod))
}

# ============================================================================
# PROBLEM SETUP AND SOLUTION
# ============================================================================

# Parameters
Lx <- 1.0    # Plate length (m)
Ly <- 0.5    # Plate height (m)
k <- 200     # Thermal conductivity (W/m-K)
T_cold <- 0  # Cold boundary temperature (°C)

# Heat source function (circular heat source in center)
heat_source_func <- function(x, y) {
  # Circular heat source with radius 0.1 m at center
  if (sqrt((x - Lx/2)^2 + (y - Ly/2)^2) < 0.1) {
    return(50000)  # W/m³
  } else {
    return(0)
  }
}

# Create mesh (using smaller mesh for testing)
mesh <- create_rectangular_mesh(Lx, Ly, nx = 15, ny = 8)

# Assemble global system
cat("Assembling stiffness matrix...\n")
K_global <- assemble_stiffness_matrix(mesh, k)

cat("Assembling load vector...\n")
F_global <- assemble_load_vector(mesh, heat_source_func)

# Identify boundary nodes (left edge is cold)
left_boundary <- which(mesh$nodes$x == 0)
bc_nodes <- left_boundary
bc_values <- rep(T_cold, length(left_boundary))

# Apply boundary conditions
cat("Applying boundary conditions...\n")
system <- apply_dirichlet_bc(K_global, F_global, bc_nodes, bc_values)

# Solve system
cat("Solving linear system...\n")
temperature <- solve(system$K, system$F)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create data frame for plotting
plot_data <- data.frame(
  x = mesh$nodes$x,
  y = mesh$nodes$y,
  temperature = as.numeric(temperature)
)

# Contour plot
ggplot(plot_data, aes(x = x, y = y)) +
  geom_tile(aes(fill = temperature)) +
  geom_point(data = plot_data[left_boundary, ], aes(x = x, y = y),
             color = "blue", size = 1, alpha = 0.5) +
  scale_fill_viridis_c(name = "Temperature (°C)") +
  labs(title = "Cold Plate Temperature Distribution",
       subtitle = "Blue points show cold boundary condition",
       x = "X (m)", y = "Y (m)") +
  theme_minimal() +
  coord_equal()

# Temperature profile along centerline
centerline_nodes <- which(abs(mesh$nodes$y - Ly/2) < 1e-10)
centerline_data <- plot_data[centerline_nodes, ]
centerline_data <- centerline_data[order(centerline_data$x), ]

ggplot(centerline_data, aes(x = x, y = temperature)) +
  geom_line(linewidth = 1, color = "red") +
  geom_point(size = 1) +
  labs(title = "Temperature Profile Along Plate Centerline",
       x = "X Position (m)", y = "Temperature (°C)") +
  theme_minimal()

# Print summary statistics
cat("\n=== SOLUTION SUMMARY ===\n")
cat("Minimum temperature:", min(temperature), "°C\n")
cat("Maximum temperature:", max(temperature), "°C\n")
cat("Cold boundary nodes:", length(left_boundary), "\n")
cat("Total nodes:", nrow(mesh$nodes), "\n")
cat("Total elements:", nrow(mesh$elements), "\n")
;;;;
%utl_rendx;
























options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;




















































































































/*___                                            _  ____     _             _     _        _       _
|___ \   ___ _   _ _ __ ___  _ __  _   _  / | __| |  ___ ___ | | __| | _ __ | | __ _| |_ ___
  __) | / __| | | | `_ ` _ \| `_ \| | | | | |/ _` | / __/ _ \| |/ _` || `_ \| |/ _` | __/ _ \
 / __/  \__ \ |_| | | | | | | |_) | |_| | | | (_| || (_| (_) | | (_| || |_) | | (_| | ||  __/
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_|\__,_| \___\___/|_|\__,_|| .__/|_|\__,_|\__\___|
             |___/          |_|    |___/                     |_|
*/



import sympy as sp
from sympy import symbols, exp, pi, sqrt, integrate, diff, simplify, pprint, erf

# Define symbols
x = sp.Symbol('x')
T = sp.Function('T')(x)
T_in, T_out, L, m_dot, c_p, P, h = sp.symbols('T_in T_out L m_dot c_p P h')

# Define the differential equation
dT_dx = (P / (m_dot * c_p * L)) * (1 - (T - T_in) / (T_out - T_in))
sp,pprint(dT_dx)
eq = sp.Eq(T.diff(x), dT_dx)

# Solve the differential equation
solution = sp.dsolve(eq, T, ics={T.subs(x, 0): T_in})

# Simplify the solution
simplified_solution = sp.simplify(solution.rhs)

print("Solution:")
sp.pprint(simplified_solution)
;;;;
%utl_pyendx;


          2. Python 1 dimensional closed form solution.
             Temp(x) along the length of the cold plate
             Requires boundary temperatures
             (for symbolic mathematics I like sympy)

             Solution: (given boundary conditions)
                                                      P*x
                                           --------------------------
                                           L*c_p*m_dot*(T_in - T_out)
             T(x)= T_out + (T_in - T_out)*e




















































/*___                                      ____     _             _     _
|___ \   ___ _   _ _ __ ___  _ __  _   _  |___ \ __| |   ___ ___ | | __| |
  __) | / __| | | | `_ ` _ \| `_ \| | | |   __) / _` |  / __/ _ \| |/ _` |
 / __/  \__ \ |_| | | | | | | |_) | |_| |  / __/ (_| | | (_| (_) | | (_| |
|_____| |___/\__, |_| |_| |_| .__/ \__, | |_____\__,_|  \___\___/|_|\__,_|
             |___/          |_|    |___/
 __ _       _
| _| | __ _| |_ ___
| || |/ _` | __/ _ \
| || | (_| | ||  __/
| ||_|\__,_|\__\___|
|__|
*/


2 ALTAIR SLC, SOLVING THE TWOD COLD PLATE PROBLEM
=================================================

Too long to post here






https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python.sas


Graphic output 2 dimensional plate
https://tinyurl.com/5n97pucx
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python/blob/main/heatequ.png

github
https://tinyurl.com/2dxceary
https://github.com/rogerjdeangelis/utl-solving-one-and-two-dimensional-cold-plate-heat-equations-r-and-python











A simple, reproducible real-world optimization example
using oct2py with Python is to call Octave and pass data back and forth seamlessly.

A simple, reproducible real-world optimization example
using simlab interface with Python is to call simlab and pass data back and forth seamlessly.



https://www.perplexity.ai/search/a-simple-reproducible-real-wor-N85SP9ewQ5yQLWyvD7RcGQ

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Define and minimize the objective function in one go
result = oc.fminsearch('@(x) (x(1)-3)**2 + (x(2)-5)**2', [0, 0])

# Pass result back to Python (result is a numpy array)
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;

These are valid matlab statements

@(x) (x(1)-3)^2 + (x(2)-5)^2;
fminsearch(objfun, [0, 0]);

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from oct2py import Oct2Py
import numpy as np

# Initialize Oct2Py session
oc = Oct2Py()

# Run the entire optimization as an Octave command
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")
oc.eval("result = fminsearch(objfun, [0, 0]);  ")

# Retrieve the result
result = oc.pull("result")
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;























# Initialize Oct2Py session
oc = Oct2Py()

# Define the objective function in Octave as an anonymous function
oc.eval("objfun = @(x) (x(1)-3)^2 + (x(2)-5)^2;")

# Use Oct ave's fminsearch to minimize the objective function starting at [0,0]
result = oc.fminsearch('objfun', [0, 0])

# Pass result back to Python (result is a numpy array)
print("Optimal solution found by Octave:", result)
endsubmit;
quit;run;



# Example: SimLab Python Automation Script (Conceptual Demonstration)
&_init_;
proc python;
submit;
from hwx import simlab

# Load the simulated model directory/project
project_dir = "ConnectingRodModel"
model_file = f"{project_dir}/ConnectingRod.xmt_txt"

# Example XML template snippet for renaming a body in the model,
# this kind of XML string is extracted/generated by SimLab automation recording.
RenameBodyTemplate = """
<RenameBody CheckBox="ON" UUID="78633e0d-3d2f-4e9a-b075-7bff122772d8">
    <SupportEntities>
        <Entities>
            <Model>{model_file}</Model>
            <Body>"Body 2",</Body>
        </Entities>
    </SupportEntities>
    <NewName Value="{new_name}"/>
    <Output/>
</RenameBody>
"""

# Function to execute a rename operation in SimLab using the template
def rename_body_in_simlab(new_name):
    # Format the template for the desired new name
    command_xml = RenameBodyTemplate.format(model_file=model_file, new_name=new_name)
    # Call the simlab API to execute this command
    simlab.execute(command_xml)
    print(f"Renamed body to: {new_name}")

# Main workflow example: update a parameter, run simulation, get output
def run_simlab_example():
    # Pause recording before injecting parameters (known SimLab limitation)
    simlab.pause_recording()

    # Example: Rename a body (as a proxy to changing a parameter)
    rename_body_in_simlab("CHANGED_BODY_NAME")

    # Resume recording and run the model
    simlab.resume_recording()
    results = simlab.run_model(project_dir)

    # Extract output value(s) from results
    max_stress = results.get("MaxStress", None)
    max_displacement = results.get("MaxDisplacement", None)

    print(f"Max Stress: {max_stress}")
    print(f"Max Displacement: {max_displacement}")

    return max_stress, max_displacement

# Run the example workflow
if __name__ == "__main__":
    run_simlab_example()




# Example: SimLab Python Automation Script (Conceptual Demonstration)

&_init_;
options noerrorabend;
options set=PYTHONHOME "D:\python310";
proc python;
submit;
from hwx import simlab

# Load the simulated model directory/project
project_dir = "ConnectingRodModel"
model_file = f"{project_dir}/ConnectingRod.xmt_txt"

# Example XML template snippet for renaming a body in the model,
# this kind of XML string is extracted/generated by SimLab automation recording.
RenameBodyTemplate = """
<RenameBody CheckBox="ON" UUID="78633e0d-3d2f-4e9a-b075-7bff122772d8">
    <SupportEntities>
        <Entities>
            <Model>{model_file}</Model>
            <Body>"Body 2",</Body>
        </Entities>
    </SupportEntities>
    <NewName Value="{new_name}"/>
    <Output/>
</RenameBody>
"""

# Function to execute a rename operation in SimLab using the template
def rename_body_in_simlab(new_name):
    # Format the template for the desired new name
    command_xml = RenameBodyTemplate.format(model_file=model_file, new_name=new_name)
    # Call the simlab API to execute this command
    simlab.execute(command_xml)
    print(f"Renamed body to: {new_name}")

# Main workflow example: update a parameter, run simulation, get output
def run_simlab_example():
    # Pause recording before injecting parameters (known SimLab limitation)
    simlab.pause_recording()

    # Example: Rename a body (as a proxy to changing a parameter)
    rename_body_in_simlab("CHANGED_BODY_NAME")

    # Resume recording and run the model
    simlab.resume_recording()
    results = simlab.run_model(project_dir)

    # Extract output value(s) from results
    max_stress = results.get("MaxStress", None)
    max_displacement = results.get("MaxDisplacement", None)

    print(f"Max Stress: {max_stress}")
    print(f"Max Displacement: {max_displacement}")

    return max_stress, max_displacement

# Run the example workflow
if __name__ == "__main__":
    run_simlab_example()
endsubmit;
quit;run;


















Must be a problem with the personal edition

My setup does work


&_init_;
options set=PYTHONHOME "D:\python310";
PROC PYTHON;
submit;
x=2;
print(x);
ENDSUBMIT;
RUN;QUIT;

OUTPUT
======

Altair SLC

The PYTHON Procedure

2


ptions set=PYTHONHOME "D:\python310";
PROC PYTHON;
EXPORT DATA=TESTDATA PYTHON=df;
SUBMIT;
df.info()
dfo = df
ENDSUBMIT;
IMPORT DATA=TESTDATA PYTHON=dfo;
RUN;QUIT;

proc print data=dfo;
run;

189       options set=PYTHONHOME "D:\python310";
190       PROC PYTHON;
191       EXPORT DATA=TESTDATA PYTHON=df;
NOTE: Creating Python data frame 'df' from data set 'WORK.TESTDATA'
ERROR: Received unknown response [10]
192       SUBMIT;
193       df.info()
194       dfo = df
195       ENDSUBMIT;

NOTE: Submitting statements to Python:

WARNING: Could not cleanly terminate Python worker: Error writing to pipe : The pipe is being closed.


ERROR: Altair SLC has encountered a problem - please contact Altair Data Analytics Customer Support
196       IMPORT DATA=TESTDATA PYTHON=dfo;
197       quit; run;
198       ODS _ALL_ CLOSE;
199       FILENAME WPSWBHTM CLEAR;
ERROR: Expected a statement keyword : found "IMPORT"
200       RUN;QUIT;
201
202       proc print data=dfo;
                          ^
ERROR: Data set "WORK.dfo" not found
NOTE: Procedure PRINT was not executed because of errors detected
203       run;
NOTE: Procedure print step took :
      real time : 0.002
      cpu time  : 0.000





%utl_rbeginx;
parmcards4;
# Load necessary library
library(pracma)
library(haven)
source("c:/oto/fn_tosas9x.R")
# Parameters
Lx <- 1       # Length of the plate in x-direction
Ly <- 1       # Length of the plate in y-direction
Nx <- 20      # Number of grid points in x-direction
Ny <- 20      # Number of grid points in y-direction
dx <- Lx / (Nx - 1)
dy <- Ly / (Ny - 1)
alpha <- 0.01 # Thermal diffusivity
dt <- 0.001   # Time step
Nt <- 100     # Number of time steps

# Initialize temperature grid
u <- matrix(0, nrow = Nx, ncol = Ny)

# Initial condition: some function f(x,y)
f <- function(x, y) {
  return(sin(pi * x / Lx) * sin(pi * y / Ly))
}

# Apply initial condition
for (i in seq_len(Nx)) {
  for (j in seq_len(Ny)) {
    x <- (i - 1) * dx
    y <- (j - 1) * dy
    u[i, j] <- f(x, y)
  }
}

# Time-stepping loop
for (n in seq_len(Nt)) {
  u_new <- u
  for (i in 2:(Nx-1)) {
    for (j in 2:(Ny-1)) {
      u_new[i, j] <- u[i, j] + alpha * dt * (
        (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 +
        (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2
      )
    }
  }
  u <- u_new
}
str(u)
xy=as.data.frame(u)
head(xy)
png("d:/png/heatequ.png")
# Plot the final temperature distribution
image(xy, main="Temperature Distribution", xlab="X", ylab="Y", col=heat.colors(100))
dev.off()
fn_tosas9x(
      inp    = xy
     ,outlib ="d:/sd1/"
     ,outdsn ="want"
     )
;;;;
%utl_rendx;

libname sd1 "d:/sd1";
proc print data=sd1.want;
format _numeric_ 4.2;
run;quit

proc transpose data=sd1.want out=wantxpo;
by rownames;
run;quit;

data wantxpox;
 set wantxpo;
 y=input(substr(_name_,2),3.);
 x=rownames;
 z=col1;
 drop _name_ col1;
run;quit;

options ls=64 ps=32;
proc plot data=wantxpox(rename=y=y12345678901234567890);
plot y12345678901234567890*x=z / contour=5  box;
run;quit;
options ps=1000;
options ps=1000;
