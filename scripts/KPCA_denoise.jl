using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using ArnoldiMethod
using NearestNeighbors

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# System Dynamics

# system parameters
μ = 2

Δt_s = 0.005 # numerical time step size (seconds)

t_final_train = 100 # max simulation time (seconds)

#Parameter vector
p = [μ]

#-------------------------------------------------------------------------------------------
# Machine parameters

sampling = 5 # machine retain 1 in every so many samples - subsampling

history = 5 # backward trajectory [seconds]
future = 5 # forward trajectory [seconds]

# ******************************************************************************************

Δt_m = sampling*Δt_s # machine time step
npast = convert(Int, round(history/Δt_m)) #Past Series sample size
nfuture = convert(Int, round(future/Δt_m)) #Future series sample size

window_size = npast + nfuture

#-------------------------------------------------------------------------------------------
# Training Data - time simulation

u0 = [1.012, -0.88]
#u0 = [-2, 10]

# ******************************************************************************************

t_s_train = 0:Δt_s:t_final_train # system time
t_m_train= 0:Δt_m:t_final_train # machine time
tspan = (0.0, t_final_train)

prob_train = ODEProblem(Van_der_Pol!, u0, tspan, p)

sol_train = solve(prob_train, saveat = Δt_s, wrap = Val(false))
u_train = reduce(hcat, sol_train.u)

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

# ******************************************************************************************
# Training data

data_train = reduce(hcat, sol_train(t_m_train).u)

N = size(data_train, 2) # number of data points
dimX = size(data_train, 1) # dimension of dynamical system - coordinates

radiusn = 3
θ = range(0, stop = 2π, length = N) # N points around the circle
data_train[1,:] = radiusn * cos.(θ) # x-coordinates of the circle points
data_train[2,:] = radiusn * sin.(θ)

#= Sx = data_train  .+ 0.0*randn(dimX, N) .+ 0.0*ones(dimX, N)
Sx_v = data_train  .+ 0.0*randn(dimX, N) .- 0.0*ones(dimX, N) =#
Sx = data_train  .+ 0.5*rand(Uniform(-1,1), dimX, N) .+ 0.5*ones(dimX, N)
Sx_v = data_train  .+ 0.5*rand(Uniform(-1,1),dimX, N) .- 0.5*ones(dimX, N)

#-------------------------------------------------------------------------------------------
# Gramian

K_NN = convert(Int, round(N/2)) # the fraction of the sample size

Sx_avg_dist, Sx_tree = Mean_Distance(Sx, KNN = true, K = K_NN)

#Sx_avg_dist = 2.6440187111760816/20

ηx = 1/(2*(Sx_avg_dist^2)) # inverse kernel bandwidth, scale: avg_dist = std

#= average_distance = Mean_Distance(Sx) / 1.7
ηx = 1/(2*average_distance^2) # inverse kernel bandwidth, scale
#ηx = 1.9005851476871414 =#

Gxx = Gramian(Sx, ηx) # training set
Gxx_v = Gramian(Sx_v, ηx) # validation set

Gaussian_kernel(Sx[:, 1], Sx[:, 2], ηx)

# ******************************************************************************************
# Whitening - decorrelation

#-------------------------------------------------------------------------------------------
# Centring

One_N = (1/N)*ones(N,N)
H = I - One_N

Gxx_c = H*(Gxx)*H # Gxx - One_N*Gxx - Gxx*One_N + One_N*Gxx*One_N
Gxx_vc = H*(Gxx)*H

#-------------------------------------------------------------------------------------------
# Kernel PCA

x_dims = 2

decomp, history  = partialschur(Gxx_c, nev = x_dims,
tol = 1e-6, restarts = 200, which = LM())
Λx, Vx = partialeigen(decomp)

Λx = real.(Λx[end:-1:1]./(N-1)) # principal component variances
Vx = real.(Vx[:, end:-1:1]) # right eigenvectors

#Λx, Vx = eigen(Gxx_c)

#-------------------------------------------------------------------------------------------
# Normalising

normsx = sqrt.(sum((Vx.^2)*(N-1)*Diagonal(Λx), dims=1))
Vx = Vx ./ (normsx) # the coefficients for the principal directions

Gxx_pc = transpose(Vx)*Gxx_c # the principal components

# verify: transpose(Vx)*Gxx_c*Vx ≈ (N-1)*Diagonal(Λx)*transpose(Vx)*Vx ≈ I

#-------------------------------------------------------------------------------------------
# Covariance matrices

Cxx_pc = Diagonal(Λx)

# verify: Cxx_pc ≈ (1/(N-1))*Gxx_pc*transpose(Gxx_pc)

# ******************************************************************************************
# Conformal map

#= Gxx_dc = Vx*Gxx_vpc # denoised and centred Gram matrix
ϵ = 1e-9
ε = 1e-5

#M = inv(Sx*transpose(Sx))*Sx*(transpose(Sx)*Sx - ε*inv(Gxx))
Gxx_inv = (Gxx + ϵ*I)\Matrix{Float64}(I,N,N)
M = pinv(transpose(Sx))*(transpose(Sx)*Sx - ε*Gxx_inv)

Dx = M*Gxx_dc =#

Gxx_vpc = transpose(Vx)*Gxx_vc
ε = 1e-9
M = pinv(transpose(Sx))*(transpose(Sx)*Sx - ε*I)*pinv(Gxx_pc)
Dx = M*Gxx_vpc

# ******************************************************************************************
# Plots

traces = [
            scatter(x = t_m_train, y = data_train[1,:], name = "x_train"),
            scatter(x = t_m_train, y = data_train[2,:], name = "y_train"),
            scatter(x = t_m_train, y = Sx[1,:], name = "x_noisy"),
            scatter(x = t_m_train, y = Sx[2,:], name = "y_noisy"),
            scatter(x = t_m_train, y = Dx[1,:], name = "x_clean"),
            scatter(x = t_m_train, y = Dx[2,:], name = "y_clean"),
            ]

plot_dyns = plot(traces,
                    Layout(
                        title = attr(
                            text = "Episodic State",
                            ),
                        title_x = 0.5,
                        xaxis_title = "time [s]",
                        yaxis_title = "x [m], y [m]",
                        ),
                    )

#-------------------------------------------------------------------------------------------

traces = [
            scatter(x = data_train[1,:], y = data_train[2,:], name = "truth"),
            #= scatter(x = Sx[1,:], y = Sx[2,:], name = "train",
            mode="markers",
            marker=attr(
            size = 3,),), =#
            scatter(x = Sx_v[1,:], y = Sx_v[2,:], name = "validation",
            mode="markers",
            marker=attr(
            size = 3,),),
            scatter(x = Dx[1,:], y = Dx[2,:], name = "cleaned",
            mode="markers",
            marker=attr(
            size = 3,),)]

plot_SS = plot(traces,
                    Layout(
                        title = attr(
                            text = "State Space",
                            ),
                        title_x = 0.5,
                        xaxis_title = "x [m]",
                        yaxis_title = "y [m]",
                        ),
                    )
# ******************************************************************************************
# Display

#display(plot_dyns)
display(plot_SS)

show(to)

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
