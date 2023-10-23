using Complexity
using Distributions
using DifferentialEquations
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using ArnoldiMethod
using NearestNeighbors

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........\n")

h(p, t) = [-1.1816783878162982; 1.490038688253868]

tau = 2
lags = [tau]

# system parameters
μ = 4
v0 = 1
v1 = 1
beta0 = 2
beta1 = 2

#Parameter vector
p = (μ, v0, v1, beta0, beta1, tau)

# initial condition
u0 = [1.9863606839964665, 0.12366121676846706]

# time series
Δt_s = 0.005 # numerical time step size for solver (seconds)
t_final_train = 150 # max simulation time (seconds)
subsampling = 5 # retain 1 in every so many samples
Δt_m = subsampling*Δt_s # machine time step

t_m = 0:Δt_m:t_final_train
tspan = (0.0, t_final_train)

prob = DDEProblem(Delayed_Van_der_Pol!, u0, h, tspan, p; constant_lags = lags)

alg = MethodOfSteps(Vern6())

sol = solve(prob, alg, saveat = Δt_s, wrap = Val(false))

data_train = reduce(hcat, sol(t_m).u)

data_train = data_train[:, 1000:end]

# ******************************************************************************************
# Plots

traces = [
            scatter(x = t_m, y = data_train[1,:], name = "x"),
            scatter(x = t_m, y = data_train[2,:], name = "y"),
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
ba = 4000
traces = [
                scatter(x = data_train[1,:], y = data_train[2,:], name = "train",
                        mode="lines",
                        marker=attr(
                        size = 3,),),
                ]

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

display(plot_dyns)
display(plot_SS)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
