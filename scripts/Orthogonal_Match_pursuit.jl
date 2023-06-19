using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra

T = 500

Actions = Array{Float64,2}(undef, 2, T)
Q = Array{Float64,1}(undef, T)
state = Array{Float64,1}(undef, T)
action = Array{Float64,1}(undef, T)

for i in 1:T

    Actions[2,i] = sin(0.1*i) - 2*exp(-1*norm(i - 250)/50)
    Actions[1,i] = cos(0.2*i) + Actions[2,i]

end

action[1] = 1
state[1] = 1

for k in 1:T

    action[k] = rand(Uniform(0, 1))
    state[k] = rand(Uniform(0, 1))

    Q[k] = cos(10*(state[k])) + sin(10*action[k])

end

ζ = 10

data = collect(1:T)
Λ, C, error = OMP(collect(1:T), ζ, Actions[1,:])

println("error = ", round(error, digits = 3), " %")
println("C size = ", 100*size(C,2)/T, " %")

fn = zeros(T)
for k in 1:T

    fn[k] = function_approximation(data[k], Λ, C, ζ)

end

data = collect(1:T)
Λ, C, error = OMP(collect(1:T), ζ, Actions)

println("error = ", round(error, digits = 3), " %")
println("C size = ", 100*size(C,2)/T, " %")

fn = zeros(2,T)
for k in 1:T

    fn[:,k] = function_approximation(data[k], Λ, C, ζ)
end

data = transpose(hcat([state, action]))

sizes = sum(abs2, data, dims = 1)
m = sqrt(maximum(sizes))
n = sqrt(minimum(sizes))

ζ = (m - n)/10

Λ, C, error = OMP(data, ζ, Q, 0.5)

println("error = ", round(error, digits = 3), " %")
println("C size = ", 100*size(C,2)/T, " %")

Q2 = zeros(T)
for k in 1:T

    Q2[k] = function_approximation(data[:,k], Λ, C, ζ)
end

#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(x=1:T, y=Actions[1,:], mode="lines"),
            scatter(x=1:T, y=Actions[2,:], mode="lines"),
            scatter(x = 1:T, y = fn[1,:], mode = "lines"),
            scatter(x = 1:T, y = fn[2,:], mode = "lines")]
Actions_plot = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Actions",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "x, y [m]",
                    ),
                )

display(Actions_plot)

traces = scatter3d(x = action, y = state, z = Q, mode="markers", marker = attr(
    color = Q,
    colorscale = "Cividis",
    size = 5),)

plot_Q = plot(traces,
                Layout(
                    title = attr(
                        text = "Q-value",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "actions",
                        yaxis_title = "states",
                        zaxis_title = "Q",
                    ),

                    scene_aspectratio = attr(x = 4, y = 4, z = 4),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )

#display(plot_Q)

traces = scatter3d(x = action, y = state, z = Q2, mode="markers", marker = attr(
    color = Q,
    colorscale = "Cividis",
    size = 5),)

plot_Q2 = plot(traces,
                Layout(
                    title = attr(
                        text = "Q-value",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "actions",
                        yaxis_title = "states",
                        zaxis_title = "Q",
                    ),

                    scene_aspectratio = attr(x = 4, y = 4, z = 4),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )

display(plot_Q2)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
