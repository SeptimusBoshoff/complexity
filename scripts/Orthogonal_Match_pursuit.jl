using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra

T = 100

Waves = Array{Float64,2}(undef, T, T)
Actions = Array{Float64,2}(undef, 2, T)
Q = Array{Float64,1}(undef, T*T)
state = Array{Float64,1}(undef, T*T)
action = Array{Float64,1}(undef, T*T)

for i in 1:T

    Actions[1,i] = cos(0.2*i)# 2*exp(-1*norm(i - 50)/50) #cos(sin(4*i*i)) + 2*exp(-1*norm(i - 50)/50)
    Actions[2,i] = sin(0.1*i)# - 2*exp(-1*norm(i - 50)/50)

    for j in 1:T

        Waves[i,j] = cos(sin(4*i*j)) + 2*exp(-1*norm([i;j] - [50;50])/50)

    end
end

action[1] = 1
state[1] = 1
for k in 1:T*T

    if k > 2

        if action[k-1] == T
            state[k-1] = state[k-1] + 1
            action[k-1] = 1
        end

    end

    if k > 1

        action[k] = action[k-1] + 1
        state[k] = state[k-1]
    end

    Q[k] = cos(0.1*state[k]) + sin(0.2*action[k])

end

ζ = 10

data = collect(1:T)
Λ, C, error = OMP(collect(1:T), ζ, Actions[1,:])
println("error = ", round(error, digits = 3), " %")
fn = zeros(T)
for k in 1:T

    fn[k] = function_approximation(data[k], Λ, C, ζ)

end

data = collect(1:T)
Λ, C, error = OMP(collect(1:T), ζ, Actions)
println("error = ", round(error, digits = 3), " %")
fn = zeros(2,T)
for k in 1:T

    fn[:,k] = function_approximation(data[k], Λ, C, ζ)
end

data = transpose(hcat([action,state]))
Λ, C, error = OMP(data, ζ, Actions,0.5; kernel = Compatible_kernel, N = nothing, ϵ = 1e-6)
println("error = ", round(error, digits = 3), " %")
fn = zeros(2,T)
for k in 1:T

    fn[:,k] = function_approximation(data[k], Λ, C, ζ)
end

#-------------------------------------------------------------------------------------------
# Plots

#= println("error = ", round(error, digits = 3), " %")

Λ, C, error = OMP(collect(1:T), ζ, Q; N = 20)

Qt = zeros(T*T)
for k in 1:T

    fn[:,k] = function_approximation(data[:,k], Λ, C, ζ)
end

println("error = ", round(error, digits = 3), " %") =#

Waves_plot = plot(

                    surface(

                        contours = attr(

                            x=attr(show=true, start= 1.5, size=0.04, color="white"),
                            x_end=2,
                            z=attr(show=true, start= 0.5, size= 0.05),
                            z_end=0.8

                        ),

                        x = action,
                        y = state,

                        z = Q

                    ),

                    Layout(

                        scene=attr(
                            xaxis_nticks=20,
                            zaxis_nticks=4,
                            camera_eye=attr(x=0, y=-1, z=0.5),
                            aspectratio=attr(x=1, y=1, z=0.2)
                        )
                    )
                    )

#display(Waves_plot)

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

plot_DSS_3d = plot(traces,
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

#display(plot_DSS_3d)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
