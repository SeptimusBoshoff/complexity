using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using Random

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# Generate Data

T = 500

Actions = Array{Float64,2}(undef, 2, T)
Q = Array{Float64,1}(undef, T)
state = Array{Float64,1}(undef, T)
action = Array{Float64,1}(undef, T)

for i in 1:T

    Actions[2,i] = sin(0.1*i) - 2*exp(-1*norm(i - T/2)/50)
    Actions[1,i] = cos(0.2*i) + Actions[2,i] + 2.0*rand()

end

maxA2 = maximum(Actions[2,:])
minA2 = minimum(Actions[2,:])
maxA1 = maximum(Actions[1,:])
minA1 = minimum(Actions[1,:])

Actions[2,:] = 2*(Actions[2,:] .- minA2)/(maxA2 - minA2) .- 1
Actions[1,:] = 2*(Actions[1,:] .- minA1)/(maxA1 - minA1) .- 1

action[1] = 1
state[1] = 1

Random.seed!(123)

for k in 1:T

    action[k] = rand(Uniform(-1, 1))
    state[k] = rand(Uniform(-1, 1))

    Q[k] = 0.5*cos(10*(state[k])) + 0.5*sin(10*action[k])

end

#-------------------------------------------------------------------------------------------
# 1-D OMP

η = 100
ϵ = 1e-6

data1 = 4*(collect(1:T)/maximum(collect(1:T))) .- 2

Λ1a, C1, β1a, error1, Press1a = OMP(data1, η, Actions[1,:], 1.0, N = 600, ϵ = ϵ, PRESS = true, sparsity = 1.1)

Λ1b, β1b, ϵ1b, η1b, Press1b, verge = CGD(data1, Actions[1,:]; ϵ = ϵ, η = η)

println("\nerror = ", round(error1, digits = 3), " %")
println("C size = ", 100*size(C1,2)/T, " %")
println("Press1a = ", round(Press1a, digits = 3))
println("Press1b = ", round(Press1b, digits = 3))
println("η1b = ", round(η1b, digits = 3))
println("ϵ1b = ", round(ϵ1b*1e6, digits = 3),"e-6")
println("converged = ", verge)

fn1 = zeros(T)
for k in 1:T

    fn1[k] = function_approximation(data1[k], Λ1b, C1, η1b, β = β1b)

end

#-------------------------------------------------------------------------------------------
# 2-D OMP

η = 50
ϵ = 1e-6

data2 = 4*(collect(1:T)/maximum(collect(1:T))) .- 2
Λ2, C2, β2, error2, Press2 = OMP(data2, η, Actions, 1.0, ϵ = 1e-6, PRESS = true, N = 50, sparsity = 1.1)

#= println("\nerror = ", round(error2, digits = 3), " %")
println("C size = ", 100*size(C2,2)/T, " %")
println("Press = ", round.(Press2, digits = 3)) =#

fn2 = zeros(2,T)
for k in 1:T

    fn2[:,k] = function_approximation(data2[k], Λ2, C2, η, β = β2)
end

#-------------------------------------------------------------------------------------------
# Q OMP

data = transpose(hcat(state, action))

η = 4

Λ, C, β, error, Press3  = OMP(data, η, Q, 1.0, PRESS = true, #= N = 100, =# sparsity = 0.1)

#= println("\nerror = ", round(error, digits = 3), " %")
println("C size = ", 100*size(C,2)/T, " %")
println("Press = ", round(Press3, digits = 3)) =#

n = 20
Q2 = zeros(n*T)
state = Array{Float64,1}(undef, n*T)
action = Array{Float64,1}(undef, n*T)
for k in 1:n*T

    action[k] = rand(Uniform(-1, 1))
    state[k] = rand(Uniform(-1, 1))
    Q2[k] = function_approximation([state[k], action[k]], Λ, C, η, β = β)
end

#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(x=1:T, y=Actions[1,:], mode="lines", name = "Actions a"),
            scatter(x=1:T, y=Actions[2,:], mode="lines", name = "Actions b"),
            scatter(x = 1:T, y = fn1, mode = "lines", name = "OMP 1 a"),
            scatter(x = 1:T, y = fn2[1,:], mode = "lines", name = "OMP 2 a"),
            scatter(x = 1:T, y = fn2[2,:], mode = "lines", name = "OMP 2 b")]
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

                    scene_aspectratio = attr(x = 7, y = 7, z = 7),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )

#display(plot_Q)

traces = scatter3d(x = action, y = state, z = Q2, mode="markers", marker = attr(
    color = Q2,
    colorscale = "Cividis",
    size = 5),)

plot_Q_approx = plot(traces,
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

                    scene_aspectratio = attr(x = 7, y = 7, z = 7),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )

#display(plot_Q_approx)

show(to)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")

#= η = 12#/(2*σ^2)
ϵ = 1e-6

D = Gramian(data1, kernel = Distance_kernel)
G, ∂G = exp_mat(D, η, derivative = true)
Λ1b, β1b, Ch, ρ = kernel_regression(G, Actions[1,:], ϵ)

Press1b = Press(Ch, Λ1b, ρ)
∇P1c, Press1c, Λ1c, β1c = Weights_and_Gradients(Actions[1,:], D, ϵ, η)

Λ1d, β1d, ϵans, ηans, Pmin = CGD(data1, Actions[1,:]; ϵ = 1e-6, η = 10.0, D = D)

println("P cgd = ", round(Pmin, digits = 3))
println("ϵ cgd = ", round(ϵans*1e6, digits = 3), "e-6")
println("η cgd = ", round(ηans, digits = 3))

ϵt = collect(1e-8:1e-7:1e-4)
gr = length(ϵt)
Press_ϵt = Array{Float64, 1}(undef, gr)
∇P_ϵt = Array{Float64, 1}(undef, gr)

for i in eachindex(ϵt)

    global D

    #Λ1t, β1t, Ct, ρt = kernel_regression(G, Actions[1,:], ϵt[i])
    #Press1t = Press(Ct, Λ1t, ρt)
    ∇Pt, Press1t, Λ1t, β1t = Weights_and_Gradients(Actions[1,:], D, ϵt[i], 12)

    Press_ϵt[i] = Press1t
    ∇P_ϵt[i] = ∇Pt[1]


end

println("\nminimum Press_ϵt = ", round(minimum(Press_ϵt), digits = 3))
println("ϵ = ", round(ϵt[argmin(Press_ϵt)]*1e6, digits = 3), "e-6")

ηt = collect(1:1.0:1000.)
gr = length(ηt)
Press_ηt = Array{Float64, 1}(undef, gr)
∇P_ηt = Array{Float64, 1}(undef, gr)

for i in eachindex(ηt)

    global D

    #Λ1t, β1t, Ct, ρt = kernel_regression(G, Actions[1,:], ϵt[i])
    #Press1t = Press(Ct, Λ1t, ρt)
    ∇Pt, Press1t, Λ1t, β1t = Weights_and_Gradients(Actions[1,:], D, 1e-6, ηt[i])

    Press_ηt[i] = Press1t
    ∇P_ηt[i] = ∇Pt[2]


end

println("minimum Press_ηt = ", round(minimum(Press_ηt), digits = 3))
println("η = ", round(ηt[argmin(Press_ηt)], digits = 3))

traces = [scatter(x = log.(10, ϵt), y = log.(2, Press_ϵt), mode="lines", name = "Pressϵ"),
            scatter(x = log.(10, ϵt), y =∇P_ϵt, mode="lines", name = "∇P_ϵ"),]

Press_tϵ = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Press ϵ",
                        ),
                    title_x = 0.5,
                    xaxis_title = "log(10, ϵ)",
                    yaxis_title = "log(10, Press_ϵ), ∇P_ϵ",
                    ),
                )

traces = [scatter(x = ηt, y = log.(2, Press_ηt), mode="lines", name = "Pressη"),
            scatter(x = ηt, y =∇P_ηt, mode="lines", name = "∇P_η"),]

Press_tη = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Press η",
                        ),
                    title_x = 0.5,
                    xaxis_title = "η",
                    yaxis_title = "log(10, Press_η), ∇P_η",
                    ),
                )
display(Press_tϵ)
display(Press_tη) =#
