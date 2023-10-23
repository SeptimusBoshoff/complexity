using Complexity
using Distributions
using DataFrames
using PlotlyJS
using LinearAlgebra
using TimerOutputs
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Zygote
using NearestNeighbors

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# Parameters

N = 100 # iterations
Mx = 1000 # state space samples - support vectors
Ma = 25 # action space samples - support vectors

# discounting factor
γ = 0.95

# inverse kernel bandwidth, scale
average_distance = Mean_Distance(X_train) / 4
η = 1/(2*average_distance^2)
#σ = sqrt(2*η)
η_V = 3.5
η_π = 3.5
η_Q = 0.3

ϵ_V = 5e-4 #1e-4
ϵ_Q = 1e-8

# dynamical system parameters
max_torque = 2.0
max_speed = 8.0

initialize = true

#-------------------------------------------------------------------------------------------
# dataframes

inner_hook = DataFrame()
episode_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# ancillary

# scaling
θ_scale = 1/π
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = 1/((π)^2 + 0.01*(max_speed^2) + 0.001*(max_torque^2))

if initialize

    # Samples

    X_state = rand(Uniform(-1, 1), 2, Mx)
    X_state[1,:] = π*X_state[1,:]
    X_state[2,:] = max_speed*X_state[2,:]

    X_train = Array{Float64,2}(undef, 3, Mx)
    X_train[1,:] = cos.(X_state[1,:])
    X_train[2,:] = sin.(X_state[1,:])
    X_train[3,:] = ω_scale*(X_state[2,:])

    A = collect(range(-max_torque, stop=max_torque, length=Ma))
    A[1] = A[1] + 1e-8
    A[end] = A[end] - 1e-8
    Qx_i = Array{Float64,1}(undef, Ma)

    X_train_nxt = Array{Float64, 3}(undef, 3, Mx, Ma)
    reward = Array{Float64, 2}(undef, Mx, Ma)

    for i in 1:Mx

        for j in 1:Ma

            x_next= pendulum(X_state[:, i], A[j], max_torque = max_torque, max_speed = max_speed)
            X_train_nxt[:, i, j] = [cos(x_next[1]); sin(x_next[1]); ω_scale*x_next[2]]
            #reward[i, j] = -(x_next[1]^2 + 0.01*(x_next[2]^2) + 0.001*(A[j]^2))*reward_scale
            reward[i, j] = -(X_state[1, i]^2 + 0.01*(X_state[2, i]^2) + 0.001*(A[j]^2))*reward_scale
        end
    end

    # ***************************
    # initial values

    πk = Array{Float64, 1}(undef, Mx)
    πk = fill!(πk, 0.)

    Vk = zeros(Mx)
    Vk_1 = Array{Float64, 1}(undef, Mx)
    Λ_V = zeros(1,2)
    C_V = ones(3, 2)
    β_V = 0
end

# gradient descent
optf = OptimizationFunction(Q_max2, Optimization.AutoZygote())

#-------------------------------------------------------------------------------------------
# Training

for k in N-1:-1:0

    global Λ_V, C_V, β_V, Vk, Vk_1, ϵ_V, η_V
    global Qx_i, ϵ_Q, η_Q
    global πk

    for i in 1:Mx

        for j in 1:Ma

            V_next = function_approximation(X_train_nxt[:, i, j], Λ_V, C_V, η_V, β = β_V)

            Qx_i[j] = reward[i, j] + γ*V_next

        end

        if k < 10

            Λ_Q, C_Q, β_Q, ϵ_Q, η_Q, Press, verge = CGD(A, Qx_i; ϵ = ϵ_Q, η = η_Q)
            C_Q = transpose(C_Q)

            if !verge
                println("CGD has not converged")
            end

            global Λ_Q, C_Q, β_Q, ϵ_Q, η_Q

            Q_params = (Λ_Q, C_Q, η_Q, β_Q[1,1])

            prob = OptimizationProblem(optf, [πk[i]*max_torque], Q_params, lb = [-max_torque], ub = [max_torque])
            sol = solve(prob, LBFGS(), maxiters = 5)

            A_max = sol.u[1]
            Vk_1[i] = function_approximation(A_max, Λ_Q, C_Q, η_Q, β = β_Q)
            πk[i] = A_max*a_scale

        else

            Vk_1[i], jdx = findmax(Qx_i)
            πk[i] = A[jdx]*a_scale
        end

    end

    Λ_V, C_V, β_V, _ = OMP(X_train, η_V, Vk_1, 1.0, N = Mx, ϵ = ϵ_V)

    Vk_change = norm(Vk .- Vk_1)/Mx
    Vk = copy(Vk_1)

    #***************************************************************************************
    # verbosity

    println("\nk = ", k)
    println("Vk_change = ", round(Vk_change*1e3, digits = 3),"e-3")
    #= println("ϵ_V = ", round(ϵ_V*1e3, digits = 3), "e-3")
    println("η_V = ", round(η_V, digits = 3))
    println("ϵ_Q = ", round(ϵ_Q*1e3, digits = 3), "e-3")
    println("η_Q = ", round(η_Q, digits = 3))
    println("converged = ", verge) =#

end

πp_idx = findall(x -> x > 0, πk)
πn_idx = setdiff(1:Mx, πp_idx)

π_switch = zeros(Mx)
π_switch[πp_idx] .= 1

η_class = 50
μ_weights, σ²_weights, C, verge = GP(X_train, π_switch, η_class; ϵ = 1e-8, max_itr = 50, tol = 1e-9);

η_π = 3.5
Λ_πp, C_πp, β_πp, σ2_πp = RVM(X_train[:,πp_idx], πk[πp_idx], η_π, noise_itr = 2, σ2 = 0.03)
Λ_πn, C_πn, β_πn, σ2_πn = RVM(X_train[:,πn_idx], πk[πn_idx], η_π, noise_itr = 2, σ2 = 0.03)

#= η_π = 7.
Λ_π, C_π, β_π, σ2_π = RVM(X_train, πk, η_π, noise_itr = 0, σ2 = 0.02) =#

#= η_π = 7.
ϵ_π = 4e-3
Λ_π, C_π, β_π, _ = OMP(X_train, η_π, πk, 1.0, N = Mx, ϵ = ϵ_π) =#

#Λ_π, C_π, β_π, ϵ_π, η_π, Press, verge1 = CGD(X_train, πk; ϵ = ϵ_π, η = η_π, ϵ_max = 5e-3, η_max = 100, Smax = 2);
#Λ_V1, C_V1, β_V1, ϵ_V1, η_V1, Press, verge1 = CGD(X_train, Vk_1; ϵ = ϵ_V, η = η_V, ϵ_max = 5e-3, η_max = 100, Smax = 2);

#-------------------------------------------------------------------------------------------
# Validation

inner_hook = DataFrame()
T = 200

x_state = [π, 0.0] # initial condition

for k in 1:T

    global x_state, inner_hook

    x_train = [cos(x_state[1]); sin(x_state[1]); ω_scale*x_state[2]]

    # ------
    # One policy

    #action = function_approximation(x_train, Λ_π, C_π, η_π, β = β_π)

    # -----
    # Two policy

    p = GP_predict(x_train, μ_weights, σ²_weights, C, η_class)

    if p >= 0.5

        action = function_approximation(x_train, Λ_πp, C_πp, η_π, β = β_πp)

    elseif p < 0.5

        action = function_approximation(x_train, Λ_πn, C_πn, η_π, β = β_πn)

    else

        action_p = function_approximation(x_train, Λ_πp, C_πp, η_π, β = β_πp)
        action_n = function_approximation(x_train, Λ_πn, C_πn, η_π, β = β_πn)

        action = p*action_p + (1 - p)*action_n
    end

    # -----

    action = clamp(action*max_torque, -max_torque, max_torque)

    R = -((x_state[1])^2 + 0.01*(x_state[2]^2) + 0.001*(action^2))*reward_scale

    #-------------------------------------------------------------
    # Manage data

    sample_hook = DataFrame(t = k,
                            y = x_train[1], x = x_train[2], ω = x_train[3],
                            a = a_scale*action,
                            r = R,
                            θ = θ_scale*x_state[1])

    append!(inner_hook, sample_hook)

    #-------------------------------------------------------------
    # evolve

    x_state = pendulum(x_state, action, max_torque = max_torque, max_speed = max_speed)

    x_state = x_state .+ 0.0*randn(2)

end

M_val = 10000
V_approx = Array{Float64, 1}(undef, M_val)
π_approx = Array{Float64, 1}(undef, M_val)

X_st_app = rand(Uniform(-1, 1), 2, M_val)
X_st_app[1,:] = π*X_st_app[1,:]
X_st_app[2,:] = max_speed*X_st_app[2,:]

X_tr_app = Array{Float64,2}(undef, 3, M_val)
X_tr_app[1,:] = cos.(X_st_app[1,:])
X_tr_app[2,:] = sin.(X_st_app[1,:])
X_tr_app[3,:] = ω_scale*X_st_app[2,:]

for i in 1:M_val

    V_approx[i] = function_approximation(X_tr_app[:, i], Λ_V, C_V, η_V, β = β_V)

    p = GP_predict(X_tr_app[:, i], μ_weights, σ²_weights, C, η_class)

    if p > 0.5

        π_approx[i] = function_approximation(X_tr_app[:, i], Λ_πp, C_πp, η_π, β = β_πp)

    else
        π_approx[i] = function_approximation(X_tr_app[:, i], Λ_πn, C_πn, η_π, β = β_πn)
    end

    #π_approx[i] = function_approximation(X_tr_app[:, i], Λ_π, C_π, η_π, β = β_π)

    π_approx[i] = clamp(π_approx[i], -1, 1)

end

π_mat = Array{Float64,2}(undef, 200, 200)
P_mat = Array{Float64,2}(undef, 200, 200)
V_mat = Array{Float64,2}(undef, 200, 200) #[function_approximation([cos(θ); sin(θ); ω_scale*ω], Λ_V, C_V, η_V, β = β_V) for ω in ω_range, θ in θ_range]

θ_range = range(-π, stop = π, length = 200)
ω_range = range(-max_speed, stop = max_speed, length = 200)

for ω in eachindex(ω_range)

    for θ in eachindex(θ_range)

        x = [cos(θ_range[θ]); sin(θ_range[θ]); ω_scale*ω_range[ω]]

        V_mat[ω, θ] = function_approximation(x, Λ_V, C_V, η_V, β = β_V)

        P_mat[ω, θ] = GP_predict(x, μ_weights, σ²_weights, C, η_class)

        if P_mat[ω, θ] > 0.5

            π_mat[ω, θ] = function_approximation(x, Λ_πp, C_πp, η_π, β = β_πp)

        else
            π_mat[ω, θ] = function_approximation(x, Λ_πn, C_πn, η_π, β = β_πn)
        end

        π_mat[ω, θ] = clamp(π_mat[ω, θ], -1, 1)
    end
end

#-------------------------------------------------------------------------------------------
# Plots

plot_df = inner_hook[1:end, :]

traces = [scatter(plot_df, x = :t, y = :x, name = "x"),
            scatter(plot_df, x = :t, y = :y, name = "y"),
            scatter(plot_df, x = :t, y = :ω, name = "ω"),
            scatter(plot_df, x = :t, y = :θ, name = "θ"),
            scatter(plot_df, x = :t, y = :r, name = "reward"),
            scatter(plot_df, x = :t, y = :a, name = "actions"),]

plot_episode = plot(traces,
                Layout(
                    title = attr(
                        text = "Episodic State",
                        ),
                    title_x = 0.5,
                    xaxis_title = "steps [k]",
                    yaxis_title = "x [m], y [m], θ [rad/π], ω [rad/s⋅8]",
                    ),
                )
#---

traces_V = scatter3d(x = vec(X_state[1,:]), y = vec(X_state[2,:]), z = Vk, mode="markers", marker = attr(
    color = Vk,
    colorscale = "Cividis",
    size = 5),
    name = "Value-function")

traces_V_approx = scatter3d(x = vec(X_st_app[1,:]), y = vec(X_st_app[2,:]), z = V_approx, mode="markers", marker = attr(
    color = V_approx,
    colorscale = "Electric",
    size = 5),
    name = "approximation-function")

traces = [traces_V_approx]
plot_V = plot(traces,
                Layout(
                    title = attr(
                        text = "V-value",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "θ",
                        yaxis_title = "ω",
                        zaxis_title = "V",
                    ),

                    scene_aspectratio = attr(x = 7, y = 7, z = 7),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )
#---

traces_π = scatter3d(x = vec(X_state[1,:]), y = vec(X_state[2,:]), z = πk, mode="markers", marker = attr(
    color = πk,
    colorscale = "Cividis",
    size = 5),
    name = "policy-function")

traces_π_approx = scatter3d(x = vec(X_st_app[1,:]), y = vec(X_st_app[2,:]), z = π_approx, mode="markers", marker = attr(
    color = π_approx,
    colorscale = "Electric",
    size = 5),
    name = "approximation-function")

traces = [traces_π_approx]
plot_π = plot(traces,
                Layout(
                    title = attr(
                        text = "π-value",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "θ",
                        yaxis_title = "ω",
                        zaxis_title = "π",
                    ),

                    scene_aspectratio = attr(x = 7, y = 7, z = 7),
                    scene_camera = attr(
                        up = attr(x = 0, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 0, y = 10, z = 10)
                        ),
                    ),
                )
#---

trace_V = heatmap(x = θ_range, y = ω_range, z = V_mat, colorscale = "Electric")
layout = Layout(title = "Value function", xaxis_title = "θ [rad]", yaxis_title = "ω [rad/s]")
plot_heatmap_V = Plot(trace_V, layout)
#---

trace_π = heatmap(x = θ_range, y = ω_range, z = π_mat, colorscale = "Electric")
layout = Layout(title = "policy function", xaxis_title = "θ [rad]", yaxis_title = "ω [rad/s]")
plot_heatmap_π = Plot(trace_π, layout)
#---

trace_P = heatmap(x = θ_range, y = ω_range, z = P_mat, colorscale = "Electric")
layout = Layout(title = "Classification", xaxis_title = "θ [rad]", yaxis_title = "ω [rad/s]")
plot_heatmap_P = Plot(trace_P, layout)

#-------------------------------------------------------------------------------------------
# Display

display(plot_V)
display(plot_π)
display(plot_heatmap_V)
display(plot_heatmap_π)
display(plot_heatmap_P)
display(plot_episode)

show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
