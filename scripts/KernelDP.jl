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

N = 120 # iterations
Mx = 5000 # state space samples - support vectors
Ma = 3 # action space samples - support vectors

# discounting factor
γ = 0.95

# dynamical system parameters
max_torque = 2.0
max_speed = 8.0

# incomplete cholesky precision parameter
nu = 0.0005

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

    X_state = rand(Uniform(-1, 1), 2, Mx)
    X_state[1,:] = π*X_state[1,:]
    X_state[2,:] = max_speed*X_state[2,:]

    X_state[1,1] = π
    X_state[2,1] = 0.0

    X_train = Array{Float64,2}(undef, 3, Mx)
    X_train[1,:] = cos.(X_state[1,:])
    X_train[2,:] = sin.(X_state[1,:])
    X_train[3,:] = ω_scale*(X_state[2,:])

    Ac = range(-1, stop = 1, length = Ma) # centres
    A = rand(Ac, Mx)

    X_state_nxt = Array{Float64,2}(undef, 2, Mx)
    X_train_nxt = Array{Float64,2}(undef, 3, Mx)

    for i in 1:Mx

        X_state_nxt[:, i] = pendulum(X_state[:, i], max_torque*A[i], max_torque = max_torque, max_speed = max_speed)
        X_train_nxt[:, i] = [cos(X_state_nxt[1, i]); sin(X_state_nxt[1, i]); ω_scale*X_state_nxt[2, i]]
    end

    X_A = vcat(X_train, transpose(A))

    average_distance = Mean_Distance(X_A) / 5
    η_V = 1/(2*average_distance^2) # inverse kernel bandwidth, scale

    Gxa = Gramian(X_A, η_V)

    R, ids, error = ichol(Gxa, nu, alg = :pgso)
    #ids = 1:Mx#vcat(collect(2000:Mx), collect(1:1999))

    Gxa = Gxa[ids, ids] # is this necessary?
    Mx_ch = length(ids)

    println("Centres = ", Mx_ch)
    println("PGSO reconstruction error = ", error)

    # conditional embedding weights
    ϵ = 1e-6
    W = Matrix{Float64}(I, Mx_ch, Mx_ch)
    ldiv!(cholesky(Gxa + ϵ*Mx_ch*I), W)

    #---------
    # initial values

    Vk = zeros(Mx_ch)
    Vk_1 = zeros(Mx_ch)

    ϑ = Array{Float64,1}(undef, Mx_ch) # feature vector
    ϑ2 = Array{Float64,1}(undef, Mx_ch) # feature vector

    πk = Array{Float64, 1}(undef, Mx_ch)
    πk = fill!(πk, 0.)

    Qx_i = Array{Float64,1}(undef, Ma)

    ids_full = ids
    Mx_l = length(ids_full)
    V = Array{Float64, 1}(undef, Mx_l)
    π_0 = Array{Float64, 1}(undef, Mx_l)

end

#-------------------------------------------------------------------------------------------
# Training

for k in N-1:-1:0

    global Vk, Vk_1, η_V
    global Qx_i
    global ϑ

    for (i, pt) in enumerate(ids)

        for j in 1:Ma

            Xnxt_a = vcat(X_train_nxt[:,pt], Ac[j])

            for (k, cen) in enumerate(ids)

                ϑ[k] = Gaussian_kernel(X_A[:,cen], Xnxt_a, η_V)
            end

            α_i = W*ϑ

            V_next = transpose(α_i)*Vk

            reward = -(X_state_nxt[1,pt]^2 + 0.01*(X_state_nxt[2,pt]^2) + 0.001*((max_torque*A[j])^2))*reward_scale

            Qx_i[j] = reward + γ*V_next
        end

        Vk_1[i], jdx = findmax(Qx_i)
        πk[i] = Ac[jdx]

    end

    Vk_change = norm(Vk .- Vk_1)/Mx_ch
    Vk = copy(Vk_1)

    println("\nk = ", k)
    println("Vk_change = ", round(Vk_change*1e3, digits = 3),"e-3")

end

for (i, pt) in enumerate(ids_full)

    global V, Vk, η_V
    global Qx_i
    global ϑ

    for j in 1:Ma

        X_a = vcat(X_train[:,pt], Ac[j])

        for (k, cen) in enumerate(ids)

            ϑ[k] = Gaussian_kernel(X_A[:,cen], X_a, η_V)
        end

        α_i = W*ϑ

        V_next = transpose(α_i)*Vk

        reward = -(X_state[1,pt]^2 + 0.01*(X_state[2,pt]^2) + 0.001*((max_torque*A[j])^2))*reward_scale

        Qx_i[j] = reward + γ*V_next
    end

    V[i], jdx = findmax(Qx_i)
    π_0[i] = Ac[jdx]
end

πp_idx = findall(x -> x > 0, π_0)
πn_idx = setdiff(1:Mx_l, πp_idx)

π_switch = zeros(Mx_l)
π_switch[πp_idx] .= 1

η_class = 50
μ_weights, σ²_weights, C, verge = GP(X_train[:,ids_full], π_switch, η_class; ϵ = 1e-8, max_itr = 50, tol = 1e-9);

average_distance = Mean_Distance(X_train[:,ids_full]) / 4
η_π = 1/(2*average_distance^2)

Λ_πp, C_πp, β_πp, σ2_πp = RVM(X_train[:,ids_full[πp_idx]], π_0[πp_idx], η_π, noise_itr = 2, σ2 = 0.1)
Λ_πn, C_πn, β_πn, σ2_πn = RVM(X_train[:,ids_full[πn_idx]], π_0[πn_idx], η_π, noise_itr = 2, σ2 = 0.1)

#-------------------------------------------------------------------------------------------
# Validation

inner_hook = DataFrame()
T = 200

x_state = [π, 0.] # initial condition

for k in 1:T

    global x_state, inner_hook, Qx_i, ϑ

    x_train = [cos(x_state[1]); sin(x_state[1]); ω_scale*x_state[2]]

    # -----
    # Option 1: Two policy

    #= p = GP_predict(x_train, μ_weights, σ²_weights, C, η_class)

    if p > 0.5

        action = function_approximation(x_train, Λ_πp, C_πp, η_π, β = β_πp)

    else

        action = function_approximation(x_train, Λ_πn, C_πn, η_π, β = β_πn)
    end =#

    # -----
    # Option 2: Direct

    for j in 1:Ma

        X_a = vcat(x_train, Ac[j])

        for (i, cen) in enumerate(ids)

            ϑ[i] = Gaussian_kernel(X_A[:,cen], X_a, η_V)
        end

        α_i = W*ϑ

        V_next = transpose(α_i)*Vk

        reward = -(x_state[1]^2 + 0.01*(x_state[2]^2) + 0.001*((max_torque*A[j])^2))*reward_scale

        Qx_i[j] = reward + γ*V_next
    end

    _, jdx = findmax(Qx_i)
    action = Ac[jdx]

    # -----

    action = clamp(action*max_torque, -max_torque, max_torque)

    Reward = -((x_state[1])^2 + 0.01*(x_state[2]^2) + 0.001*(action^2))*reward_scale

    #-------------------------------------------------------------
    # Manage data

    sample_hook = DataFrame(t = k,
                            y = x_train[1], x = x_train[2], ω = x_train[3],
                            a = a_scale*action,
                            r = Reward,
                            θ = θ_scale*x_state[1])

    append!(inner_hook, sample_hook)

    #-------------------------------------------------------------
    # evolve

    x_state = pendulum(x_state, action, max_torque = max_torque, max_speed = max_speed)

    x_state = x_state .+ 0.0*randn(2)

end

mat_size = 100
π_mat = Array{Float64,2}(undef, mat_size, mat_size)
P_mat = Array{Float64,2}(undef, mat_size, mat_size)

θ_range = range(-π, stop = π, length = mat_size)
ω_range = range(-max_speed, stop = max_speed, length = mat_size)

for ω in eachindex(ω_range)

    for θ in eachindex(θ_range)

        x = [cos(θ_range[θ]); sin(θ_range[θ]); ω_scale*ω_range[ω]]

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
#--

traces_V = scatter3d(x = vec(X_state[1,ids_full]), y = vec(X_state[2,ids_full]), z = V, mode="markers", marker = attr(
    color = V,
    colorscale = "Cividis",
    size = 5),
    name = "Value-function")

traces = [traces_V]
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

trace_π = heatmap(x = θ_range, y = ω_range, z = π_mat, colorscale = "Electric")
layout = Layout(title = "policy function", xaxis_title = "θ [rad]", yaxis_title = "ω [rad/s]")
plot_heatmap_π = Plot(trace_π, layout)
#---

trace_P = heatmap(x = θ_range, y = ω_range, z = P_mat, colorscale = "Electric")
layout = Layout(title = "Classification", xaxis_title = "θ [rad]", yaxis_title = "ω [rad/s]")
plot_heatmap_P = Plot(trace_P, layout)
#---

#-------------------------------------------------------------------------------------------
# Display

display(plot_V)
display(plot_heatmap_π)
display(plot_heatmap_P)
display(plot_episode)

show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
