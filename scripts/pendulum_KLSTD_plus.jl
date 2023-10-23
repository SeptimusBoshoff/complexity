using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Zygote

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------------------
# Parameters

T = 100 # horison
J = 20 # episodes

# discounting factor
γ = 0.90

# inverse kernel bandwidth, scale
critic_η = 1.0
actor_η = 1.0
actor_ϵ = 1e-6

# sparsity
μ = 0.001

# sample maximum
N = 5000

# dynamical system parameters
max_torque = 2
max_speed = 8

#RL dims
state_dims = 3
action_dims = 1

# action noise
σ = 1.0

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# ancillary

# initial critic weights
critic_Λ = []
critic_C = []

# initial critic weights
acton_Λ = []
actor_C = []
actor_β = []

# scaling
θ_scale = 1/π
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = 1/((π)^2 + 0.1*(max_speed^2) + 0.001*(max_torque^2))

# gradient descent
optf = OptimizationFunction(Q_max, Optimization.AutoZygote())

for j in 1:J # episodes

    global x, η, A, σ
    global critic_Λ, critic_C, critic_η
    global actor_Λ, actor_C, actor_η, actor_β, actor_ϵ
    global episode_hook, inner_hook, sample_hook, learn_hook
    global max_torque, θ_scale, ω_scale, a_scale, reward_scale
    global state_dims, action_dims

    # initial condition
    if j < J - 2 && J + 2 > 0
        x = [rand(collect(-π:π/20:π)),
            rand(collect(-max_speed:max_speed/20:max_speed))]
    else
        x = [π, 0.0]
    end

    σ = 0.95*σ

    for k in 1:T # time steps

        global x, η, A, N, σ
        global critic_Λ, critic_C, critic_η
        global actor_Λ, actor_C, actor_η, actor_β
        global episode_hook, inner_hook, sample_hook, learn_hook
        global max_torque, θ_scale, ω_scale, a_scale, reward_scale
        global state_dims, action_dims

        state = [cos(x[1]); sin(x[1]); ω_scale*x[2]]

        #-------------------------------------------------------------
        # Control

        if j == 1

            a = rand(Uniform(-max_torque, max_torque))
            ac = a

        else

            a0 = function_approximation(state, actor_Λ, actor_C, actor_η, β = actor_β)
            a0 = clamp(a0, -0.99, 0.99)

            critic_params = (state, critic_Λ, critic_C, critic_η)

            prob = OptimizationProblem(optf, [a0], critic_params, lb = [-1.0], ub = [1.0])
            sol = solve(prob, BFGS())

            ac = sol.u[1]

            a = rand(Normal(ac*max_torque, σ*max_torque))

            a = clamp(a, -max_torque, max_torque)

        end

        #-------------------------------------------------------------
        # Manage data

        reward = -((x[1])^2 + 0.1*(x[2]^2) + 0.001*(a[1]^2))*reward_scale

        sample_hook = DataFrame(t = k,
                                y = state[1], x = state[2], ω = state[3],
                                a = a_scale*a,
                                r = reward,
                                θ = θ_scale*x[1],
                                ac = ac)

        append!(inner_hook, sample_hook)

        #-------------------------------------------------------------
        # evolve

        x = pendulum(x, a, max_torque = max_torque, max_speed = max_speed, dt = 0.05)

        if x[1] > π
            x[1] = x[1] - 2π
        elseif x[1] < -π
            x[1] = x[1] + 2π
        end

    end

    #-------------------------------------------------------------
    # Manage data

    if j == J
        append!(episode_hook, inner_hook[end-T+1:end, :])
    end

    steps = @views (inner_hook[:, 1])
    state_data = @views transpose(Matrix(inner_hook[:, 2:state_dims + 1]))
    action_data = @views transpose(inner_hook[:, state_dims + 2])
    rewards_data = @views inner_hook[:, state_dims + 3]
    action_data2 = @views transpose(inner_hook[:, end])

    data = @views [state_data; action_data]

    data_points = length(rewards_data)

    #-------------------------------------------------------------
    # learning

    valid_prev = deleteat!(collect(range(1, data_points - 1)), findall(diff(steps) .!= 1))

    if N >= length(valid_prev)

        data_ids = valid_prev

    else
        data_ids = rand(valid_prev, N)
    end

    # Critic - Q value

    critic_C, _, _, _ = ALD(data, data_ids, μ, critic_η)
    critic_Λ = KLSTD(data, data_ids, critic_C, rewards_data, critic_η, γ)

    # Actor - policy

    actor_Λ, actor_C, actor_β, actor_err, actor_P = OMP(state_data[:, data_ids], actor_η, action_data2[:, data_ids], N = 200, PRESS = true)

    #= actor_Λ, actor_β, actor_ϵ, actor_η, actor_P, verge = CGD(state_data[:, data_points-T+1:data_points], action_data2[:, data_points-T+1:data_points]; ϵ = actor_ϵ, η = actor_η) # TODO: action data is noisy - train on not noisy?

    actor_C = state_data[:, data_points-T+1:data_points] =#

    actor_Λ, actor_β, actor_ϵ, actor_η, actor_P, verge = CGD(state_data[:, data_ids], action_data2[:, data_ids]; ϵ = actor_ϵ, η = actor_η) # TODO: action data is noisy - train on not noisy?

    actor_C = state_data[:, data_ids]

    #-------------------------------------------------------------
    # Episode Results

    total_reward = sum(rewards_data)/size(state_data, 2)
    episode_reward = sum(rewards_data[end-T+1:end])/T

    println("\ntotal rewards = ", round(total_reward, digits = 3))
    println("episode rewards = ", round(episode_reward, digits = 3))
    println("Critic Centres = ", size(critic_C, 2))
    #println("Actor residual error = ", round(actor_err, digits = 3))
    println("Actor PRESS = ", round(actor_P, digits = 3))
    #println("verge = ", verge)
    println("j = ", j)

end

#-------------------------------------------------------------------------------------------
# Plots

plot_df = episode_hook[1:end, :]

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

display(plot_episode)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
