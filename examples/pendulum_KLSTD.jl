using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------------------
# Parameters

T = 100 # horison
J = 100 # episodes

# discounting factor
γ = 0.90

# inverse kernel bandwidth, scale
η = 1.0

# sparsity
μ = 0.001

# greedy policy randomness
ε_min = 0.2

# sample maximum
N = 10000

# dynamical system parameters
max_torque = 2
max_speed = 8

#RL dims
state_dims = 3
action_dims = 1

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# ancillary

# initial critic weights
#= critic_Λ = []
critic_C = [] =#

# scaling
θ_scale = 1/π
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = 1/((π)^2 + 0.1*(max_speed^2) + 0.001*(max_torque^2))

# action space
A = collect(-max_torque:2.0:max_torque)

# state-action value
Q = Array{Float64,1}(undef, length(A))

for j in 1:J # episodes

    global x, η, μ, ε_min, A
    global critic_Λ, critic_C, Q
    global episode_hook, inner_hook, sample_hook, learn_hook
    global max_torque, θ_scale, ω_scale, a_scale, reward_scale
    global state_dims, action_dims

    # initial condition
    if j < J - 2 && J + 2 > 0 && j != 1
        x = [rand(Uniform(-π, π)),
            rand(Uniform(-max_speed, max_speed))]
    else
        x = [π, 0.0]
    end

    ε = 1/(j^0.5)
    if ε < ε_min
        ε = ε_min
    end

    for k in 1:T # time steps

        global x, η, μ, A, N
        global critic_Λ, critic_C, Q
        global episode_hook, inner_hook, sample_hook, learn_hook
        global max_torque, θ_scale, ω_scale, a_scale, reward_scale
        global state_dims, action_dims

        state = [cos(x[1]); sin(x[1]); ω_scale*x[2]]

        #-------------------------------------------------------------
        # Control

        if (rand() < ε && j < J) || j == -1

            a = rand(A)

        else

            for a in eachindex(A)

                action = a_scale*A[a]

                Q[a] = function_approximation([state; action], critic_Λ, critic_C, η)

            end

            a = A[argmax(Q)]

        end

        #-------------------------------------------------------------
        # Manage data

        reward = -((x[1])^2 + 0.1*(x[2]^2) + 0.001*(a[1]^2))*reward_scale

        sample_hook = DataFrame(t = k,
                                y = state[1], x = state[2], ω = state[3],
                                a = a_scale*a,
                                r = reward,
                                θ = θ_scale*x[1])

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

    critic_C, _, _, _ = ALD(data, data_ids, μ, η)
    critic_Λ = KLSTD(data, data_ids, critic_C, rewards_data, η, γ)

    #-------------------------------------------------------------
    # Episode Results

    total_reward = sum(rewards_data)/size(state_data, 2)
    episode_reward = sum(rewards_data[end-T+1:end])/T

    println("\ntotal rewards = ", round(total_reward, digits = 3))
    println("episode rewards = ", round(episode_reward, digits = 3))
    println("Centres = ", size(critic_C, 2))
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
