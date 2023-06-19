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

T = 200 # horison
J = 50 # episodes

# discounting factor
γ = 0.90

# kernel bandwidth, scale
ζ = 1.0

# sparsity
μ = 0.001

# dynamical system parameters
max_torque = 2
max_speed = 8

# action space
A = collect(-max_torque:2.0:max_torque)

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

learn_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# Learning

# initial critic weights
critic_Λ = []
critic_C = []

# scaling
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = 1/((π)^2 + 0.1*(max_speed^2) + 0.001*(max_torque^2))

#RL dims
state_dims = 3
action_dims = 1

for j in 1:J # episodes

    global x, ζ, μ, A
    global critic_Λ, critic_C
    global episode_hook, inner_hook, sample_hook, learn_hook
    global max_torque, ω_scale, a_scale, reward_scale
    global state_dims, action_dims

    # initial condition
    if j < J
        x = [rand(Uniform(-π, π)),
            rand(Uniform(-1, 1))]
    else
        x = [π, 0.0]
    end

    for k in 1:T # time steps

        global x, ζ, μ, A
        global critic_Λ, critic_C
        global episode_hook, inner_hook, sample_hook, learn_hook
        global max_torque, ω_scale, a_scale, reward_scale
        global state_dims, action_dims

        state = [cos(x[1]); sin(x[1]); ω_scale*x[2]]

        if j == 1

            a = rand(A)

        else

            Q = Array{Float64,1}(undef, length(A))

            for a in eachindex(A)

                action = a_scale*A[a]

                Q[a] = function_approximation([state; action], critic_Λ, critic_C, ζ)

            end

            a = A[argmax(Q)]

            if rand() < 0.1 && j < J
                a = rand(A)
            end

        end

        reward = -((x[1])^2 + 0.1*(x[2]^2) + 0.001*(a[1]^2))*reward_scale

        sample_hook = DataFrame(t = k,
                                y = state[1], x = state[2], ω = state[3],
                                a = a_scale*a,
                                r = reward,
                                θ = x[1]/π)

        append!(inner_hook, sample_hook)

        state_data = @views transpose(Matrix(inner_hook[:, 2:state_dims + 1]))
        action_data = @views transpose(inner_hook[:, state_dims + 2])
        rewards_data = @views inner_hook[:, state_dims + 3]

        # evolve
        x = pendulum(x, a, max_torque = max_torque, max_speed = max_speed, dt = 0.05)

        if x[1] > π
            x[1] = x[1] - 2π
        elseif x[1] < -π
            x[1] = x[1] + 2π
        end

    end

    state_data = @views transpose(Matrix(inner_hook[:, 2:state_dims + 1]))
    action_data = @views transpose(inner_hook[:, state_dims + 2])
    rewards_data = @views inner_hook[:, state_dims + 3]

    data = @views [state_data; action_data]

    critic_C, _, _ = ALD(data, μ, ζ)
    critic_Λ = KLSTD(data, critic_C, rewards_data, ζ, γ, j)

    total_reward = sum(rewards_data)/size(state_data, 2)

    println("rewards = ", round(total_reward, digits = 3))
    println("Centres = ", size(critic_C, 2))
    println("j = ", j)

    empty!(episode_hook)
    append!(episode_hook, inner_hook[end-T+1:end, :])

end

#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(episode_hook, x = :t, y = :x, name = "x"),
            scatter(episode_hook, x = :t, y = :y, name = "y"),
            scatter(episode_hook, x = :t, y = :ω, name = "ω"),
            scatter(episode_hook, x = :t, y = :θ, name = "θ"),
            scatter(episode_hook, x = :t, y = :r, name = "reward"),
            scatter(episode_hook, x = :t, y = :a, name = "actions"),]

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
