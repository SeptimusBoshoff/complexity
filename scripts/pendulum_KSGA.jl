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

T = 150 # horison
J = 2 # episodes

# discounting factor
γ = 0.9

# kernel bandwidth, scale
ζ = 1.0

# sparsity
μ = 0.001

# standard deviation
σ = 0.05

# actor step size
η = 0.0001

# scaling
max_torque = 2
max_speed = 8

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

learn_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# Learning

#RL dims
state_dims = 2
action_dims = 1

# initial actor weights
actor_Λ = [0]
actor_C = zeros(state_dims)

# initial critic weights
critic_Λ = []
critic_C = []

θ_scale = 1/π
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = -1/((π)^2 + 0.1*(max_speed^2) + 0.001*(max_torque^2))

for j in 1:J # episodes

    global x, ζ, μ, σ, η
    global critic_Λ, critic_C, actor_Λ, actor_C
    global episode_hook, inner_hook, sample_hook, learn_hook
    global max_torque, θ_scale,  ω_scale, a_scale, reward_scale
    global state_dims, action_dims

    # initial condition
    x = [π, 0.0]

    #σ = 0.95*σ
    if σ < 0.01
        σ = 0.01
    end

    Σ = (σ^2)*Matrix(I,1,1) # covariance matrix
    Σinv = inv(Σ) # for compatible kernel

    actor_error = 0

    for k in 1:T # time steps

        global x, ζ, μ, σ, η
        global critic_Λ, critic_C, actor_Λ, actor_C
        global episode_hook, inner_hook, sample_hook, learn_hook
        global max_torque, θ_scale,  ω_scale, a_scale, reward_scale
        global state_dims, action_dims

        #state = [cos(x[1]); sin(x[1]); ω_scale*x[2]]
        state = [θ_scale*x[1]; ω_scale*x[2]]

        # mean vector
        μ = function_approximation(state, actor_Λ, actor_C, ζ)

        # Create the multivariate normal distribution
        π_ax = MvNormal([μ], Σ)

        # Generate a random sample from the distribution
        a = clamp.(rand(π_ax), -max_torque, max_torque)

        reward = reward_scale*((x[1])^2 + 0.1*(x[2]^2) + 0.001*(a[1]^2))

        sample_hook = DataFrame(t = k,
                                θ = θ_scale*x[1], ω = ω_scale*x[2],
                                a = a_scale*a[1],
                                μ = a_scale*μ,
                                r = reward,
                                y = cos(x[1]), x = sin(x[1]))

        append!(inner_hook, sample_hook)

        # evolve
        x = pendulum(x, a, max_torque = max_torque, max_speed = max_speed)

        if x[1] > π
            x[1] = x[1] - 2π
        elseif x[1] < -π
            x[1] = x[1] + 2π
        end

        state_data = @views transpose(Matrix(inner_hook[:, 2:state_dims + 1]))
        action_data = @views transpose(inner_hook[:, state_dims + 2])
        mean_data = @views transpose(inner_hook[:, state_dims + 3])
        rewards_data = @views inner_hook[:, state_dims + 4]

        # learn
        data = @views (state_data, action_data, mean_data)

        critic_C, _, _ = ALD(data, μ, ζ, kernel = Compatible_kernel, Σinv = Σinv)
        critic_Λ = KLSTD(data, critic_C, rewards_data, ζ, γ, kernel = Compatible_kernel, Σinv = Σinv)

        f_μ = SGA(data, ζ, Σinv, η, critic_Λ, critic_C)

        f_μ = clamp.(f_μ, -max_torque, max_torque)

        actor_Λ, actor_C, actor_error = OMP(data[1], ζ, f_μ; N = 100)

    end

    state_data = @views transpose(Matrix(inner_hook[:, 2:state_dims + 1]))
    action_data = @views transpose(inner_hook[:, state_dims + 2])
    mean_data = @views transpose(inner_hook[:, state_dims + 3])
    rewards_data = @views inner_hook[:, state_dims + 4]

    total_reward = sum(rewards_data)/size(state_data,2)

    println("\nrewards = ", round(total_reward, digits = 3))

    # learn

    #= data = @views (state_data, action_data, mean_data)

    critic_C, _, _ = ALD(data, μ, ζ, kernel = Compatible_kernel, Σinv = Σinv)
    critic_Λ = KLSTD(data, critic_C, rewards_data, ζ, γ, kernel = Compatible_kernel, Σinv = Σinv)

    f_μ = SGA(data, ζ, Σinv, η, critic_Λ, critic_C)

    f_μ = clamp.(f_μ, -max_torque, max_torque)

    actor_Λ, actor_C, actor_error = OMP(data[1], ζ, f_μ; N = 100) =#


    println("Critic Centres = ", size(critic_C[1], 2))
    println("Actor Centres = ", size(actor_C, 2))
    println("Actor Error = ", actor_error)
    println("j = ", j)

    empty!(episode_hook)
    append!(episode_hook, inner_hook)
    empty!(inner_hook)

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
