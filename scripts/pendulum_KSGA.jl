using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using TimerOutputs

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# Parameters

T = 100 # horison
J = 30 # episodes

# discounting factor
γ = 0.90

# inverse kernel bandwidth, scale
actor_η = 0.15540143550398874
critic_η = 1.0

# critic sparsity
μ = 0.001

# standard deviation
σ = 0.2

# actor step size / learning rate
ζ = 0.05

# sample maximum
N = 10000

# dynamical system parameters
max_torque = 2 # -max_torque < action space < max_torque
max_speed = 8

#RL dims
state_dims = 3
action_dims = 1

# M sub-episodes
Z = 5

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

learn_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# ancillary

# initial critic weights
critic_Λ = []
critic_C = []

# initial actor weights
actor_Λ = [0.0 0.0]
actor_β = [0.0]
actor_C = zeros(state_dims, 2)

# scaling
θ_scale = 1/π
ω_scale = 1/max_speed
a_scale = 1/max_torque
reward_scale = 1/((π)^2 + 0.1*(max_speed^2) + 0.001*(max_torque^2))

# action space
#A = collect(-max_torque:2.0:max_torque)

for j in 1:J # episodes

    global x, μ, ζ, σ, N, Z
    global critic_Λ, critic_C, critic_η
    global actor_Λ, actor_C, actor_β, actor_η
    global episode_hook, inner_hook, sample_hook, learn_hook
    global max_torque, θ_scale, ω_scale, a_scale, reward_scale
    global state_dims, action_dims

    # initial condition
    if j < J - 2 #&& 1 == 2
        x = [rand(Uniform(-π, π)),
            rand(Uniform(-1, 1))]
    else
        x = [π, 0.0]
    end

    if (j-1)%Z == 0

        #= σ = 0.95*σ

        if σ < 0.01
            σ = 0.01
        end =#
    end

    Σ = (σ^2)*Matrix(I,1,1) # covariance matrix
    Σinv = inv(Σ) # for compatible kernel

    for k in 1:T # time steps

        global x, ζ, μ, σ
        global critic_Λ, critic_C, critic_η
        global actor_Λ, actor_C, actor_β, actor_η
        global episode_hook, inner_hook, sample_hook, learn_hook
        global max_torque, θ_scale, ω_scale, a_scale, reward_scale
        global state_dims, action_dims

        state = [cos(x[1]); sin(x[1]); ω_scale*x[2]]

        #-------------------------------------------------------------
        # Control

        # mean vector
        m = max_torque*function_approximation(state, actor_Λ, actor_C, actor_η, β = actor_β)

        m = clamp(m, -max_torque, max_torque)

        # Create the multivariate normal distribution
        π_ax = MvNormal([m], Σ)

        # Generate a random sample from the distribution
        a = clamp.(rand(π_ax), -max_torque, max_torque)

        #-------------------------------------------------------------
        # Manage data

        reward = -((x[1])^2 + 0.1*(x[2]^2) + 0.001*(a[1]^2))*reward_scale

        sample_hook = DataFrame(t = k,
                                y = state[1], x = state[2], ω = state[3],
                                a = a_scale*a,
                                m = a_scale*m,
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
    state_data = @views Matrix(transpose(Matrix(inner_hook[:, 2:state_dims + 1])))
    action_data = @views transpose(inner_hook[:, state_dims + 2])
    mean_data = @views transpose(inner_hook[:, state_dims + 3])
    rewards_data = @views inner_hook[:, state_dims + 4]

    data_points = length(rewards_data)

    #-------------------------------------------------------------
    # Episode Results

    total_reward = sum(rewards_data)/data_points
    episode_reward = sum(rewards_data[end-T+1:end])/T


    #-------------------------------------------------------------
    # learning

    if j%Z == 0

        valid_prev = deleteat!(collect(range(1, data_points - 1)), findall(diff(steps) .!= 1))

        if N >= length(valid_prev)

            data_ids = valid_prev

        else
            data_ids = rand(valid_prev, N)
        end

        #= data = @views [state_data; action_data]

        critic_C, _, _, ids = ALD(data, data_ids, μ, critic_η)
        critic_Λ = KLSTD(data, data_ids, critic_C, rewards_data, critic_η, γ) =#

        data = @views (state_data, action_data, mean_data)

        critic_C, _, _, ids = ALD(data, data_ids, μ, critic_η,
                                kernel = Compatible_kernel, Σinv = Σinv)
        critic_Λ = KLSTD(data, data_ids, critic_C, rewards_data, critic_η, γ,
                                kernel = Compatible_kernel, Σinv = Σinv)


        ∇J = SGA(data, data_ids, T, Z, critic_η, Σinv, critic_Λ, critic_C, critic_η)

        f_m = data[3][:, data_ids] .+ ζ*∇J

        f_m = clamp.(f_m, -1, 1)

        global data, data_ids, f_m, valid_prev

        actor_Λ, actor_C, actor_β, actor_error, recon = OMP(data[1][:,  data_ids], actor_η, f_m; N = 10000, ϵ = 1e-3)

        println("\ntotal rewards = ", round(total_reward, digits = 3))
        println("episode rewards = ", round(episode_reward, digits = 3))
        println("Critic Centres = ", size(critic_C[1], 2))
        println("Actor Centres = ", size(actor_C, 2))
        println("actor_error = ", round(actor_error, digits = 3))
        println("maximum ζ*∇J = ", round(ζ*maximum(abs.(∇J)), digits = 3))
        println("maximum ∇J = ", round(maximum(abs.(∇J)), digits = 3))



    end

    println("j = ", j)
    # saving

    if j == J && j%Z == 0
        global train_mean = vec(f_m)
        global reconstruct = vec(recon)
    end

end

#-------------------------------------------------------------------------------------------
# Plots

plot_df = episode_hook[1:end, :]

traces = [scatter(plot_df, x = :t, y = :x, name = "x"),
            scatter(plot_df, x = :t, y = :y, name = "y"),
            scatter(plot_df, x = :t, y = :ω, name = "ω"),
            scatter(plot_df, x = :t, y = :θ, name = "θ"),
            scatter(plot_df, x = :t, y = :m, name = "mean"),
            #scatter(plot_df, x = :t, y = :a, name = "actions"),
            scatter(plot_df, x = :t, y = :r, name = "reward"),]

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

datsa = (data[1][1,  data_ids] .- mean(data[1][1,  data_ids]))./std(data[1][1,  data_ids])
datsb = (data[1][2,  data_ids] .- mean(data[1][2,  data_ids]))./std(data[1][2,  data_ids])
datsc = (data[1][3,  data_ids] .- mean(data[1][3,  data_ids]))./std(data[1][3,  data_ids])

f_t = (f_m .- mean(f_m))./std(f_m)

dats = Matrix(transpose(hcat(datsa, datsb, datsc)))

η = 0.15540143550398874
ϵ = 1.4424310308386514e-6

actor_Λ, actor_C, actor_β, actor_error, recon, Press1a = OMP(dats, η, f_t; N = 1000000, ϵ = ϵ, PRESS = true)
Λ1b, β1b, ϵ1b, η1b, Press1b, verge = CGD(dats, f_t; ϵ = ϵ, η = η, nmax = 10)

train_mean = vec(f_t)
reconstruct = vec(recon)

traces = [scatter(x = collect(1:1:length(train_mean)), y = train_mean, name = "train_mean"),
        scatter(x = collect(1:1:length(train_mean)), y = reconstruct, name = "reconstruct")]

plot_OMP = plot(traces,
        Layout(
            title = attr(
                text = "OMP",
                ),
            title_x = 0.5,
            xaxis_title = "steps [k]",
            ),
        )

display(plot_OMP)

show(to)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
