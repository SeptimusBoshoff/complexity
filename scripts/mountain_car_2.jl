using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using LinearAlgebra

#= Details of the Mountain-Car Task

    The mountain-car taks has two continous state variables, the position of the car x(k),
    and the velocity of the car, v(k). At the start of each episode, the initial state is
    chosen randomly, uniformly from th allowed ranges: -1.2 <= x <= 0.5, -0.7 <= v <= 0.7.
    The mountain car geography is described by altitude(k) = sin(3x(k)). That action, a(k),
    takes values -1 <= a <= 1. The state evolution is according to the following simplified
    physics:

    v(k+1) = bound[v(k) + 0.001*a(k) - g*cos(3*x(k))]

    and

    x(k+1) = bound[x(k) + v(k+1)]

    where g = -0.0025 is the force of gravity and the bound operation clips each variable
    within its allowed range. If x(k+1) is clipped in this way, then v(k+1) is also reset to
    zero. The episode terminates with the first position value that exceeds x(k+1) > 0.5.

=#

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------------------
# Parameters

T = 100 # horison
J = 1000 # episodes

γ = 0.99 # discounting factor

# kernel bandwidth, scale
ζ = 1.0

# actor step size
η = 0.001

σ = 0.1 # standard deviation

M = 10

# initial condition ranges
x_range = [-1.2, 0.5]
v_range = [-0.07, 0.07]

xtrain = zeros(2, 10*T)

for i in 1:10*T
    xtrain[:,i] = [rand(Uniform(x_range[1], x_range[2])), rand(Uniform(v_range[1], v_range[2]))/0.07]
end

# initial actor weights
actor_Λ = [0]
actor_C = zeros(2)

#-------------------------------------------------------------------------------------------
# dataframes

sample_hook = DataFrame()
inner_hook = DataFrame()
episode_hook = DataFrame()

learn_hook = DataFrame()

#-------------------------------------------------------------------------------------------
# Learning

∇J = zeros(1,10*T)

m = 1
j = 1
for j in 1:J # episodes

    global x, actor_Λ, actor_C, η, ζ, γ, σ, ∇J, m, M
    global episode_hook, inner_hook, sample_hook, learn_hook

    # initial conditions

    x = [rand(Uniform(x_range[1], x_range[2])), 0]

    #x = [-0.5, 0]

    # policy parameters

    if j < J/2

        σ = σ*0.99 # standard deviation

        if σ < 0.01 σ = 0.01 end

    end

    Σ = (σ^2)*Matrix(I,1,1) # covariance matrix
    Σinv = inv(Σ) # for compatible kernel

    for k in 1:T # time steps

        global x, actor_Λ, actor_C, η, ζ, γ, σ, ∇J, m, M
        global episode_hook, inner_hook, sample_hook, learn_hook

        # mean vector
        μ = function_approximation([x[1]; x[2]/0.07], actor_Λ, actor_C, ζ)

        # Create the multivariate normal distribution
        π_ax = MvNormal([μ], Σ)

        # Generate a random sample from the distribution
        a = clamp.(rand(π_ax), -1, 1)

        reward = exp(-8*(x[1] - 0.6)^2)

        sample_hook = DataFrame(t = k, x = x[1], v = x[2]/0.07, a = a, μ = μ, r = reward)
        append!(inner_hook, sample_hook)

        if x[1] >= 0.5
            break
        end

        # evolve
        x = Mountain_Car(x, a)

        if x[1] >= 0.5
            #println("Success")
            rewards_data = @views inner_hook[:,6]
            #println("rewards = ", round(1000*sum(rewards_data), digits = 3))
            #println("position = ", round(x[1], digits = 3))
            #println("velocity = ", round(x[2], digits = 3))
        end

    end

    state_data = @views transpose(Matrix(inner_hook[:,2:3]))
    action_data = @views transpose(inner_hook[:,4])
    mean_data = @views transpose(inner_hook[:,5])
    rewards_data = @views inner_hook[:,6]

    data = @views (state_data, action_data, mean_data)

    Q = Q_MC(γ, rewards_data)

    ∇J += SGA(data, ζ, Σinv, Q, xtrain)

    if m >= M

        f_μ = zeros(10*T)

        for i in 1:10*T
            f_μ = function_approximation(xtrain[:,i], actor_Λ, actor_C, ζ)
        end

        f_μ = f_μ .+ η*(1/M)*∇J

        f_μ = clamp.(f_μ, -1, 1)

        actor_Λ, actor_C, err = OMP(xtrain, ζ, f_μ; N = 100)

        println("rewards = ", round(1000*sum(rewards_data), digits = 3))
        println("error = ", round(err, digits = 3))
        println("mean change = ", round(mean( η*(1/M)*∇J), digits = 3))
        println("j = ", j)

        ∇J = zeros(1,10*T)
        m = 1

    else

        m += 1
    end

    if j == 1

        actor_Λ, actor_C, err = OMP(data[1], ζ, data[3]; N = 15)
    end

    #println("rewards = ", round(1000*sum(rewards_data), digits = 3))
    #println("max Q = ", maximum(Q))
    #println("min Q = ", minimum(Q))
    #println("j = ", j)

    empty!(learn_hook)
    learn_hook = DataFrame(Q = vec(Q))

    if isnan(Q[1])

        display(f_μ)
        display(Gramian(data[1], ζ))
        display(Σinv)

        #break
    else
        empty!(episode_hook)
        append!(episode_hook, inner_hook)
    end
    empty!(inner_hook)

end

#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(episode_hook, x = :t, y = :x, name = "position"),
            scatter(episode_hook, x = :t, y = :v, name = "velocity"),
            scatter(episode_hook, x = :t, y = :r, name = "reward"),
            scatter(episode_hook, x = :t, y = :a, name = "actions"),
            scatter(episode_hook, x = :t, y = :μ, name = "mean"),
            scatter(learn_hook, x = :t, y = :Q, name = "Q"),]

plot_episode = plot(traces,
                Layout(
                    title = attr(
                        text = "Episodic State",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "x [m], y [m/s]",
                    ),
                )

display(plot_episode)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
