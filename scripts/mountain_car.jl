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
J = 1 # episodes

γ = 0.99 # discounting factor

# initial condition ranges
x_range = [-1.2, 0.5]
v_range = [-0.07, 0.07]

# actor weights
actor_α = [0]
actor_C = zeros(2)

# kernel bandwidth, scale
ζ = 0.5

# actor step size
η = 0.01

#-------------------------------------------------------------------------------------------
# Hook

hook = DataFrame()

#-------------------------------------------------------------------------------------------
# Learning

f = zeros(T)

#for j in 1:J # episodes

    global x, hook, actor_α, actor_C, η

    # initial conditions

    x = [rand(Uniform(x_range[1], x_range[2])),
        rand(Uniform(v_range[1], v_range[2]))]

    x = [-0.5, 0]

    # policy parameters

    σ = 1.0 # standard deviation

    Σ = (σ^2)*Matrix(I,1,1) # covariance matrix
    Σinv = inv(Σ) # for compatible kernel

    for k in 1:T # time steps

        global x, hook, actor_α, actor_C

        # mean vector
        μ = function_approximation(x, actor_α, actor_C, ζ)

        # Create the multivariate normal distribution
        π_ax = MvNormal([μ], Σ)

        # Generate a random sample from the distribution
        a = clamp.(rand(π_ax), -1, 1)

        reward = exp(-8(x[1] - 0.6)^2)

        sample = DataFrame(t = k, x = x[1], v = x[2], a = a, μ = μ, r = reward)
        append!(hook, sample)

        if x[1] >= 0.5
            break
        end

        # evolve
        #a = 1.0
        x = Mountain_Car(x, a)

    end

    #σ = 0.01 # standard deviation
    #Σ = (σ^2)*Matrix(I,2,2) # covariance matrix
    #Σinv = inv(Σ) # for compatible kernel

    state_data = @views transpose(Matrix(hook[:,2:3]))
    action_data = @views transpose(hook[:,4])
    mean_data = @views transpose(hook[:,5])
    rewards_data = @views hook[:,6]

    data = @views (state_data, action_data, mean_data)

    α, Q, D, b = KLSTD(data, rewards_data, ζ, Σinv, γ)

    display(Q)

    f = SGA(data, ζ, Σinv, η, Q; kernel = Gaussian_kernel)

    #= G1  = Gramian(data[1], ζ)
    G2  = Gramian(data, ζ; kernel = Compatible_kernel, Σinv = Σinv)

    z₁ = (data[1][:,1], data[2][:,1], data[3][:,1])
    z₂ = (data[1][:,5], data[2][:,5], data[3][:,5])





    k1 = Gaussian_kernel(z₁[1], z₂[1], ζ, dims = 2)

    k = Compatible_kernel(z₁, z₂, ζ; kernel = Gaussian_kernel, Σinv = Σinv) =#

#end

U  = Gramian(f, ζ)

T = size(U,1)

#U = Array{Float64,2}(undef, T, T)

for i in 1:T

    U[i,:] = U[i,:]/sum(abs2, U[i,:])

end

R = transpose(f)

d = argmax(diag(transpose(U)*transpose(R)*R*U))



#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(hook, x = :t, y = :x, name = "position"),
            scatter(hook, x = :t, y = :v, name = "velocity"),
            scatter(hook, x = :t, y = :r, name = "reward"),
            scatter(hook, x = :t, y = :a, name = "actions")]

plot_SS_t = plot(traces,
                Layout(
                    title = attr(
                        text = "State Space: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "x, y [m]",
                    ),
                )

display(plot_SS_t)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
