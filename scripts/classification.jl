using Complexity
using Distributions
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using Random

println("\n...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........\n")

to = TimerOutput()

T = 1000 # samples

width = 1.0 # Gaussian kernel width - standard deviation
eps = 1 # episodes

η = 1/(2*width^2)

# ******************************************************************************************
# Training

Σ = 0.05*Matrix{Float64}(I, 2, 2) # covariance matrix

dist_1a = MvNormal([-0.4; -0.2], Σ)
dist_1b = MvNormal([0.4; 0.2], Σ)
dist_2a = MvNormal([-0.4; 0.9], Σ)
dist_2b = MvNormal([0.4; -0.9], Σ)

X_1 = rand(dist_1a, Int(round(T/4)))
X_1 = hcat(X_1, rand(dist_1b, Int(round(T/4))))

X_2 = rand(dist_2a, Int(round(T/4)))
X_2 = hcat(X_2, rand(dist_2b, Int(round(T/4))))

X = hcat(X_1, X_2)

Y = vcat(ones(Int(round(T/2))), zeros(Int(round(T/2))))

#= cen = 0.

for i in 1:eps

    #---------------------------------------------------------------------------------------
    # Generate Data

    global cen, σ2, stdev, η, ϵ



    global X, Y

    #---------------------------------------------------------------------------------------
    # Train

    @timeit to "RVM" begin

        Λ, C, β, sigma2 = RVM(X, Y, η, alg = :classification)

    end

    global Λ, C, β

    cen += length(Λ)
end

cen = cen/eps =#

# ******************************************************************************************
# Validation

#= Tp = 1000

X_1p = rand(dist_1a, Int(Tp/4))
X_1p = hcat(X_1p, rand(dist_1b, Int(Tp/4)))

X_2p = rand(dist_2a, Int(Tp/4))
X_2p = hcat(X_2p, rand(dist_2b, Int(Tp/4)))

Xp = hcat(X_1p, X_2p)

Yp = Array{Float64,1}(undef, Tp)
Yt = Array{Float64,1}(undef, Tp)

for k in 1:Tp

    Yp[k] = function_approximation(Xp[:,k], Λ, C, η, β = β)

end =#

# ******************************************************************************************
# Plots

show(to)


traces =  [
            scatter(x = X_1[1,:], y = X_1[2,:],
            mode = "markers",
            name = "1",
            marker = attr(size=10, line_width=2),),
            scatter(x = X_2[1,:], y = X_2[2,:],
            mode = "markers",
            name = "0",
            marker = attr(size=10, line_width=2),),
            ]

RVM_plot = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Relevance Vector Machine - classifier",
                        ),
                    title_x = 0.5,
                    xaxis_title = "X",
                    yaxis_title = "Y",
                    ),
                )

display(RVM_plot)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")

μ_weights, σ²_weights, C, verge = GP(X, Y, η; ε = 1e-6, max_itr = 20, tol = 1e-9)

x1_range = range(-1, stop = 1, length = 150)
x2_range = range(-1, stop = 1, length = 150)

P_mat = [GP_predict([x1; x2], μ_weights, σ²_weights, C, η) for x2 in x2_range, x1 in x1_range]

trace_P = heatmap(x = x1_range, y = x2_range, z = P_mat, colorscale = "Electric")
layout = Layout(title = "Classification", xaxis_title = "x₁", yaxis_title = "x₂")
plot_heatmap_P = Plot(trace_P, layout)
