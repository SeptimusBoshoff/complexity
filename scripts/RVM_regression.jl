using Complexity
using Distributions
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using Random

println("\n...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........\n")

T = 1000 # samples

width = 2.5 # Gaussian kernel width - standard deviation
eps = 2 # episodes
stdev = 0.1

x_range = [-10, 10]

# ******************************************************************************************
# Training

to = TimerOutput()

Y = Array{Float64,1}(undef, T)
X = Array{Float64,2}(undef, 2, T)

η = 1/(2*width^2)
ϵ = 1e-8

X = rand(Uniform(x_range[1], x_range[2]), 2, T)

average_distance = 0.0

for i in 1:T

    global average_distance, X

    distance = 0

    for j in 1:T

        if i != j
            distance += norm(X[:, i] .- X[:, j])
        end
    end

    average_distance += distance/T
end

average_distance = average_distance / (4*T)

η = 1/(2*average_distance^2)

println("\t\taverage distance = ", round(average_distance, digits = 3))
println("\t\tη = ", round(η, digits = 3))

#= CGD
    η = 0.06
    ϵ = 0.118
=#

cen = 0.
σ2 = 0.

for i in 1:eps

    #---------------------------------------------------------------------------------------
    # Generate Data

    global cen, σ2, stdev, η, ϵ

    X = rand(Uniform(x_range[1], x_range[2]), 2, T)

    for k in 1:T

        if isapprox(norm(X[:, k]), 0.0, atol = 1e-12)

            Y[k] = 1.
        else

            Y[k] = sin(norm(X[:, k]))/norm(X[:, k]) # two-dimensional sinc function
        end
    end

    Yn = Y .+ stdev*randn(T)

    global X

    #---------------------------------------------------------------------------------------
    # Train

    @timeit to "RVM" begin

        Λ, C, β, sigma2 = RVM(X, Yn, η, noise_itr = 5, σ2 = 0.01)

        #Λ, C, β, error  = OMP(X, η, Yn, 0.1, N = 30)

        #Λ, β, ϵ, η, Press, verge = CGD(X, Yn; ϵ = ϵ, η = η)

        #= G = Gramian(X, η)
        Λ, β = kernel_regression(G, Yn, ϵ) =#
        #C = copy(X)

        #= μ = 0.1
        C, _, G, _ = ALD(X, collect(1:T), μ, η)

        Λ = transpose(Yn)*((transpose(G)*G + ϵ*I)\transpose(G))
        β = 0 =#

    end

    global Λ, C, β

    σ2 += sigma2
    cen += length(Λ)
end

cen = cen/eps
σ2 = σ2/eps

σ = sqrt(σ2)

# ******************************************************************************************
# Validation

Xp = collect(x_range[1]:0.1:x_range[2])
Tp = length(Xp)
Yp = Array{Float64,1}(undef, Tp)
Yt = Array{Float64,1}(undef, Tp)

Trms = 1000

RMSE = 0.
N = 10 # episodes

for n in 1:N

    global RMSE, X, η

    rmse = 0.

    for k in 1:Trms

        Xrms = rand(Uniform(x_range[1], x_range[2]), 2) #X[:,k]#

        Yf = function_approximation(Xrms, Λ, C, η, β = β)

        if isapprox(norm(Xrms), 0.0, atol = 1e-6)

            Yrms = 1.
        else

            Yrms = sin(norm(Xrms))/norm(Xrms)
        end

        rmse += (Yf - Yrms)^2
    end

    rmse = sqrt(rmse/Trms)

    RMSE += rmse
end

RMSE = RMSE/N

for k in 1:Tp

    Yp[k] = function_approximation([Xp[k]; 0], Λ, C, η, β = β)

    if isapprox(norm([Xp[k]; 0]), 0.0, atol = 1e-6)

        Yt[k] = 1.
    else

        Yt[k] = sin(norm([Xp[k]; 0]))/norm([Xp[k]; 0])
    end
end

Ytn = Yt .+ stdev*randn(Tp)

# ******************************************************************************************
# Plots

println("\t\tstandard deviation \t= ", round(σ, digits = 3))
#println("\t\tvariance \t\t= ", round(σ2, digits = 3))
println("\t\tcentres \t\t= ", round(cen, digits = 3))
println("\t\tRMS error \t\t= ", round(RMSE, digits = 3))

show(to)

error_lower = Yp .+ σ
error_upper = Yp .- σ

trace_error_band = scatter(
                        x = vcat(Xp, reverse(Xp)),
                        y = vcat(error_upper, reverse(error_lower)),
                        fill = "toself",
                        fillcolor = "rgba(200,0,80,0.3)",
                        line = attr(color = "rgba(255,255,255,0)"),
                        hoverinfo = "skip",
                        showlegend = false
                        )

traces =  [
            trace_error_band,
            scatter(x = Xp, y = Yp, mode="lines", name = "Approximation",
                    line = attr(color = "rgba(255,0,0,1.0)"),),
            #scatter(x = Xp, y = Ytn, mode="markers", name = "noisy Targets"),
            scatter(x = Xp,
                    y = Yt,
                    mode="lines",
                    name = "Target",
                    line = attr(color = "rgba(0,0,255,1.0)"),),
            ]

RVM_plot = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Relevance Vector Machine - sinc(x) = sin(||x||)/||x||",
                        ),
                    title_x = 0.5,
                    xaxis_title = "X",
                    yaxis_title = "Y",
                    ),
                )

display(RVM_plot)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
