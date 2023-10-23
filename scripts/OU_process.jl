using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using LinearAlgebra
using TimerOutputs
using ArnoldiMethod

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# System Dynamics

# system parameters
α = 4
D = 0.25

# initial condition ranges
x₀_range = [-2 2]

# ******************************************************************************************

#Parameter vector
p = [α, D]

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

#bandwidth, i.e. Kernel Scale
σ² = 0.3
η = 1/(2*σ²)

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

M = 10000

X = rand(Uniform(x₀_range[1], x₀_range[2]), M)#data_train[:,1:end-1]
Y = Array{Float64,1}(undef, M)

t_final_val = 0.5
tspan = (0.0, t_final_val)
μ_s = 0.005
μ_m = 100*μ_s
tₘ_train= 0:μ_m:t_final_val

for ym in 1:M

    prob_train = SDEProblem(Ornstein_Uhlenbeck!, σ_Ornstein_Uhlenbeck!, [X[ym]], tspan, p)

    sol_train = solve(prob_train, saveat = μ_s, wrap = Val(false))
    #u_train = reduce(hcat, sol_train.u)
    xₘ_train = sol_train(tₘ_train) # machine samples of the continuous system
    uₘ_train = reduce(vcat, xₘ_train.u)
    Y[ym] = uₘ_train[end]
end

X = transpose(X)
Y = transpose(Y)

println("\nA. Training")
@timeit to "Training" begin

    @timeit to "Generating gram matrices" begin

        Gxx = Array{Float64,2}(undef, M, M)
        Gyx = Array{Float64,2}(undef, M, M)

        for mr in 1:M
            for mc in 1:M
                Gxx[mr, mc] = Gaussian_kernel(X[:,mr], X[:,mc], η)
                Gyx[mr, mc] = Gaussian_kernel(Y[:,mr], X[:,mc], η)
            end
        end
    end

    @timeit to "RKHS" begin

        K1 = 3

        ε = 1e-8

        invG = Matrix{Float64}(I, M, M) #inv(G + M*ε*I)
        ldiv!(cholesky(Gxx + M*ε*I), invG)

        P = transpose(Gyx)*invG
        U = invG*Gyx

        decomp, history  = partialschur(U, nev = K1,
        tol = 1e-10, restarts = 200, which = LM())
        λu, Vu = partialeigen(decomp)

        Ku = length(λu)

        λu = λu[end:-1:1]
        Vu = Vu[:, end:-1:1]

        λu = λu./real(λu[1])
        Vu = Vu * Diagonal(λu)

        Wu = transpose(Vu)

        maximum(abs.(U*Vu .- Vu*diagm(λu)))

        decomp, history  = partialschur(P, nev = K1,
        tol = 1e-6, restarts = 200, which = LM())
        λp, Vp = partialeigen(decomp)

        Kp = length(λp)

        λp = λp[end:-1:1]
        Vp = Vp[:, end:-1:1]

        λp = λp./real(λp[1])
        Vp = Vp * Diagonal(λp)

        Wp = transpose(invG*Vp)

        maximum(abs.(P*Vp .- Vp*diagm(λp)))

    end

    #= @timeit to "KEDMD" begin

        num_basis = 4

        ε = 0#1e-8

        Σ², Q = eigen(Gxx + M*ε*I) # G = Q*diagm(Σ²)*transpose(Q)

        #Σ² = abs.(Σ²[end:-1:1])
        Σ² = Σ²[end:-1:1]
        Q = Q[:, end:-1:1]

        Σ = sqrt.(Σ²[1:num_basis])
        Q = Q[:,1:num_basis]

        Σ = Diagonal(Σ)
        Σ_inv = inv(Σ)
        Uproj = Σ_inv*transpose(Q)*Gyx*Q*Σ_inv

        eigval, Vproj = eigen(Uproj) # U is not the Koopman matrix, it's a projection

        eigval = eigval[end:-1:1]
        Vproj = Vproj[:, end:-1:1]

        eigval = eigval./real(eigval[1])
        Vproj = Vproj * Diagonal(eigval)

        Vinv = Vproj \ Matrix{Float64}(I, num_basis, num_basis)
        Phi = X*Q*Σ_inv*transpose(Vinv) # rows are modes

        W = transpose(Vproj)*Σ_inv*transpose(Q)

    end =#

    @timeit to "predict" begin

        ϑ = Array{Float64,1}(undef, M)

        x_val = transpose(collect(-2:0.005:2))

        N = length(x_val)

        ϕu = Array{ComplexF64,2}(undef, N, Ku)
        ϕp = Array{ComplexF64,2}(undef, N, Kp)

        for n in 1:N

            for m in 1:M

                ϑ[m] = Gaussian_kernel(X[:, m], x_val[:,n], η)
            end

            ϕp[n,:] = Wp*ϑ
            ϕu[n,:] = Wu*ϑ

        end

        ϕp = real.(ϕp./maximum(abs.(ϕp), dims = 1))
        ϕu = real.(ϕu./maximum(abs.(ϕu), dims = 1))
    end
end

# ******************************************************************************************
# Koopman Mode Decomposition

plot_ϕu = plot(vec(x_val), ϕu,
                Layout(
                    title = attr(
                        text = "Real(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Real(ϕ)",
                    ),
                )

display(plot_ϕu)

plot_ϕp = plot(vec(x_val), ϕp,
                Layout(
                    title = attr(
                        text = "Real(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Real(ϕ)",
                    ),
                )

display(plot_ϕp)

show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
