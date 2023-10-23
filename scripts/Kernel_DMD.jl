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
μ = 2

# initial condition ranges
x₀_range = [-4.5, 4.5]
y₀_range = [-6, 6]

μ_s = 0.005 # dt - time step size (seconds)

# ******************************************************************************************

#Parameter vector
p = [μ]

#-------------------------------------------------------------------------------------------
# Machine parameters

sampling = 10 # machine retain 1 in every so many samples - subsampling

history = 5 # backward trajectory [seconds]
future = 5 # forward trajectory [seconds]

# ******************************************************************************************

μ_m = sampling*μ_s # machine time step
npast = convert(Int, round(history/μ_m)) #Past Series sample size
nfuture = convert(Int, round(future/μ_m)) #Future series sample size

window_size = npast + nfuture

#-------------------------------------------------------------------------------------------
# Training Data - time simulation

u0 = [rand(Uniform(x₀_range[1], x₀_range[2])),
        rand(Uniform(y₀_range[1], y₀_range[2]))] # initial conditions
u0 = [1.012, -0.88]
#u0 = [-2, 10]

t_final_train = 100 # max simulation time (seconds)

# ******************************************************************************************

tₛ_train = 0:μ_s:t_final_train # system time
tₘ_train= 0:μ_m:t_final_train # machine time
tspan = (0.0, t_final_train)

prob_train = ODEProblem(Van_der_Pol!, u0, tspan, p)

sol_train = solve(prob_train, saveat = μ_s, wrap = Val(false))
u_train = reduce(hcat, sol_train.u)

#-------------------------------------------------------------------------------------------
# Validation Data: Damped Pendulum Dynamics - time simulation

u0 = [rand(Uniform(x₀_range[1], x₀_range[2])),
        rand(Uniform(y₀_range[1], y₀_range[2]))] # initial conditions
u0 = [-1.941, 0.303]# .+ 1e-1

t_final_val = 200 # max simulation time (seconds)

# ******************************************************************************************

tₛ_val = 0:μ_s:t_final_val # system time
tₘ_val= 0:μ_m:t_final_val # machine time
tspan = (0.0, t_final_val)

prob_val = ODEProblem(Van_der_Pol!, u0, tspan, p)

sol_val = solve(prob_val, saveat = μ_s, wrap = Val(false))
u_val = reduce(hcat, sol_val.u)

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

# ******************************************************************************************
# Training

xₘ_train = sol_train(tₘ_train) # machine samples of the continuous system
uₘ_train = reduce(hcat, xₘ_train.u)

data_train = uₘ_train

X = data_train[:, 1:end-1]
Y = data_train[:, 2:end]

#bandwidth, i.e. Kernel Scale
η = 100.

# ******************************************************************************************
# Validation

xₘ_val = sol_val(tₘ_val) # machine samples of the continuous system
uₘ_val = reduce(hcat, xₘ_val.u)

data_val = uₘ_val

M = length(data_train[1,:])-1 # number of snapshot pairs

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

println("\nA. Training")
@timeit to "Training" begin

    @timeit to "Generating gram matrices" begin

        Gx = Gramian(data_train, η)
        A = transpose(Gx[1:M, 2:end]) #Gyx
        G = Gx[1:M, 1:M] #Gxx
    end

    @timeit to "eigendecomposition" begin

        K = 101

        #Σ², Q = eigen(G) # G = Q*diagm(Σ²)*transpose(Q)

        decomp, history  = partialschur(G, nev = K,
        tol = 1e-6, restarts = 200, which = LM())
        Σ², Q = partialeigen(decomp)

        #Σ² = abs.(Σ²[end:-1:1])
        Σ² = Σ²[end:-1:1]
        Q = Q[:, end:-1:1]

    end

    @timeit to "Koopman matrix" begin

        Σ = (sqrt.(Σ²))[1:K]
        Σ = Diagonal(Σ)
        Σ_inv = inv(Σ)
        Q = Q[:,1:K]
        U = Σ_inv*transpose(Q)*A*Q*Σ_inv

    end

    @timeit to "RKHS" begin

        K2 = 101

        ε = 1e-8

        invG = Matrix{Float64}(I, M, M)
        ldiv!(cholesky(G + M*ε*I), invG)
        #invG = inv(G + M*ε*I)

        U1 = (invG*A)

        decomp, history  = partialschur(U1, nev = K2,
        tol = 1e-6, restarts = 200, which = LM())
        λ1, V1 = partialeigen(decomp)

        λ1 = λ1[end:-1:1]
        V1 = V1[:, end:-1:1]

        λ1 = λ1./λ1[1]
        V1 = V1 * Diagonal(λ1)

        φ1 = G*V1 # numerically approximated Koopman eigenfunctions
        Ξ1 = φ1 \ transpose(data_train[:,1:M])

        maximum(abs.(U1*V1 .- V1*diagm(λ1)))

    end

    @timeit to "KMD" begin

        λ, V = eigen(U)
        λ = λ[end:-1:1]
        V = V[:, end:-1:1]

        λ = λ./λ[1]
        V = V * Diagonal(λ)

        φ = Q*Σ*V

        ω = V \ Matrix{Float64}(I, K, K)

        Ξ = ω*Σ_inv*transpose(data_train[:,1:M]*Q) # rows are modes

        norm(real.(φ*Ξ) .- transpose(data_train[:,1:M]))
        norm(real.(φ1*Ξ1) .- transpose(data_train[:,1:M]))

    end

    @timeit to "predict" begin

        ϑ = Array{Float64,1}(undef, M)

        x = X[:, 1]

        for m in 1:M

            ϑ[m] = Gaussian_kernel(data_train[: , m], x, η)
        end

        ϕ = transpose(Q*Σ_inv*V)*ϑ

        Xval = Array{Float64, 2}(undef, 2, M)
        Xval1 = Array{Float64, 2}(undef, 2, M)#transpose(real.(φ2*Ξ2))

        b = transpose(V1)*ϑ

        for m in 1:M

            Xval[:, m] = real.(transpose(Ξ)*Diagonal(λ.^(m-1))*ϕ)
            #Xval1[:, m] = real.(transpose(Ξ1)*Diagonal(λ1.^(m-1))*b)
            Xval1[:, m] = real.(transpose(V1*Diagonal(λ1.^(m-1))*Ξ1)*ϑ)

        end
    end

end

# ******************************************************************************************

traces = [scatter(x = tₘ_train, y = Xval[1,1:M], name = "val-x"),
            scatter(x = tₘ_train, y = Xval[2,1:M], name = "val-y"),
            scatter(x = tₘ_train, y = Xval1[1,1:M], name = "val1-x"),
            scatter(x = tₘ_train, y = Xval1[2,1:M], name = "val1-y"),
            scatter(x = tₘ_train, y = data_train[1,1:M], name = "train-x"),
            scatter(x = tₘ_train, y = data_train[2,1:M], name = "train-y"),]

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
# ******************************************************************************************
# Koopman Modes Decomposition

order = sortperm(abs.(λ), rev = true)

Ω = log.(λ)/μ_m # continuous time DMD eigenvalues

m_disc_Λ = abs.(λ[order[:]])

plot_m_disc_Λ = plot(m_disc_Λ,
                Layout(
                    title = attr(
                        text = "Discrete DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    yaxis_title = "|Λ|",
                    ),
                )

#display(plot_m_disc_Λ)

RI_disc_Λ = scatter(x = (real.(λ[order[:]])),
            y = (imag.(λ[order[:]])),
            mode="markers",
            marker=attr(color = order,
                    colorscale = "Bluered",
                    size=6),
            )


plot_RI_disc_Λ = plot(RI_disc_Λ,
                Layout(
                    title = attr(
                        text = "Discrete DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    xaxis_title = "Real",
                    yaxis_title = "Imaginary",
                    ),
                )

#display(plot_RI_disc_Λ)

Ωo = scatter(x = real.(Ω[order[:]]),
            y = imag.(Ω[order[:]]),
            mode="markers",
            marker=attr(color = order,
                    colorscale = "Bluered",
                    size=6),
            )


plot_Ωo = plot(Ωo,
                Layout(
                    title = attr(
                        text = "Continuous DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    xaxis_title = "Real",
                    yaxis_title = "Imaginary",
                    ),
                )

#display(plot_Ωo)

tr = 3
Phi = φ[:, order[1:tr]]

traces = [real(Phi), imag(Phi)]

plot_ϕr = plot(traces[1],
                Layout(
                    title = attr(
                        text = "Real(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Real(ϕ)",
                    ),
                )

plot_ϕi = plot(traces[2],
                Layout(
                    title = attr(
                        text = "Imaginary(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Imag(ϕ)",
                    ),
                )

plot_ϕ = [plot_ϕr; plot_ϕi]
relayout!(plot_ϕ,
        title_text = "Koopman Eigenfunctions",
        title_x = 0.5,)
#display(plot_ϕ)

tr = 3
Phi1 = φ1[:, 1:tr]

traces = [real(Phi1), imag(Phi1)]

plot_ϕr1 = plot(traces[1],
                Layout(
                    title = attr(
                        text = "Real(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Real(ϕ)",
                    ),
                )

plot_ϕi1 = plot(traces[2],
                Layout(
                    title = attr(
                        text = "Imaginary(ϕ)",
                    ),
                    title_x = 0.5,
                    #yaxis_title = "Imag(ϕ)",
                    ),
                )

plot_ϕ1 = [plot_ϕr1; plot_ϕi1]
relayout!(plot_ϕ1,
        title_text = "Koopman Eigenfunctions",
        title_x = 0.5,)
#display(plot_ϕ1)
show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
