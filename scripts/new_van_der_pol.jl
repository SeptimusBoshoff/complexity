using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs
using ArnoldiMethod
using LinearAlgebra

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

history = 10 # backward trajectory [seconds]
future = 10 # forward trajectory [seconds]

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

# positions
x_train = uₘ_train[1,:] .+ 0.05*randn(length(uₘ_train[1,:]))
y_train = uₘ_train[2,:] .+ 0.05*randn(length(uₘ_train[2,:]))

data_train = [vec(x_train), vec(y_train)]

#bandwidth, i.e. Kernel Scale
scale = 1*[maximum(data_train[1]) - minimum(data_train[1]);
        maximum(data_train[2]) - minimum(data_train[2])]

# ******************************************************************************************
# Validation

xₘ_val = sol_val(tₘ_val) # machine samples of the continuous system
uₘ_val = reduce(hcat, xₘ_val.u)

# positions
x_val = uₘ_val[1,:]
y_val = uₘ_val[2,:]

data_val = [vec(x_val), vec(y_val)]

N = length(data_train[1]) # number of samples

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

println("\nA. Training")
@timeit to "Training" begin

    println("\t1. Generating gram matrices")
    @timeit to "Generating gram matrices" begin

        Gx, Gy, index_map = series_Gxy(data_train, scale, npast, nfuture)

    end

    println("\t2. Computing Gs")
    @timeit to "Computing Gs" begin

        # Compute the state similarity matrix.
        # Embedding to get the similarity matrix between conditional distributions

        Gs, embedder = embed_states(Gx, Gy)

    end

    println("\t3. Koopman Mode Decomposition")
    @timeit to "Koopman Mode Decomposition" begin

        DMD, residual, φ = KMD(uₘ_train, Gs, index_map, residual = true, num_basis = 100, ε = 1e-8, alg = :RKHS)
        println("\tResidual: ", round.(residual, digits = 3))
        norm(real.(φ*transpose(DMD[1])) .- transpose(uₘ_train[:,index_map[1:end-1]]))
        #DMD = KMD(uₘ_train, Gs, index_map, num_basis = 100, alg = :RKHS, ε = 1e-8)

    end

end

# ******************************************************************************************
# Validation

println("\nB. Validation")
@timeit to "Validation" begin

    # initial conditions
    ic = [data_val[1][1:npast],
            data_val[2][1:npast]]

    println("\t4. Similarity vector")
    @timeit to "Similarity vector" begin

        # step 1. Build a kernel similarity vector with sample data
        Kx = series_newKx(ic, data_train, index_map, scale, npast)

    end

    println("\t5. Embed vector")
    @timeit to "Embed vector" begin

        # step 2. Embed to get similarity vector in state space
        Ks = embed_Kx(Kx, Gx, embedder)

    end

    pred_hor = length(tₘ_val) - npast # prediction horizon

    println("\t6. Prediction")
    @timeit to "prediction" begin

        pred = evolve(pred_hor, Ks, DMD)

    end

end

#-------------------------------------------------------------------------------------------
nans = Array{Float64, 1}(undef, npast)
nans = fill!(nans, NaN)

nans2 = Array{Float64, 1}(undef, length(vec([nans; pred[1,:]])) - length(data_train[1]))
nans2 = fill!(nans2, NaN)

# State Spaces and Data Frames
SS = DataFrame(x_train = vec([data_train[1]; nans2]),
                y_train = vec([data_train[2]; nans2]),
                x_val = data_val[1],
                y_val = data_val[2],
                x_pred = vec([nans; pred[1,:]]),
                y_pred = vec([nans; pred[2,:]]),
                t = tₘ_val)

#-------------------------------------------------------------------------------------------
# Plots

# ******************************************************************************************
# State Space Time domain signals

traces = [scatter(SS, x = :t, y = :x_val, name = "val-x"),
            scatter(SS, x = :t, y = :y_val, name = "val-y"),
            scatter(SS, x = :t, y = :x_train, name = "train-x"),
            scatter(SS, x = :t, y = :y_train, name = "train-y"),
            scatter(SS, x = :t, y = :x_pred, name = "pred-x"),
            scatter(SS, x = :t, y = :y_pred, name = "pred-y")]

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
# State Space

traces = [scatter(SS, x = :x_val, y = :y_val, name = "val", mode = "lines"),
        scatter(SS, x = :x_train, y = :y_train, name = "train", mode = "lines"),
        scatter(SS, x = :x_pred, y = :y_pred, name = "pred", mode = "lines")]

plot_SS = plot(traces,
                Layout(
                    title = attr(
                        text = "State Space",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "x [m]",
                        yaxis_title = "y [m]",
                    ),
                    ),
                )

display(plot_SS)

# ******************************************************************************************
# Koopman Modes Decomposition

order = sortperm(abs.(DMD[2][:]), rev = true)

Ω = log.(DMD[2])/μ_m # continuous time DMD eigenvalues

m_disc_Λ = abs.(DMD[2][order[:]])

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

RI_disc_Λ = scatter(x = (real.(DMD[2][order[:]])),
            y = (imag.(DMD[2][order[:]])),
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

tr = 4
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

# ******************************************************************************************

show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
