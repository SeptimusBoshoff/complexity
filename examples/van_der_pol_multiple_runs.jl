using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS
using TimerOutputs

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# System Dynamics

# system parameters
μ = 2

# initial condition ranges
x₀_range = [-2.5, 2.5]
y₀_range = [-4, 4]

#= x₀_range = [1.012, 1.012001]
y₀_range = [-0.88001, -0.88] =#

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

num_eps = 2
t_final_train = 150 # max simulation time (seconds)

u0 = [rand(Uniform(x₀_range[1], x₀_range[2]), num_eps),
        rand(Uniform(y₀_range[1], y₀_range[2]), num_eps)] # initial conditions


u0 = reduce(hcat, u0)

#u0[1,:] = [1.012, -0.88]

# ******************************************************************************************

tₛ_train = 0:μ_s:t_final_train # system time
tₘ_train = 0:μ_m:t_final_train # machine time
tspan = (0.0, t_final_train)

data_train = Vector{Vector{Vector{Float64}}}(undef, num_eps)

for i in 1:num_eps

    global data_train

    prob_train = SDEProblem(Van_der_Pol!, σ_Van_der_Pol!, u0[i,:], tspan, p)

    sol_train = solve(prob_train, saveat = μ_s, wrap = Val(false))

    xₘ_train = sol_train(tₘ_train) # machine samples of the continuous system
    uₘ_train = reduce(hcat, xₘ_train.u)

    # positions
    x_train = uₘ_train[1,:] .+ 0.1*randn(length(tₘ_train))
    y_train = uₘ_train[2,:] .+ 0.1*randn(length(tₘ_train))

    data_train[i] = [vec(x_train), vec(y_train)]
end

N_train = length(data_train[1][1]) # number of samples

#-------------------------------------------------------------------------------------------
# Validation Data: Damped Pendulum Dynamics - time simulation

t_final_val = 200 # max simulation time (seconds)

u0 = [rand(Uniform(x₀_range[1], x₀_range[2])),
        rand(Uniform(y₀_range[1], y₀_range[2]))] # initial conditions

u0 = [-1.941, 0.303]
#u0 = [-2, 4]
#u0 = u0[1,:]

# ******************************************************************************************

tₛ_val = 0:μ_s:t_final_val # system time
tₘ_val= 0:μ_m:t_final_val # machine time
tspan = (0.0, t_final_val)

prob_val = SDEProblem(Van_der_Pol!, σ_Van_der_Pol!, u0, tspan, p)

sol_val = solve(prob_val, saveat = μ_s, wrap = Val(false))
u_val = reduce(hcat, sol_val.u)

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

# ******************************************************************************************
# Validation

xₘ_val = sol_val(tₘ_val) # machine samples of the continuous system
uₘ_val = reduce(hcat, xₘ_val.u)

# positions
x_val = uₘ_val[1,:]
y_val = uₘ_val[2,:]

data_val = [vec(x_val), vec(y_val)]

N_val = length(data_val[1])

# ******************************************************************************************

#bandwidth, i.e. Kernel Scale
scale = [maximum(data_val[1]) - minimum(data_val[1]);
        maximum(data_val[2]) - minimum(data_val[2])]

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

println("\nA. Training")
@timeit to "Training" begin

    @timeit to "Generating gram matrices" begin
        println("\n\t1. Generating gram matrices")
        Gx, Gy, index_map, series_list = series_Gxy(data_train, scale, npast, nfuture)
    end

    @timeit to "Computing Gs" begin
        # Compute the state similarity matrix.
        # Embedding to get the similarity matrix between conditional distributions
        println("\n\t2. Computing Gs")
        Gs, embedder = embed_states(Gx, Gy, return_embedder = true)
    end

    @timeit to "Projection" begin

        # Compute a spectral basis for representing the causal states.
        # Find a reduced dimension embedding and extract the significant coordinates"

        println("\n\t3. Projection")
        coords_train = spectral_basis(Gs, num_basis = 50)
    end

    @timeit to "Forward Shift Operator" begin

        # This is the forward operator in diffusion state space. It is built from
        # consecutive indices in the index map. Data series formed by multiple contiguous
        # time blocks are supported, as well as the handling of NaN values

        println("\n\t4. Forward Shift Operator")

        h_rank = 1 #100

        DMD, Svd, s_residual  = shift_operator(coords_train, alg = :hankel, hankel_rank = h_rank, index_map = index_map, residual = true) #svd_tr = 0.9985
        println("\tResidual: ", round.(s_residual, digits = 5))

        #shift_op = shift_operator(coords_train, alg = :GA, index_map = index_map)
    end

    @timeit to "Expectation Operator" begin

        # This is the expectation operator, using its default function that predicts
        # the first entry in the future sequence from the current state distribution.
        # You can specify other functions, see the documentation

        println("\n\t5. Expectation Operator")

        d_rank = 1

        expect_op, e_residual = expectation_operator(coords_train, index_map, series_list, delay_coords = d_rank)
        println("\tResidual: ", round.(e_residual, digits = 5))
    end

end

# ******************************************************************************************
# Validation

println("\nB. Validation")
@timeit to "Validation" begin

    @timeit to "New coordinates" begin

        println("\n\t6. New coordinates")

        ic_rank = maximum([h_rank; d_rank])

        # initial conditions
        ic = [data_val[1][1:npast + ic_rank - 1],
                data_val[2][1:npast + ic_rank - 1]]

        @timeit to "Build Similarity vector" begin

            # step 1. Build a kernel similarity vector with sample data
            Kx = series_newKx(ic, series_list, index_map, scale, npast)

        end

        @timeit to "Embed Similarity vector" begin
            # step 2. Embed to get similarity vector in state space
            Ks = embed_Kx(Kx, Gx, embedder)
        end

        @timeit to "Linear regression" begin

            # step 3. Build a probability distribution over states.
            coords_ic = new_coords(Ks, Gs, coords_train, alg = :ldiv)

        end

        #coords_ic = coords_train[:, 1:ic_rank]

    end

    @timeit to "Prediction" begin

        println("\n\t7. Prediction")

        pred_hor = length(tₘ_val) - npast + 1 - ic_rank # prediction horizon

        pred, coords_pred = predict(pred_hor, coords_ic[:, :], expect_op, DMD = DMD)
        #pred, coords_pred = predict(pred_hor, coords_ic[:, :], expect_op, shift_op = shift_op)
        #,knn_convexity = 5, knndim = 10, coords = coords_train, extent = 0.05)

    end
end

#-------------------------------------------------------------------------------------------

# Measurement State Space Data Frame

nans_train = fill(NaN, N_val - N_train)
nans_pred = fill(NaN, N_val - pred_hor)

SS = DataFrame(x_train = vec([data_train[1][1]; nans_train]),
                y_train = vec([data_train[1][2]; nans_train]),
                x_val = data_val[1],
                y_val = data_val[2],
                x_pred = vec([nans_pred; pred[1]]),
                y_pred = vec([nans_pred; pred[2]]),
                t = tₘ_val)

# Causal State Space Data Frame

N_ctrain = size(Gs,1)
num_nans = (pred_hor + ic_rank) - N_ctrain
nans_causal_end = fill(NaN, abs(num_nans))
nans_causal_start = fill(NaN, ic_rank)

if num_nans > 0

    tₘ_DSS = tₘ_val[N_val - pred_hor - ic_rank + 1]:μ_m:tₘ_val[end]

    DSS = DataFrame(Ψ₁_train = vec([coords_train[2, :]; nans_causal_end]),
                        Ψ₂_train = vec([coords_train[3, :]; nans_causal_end]),
                        Ψ₃_train = vec([coords_train[4, :]; nans_causal_end]),
                        Ψ₁_pred = [nans_causal_start; coords_pred[2,:]],
                        Ψ₂_pred = [nans_causal_start; coords_pred[3,:]],
                        Ψ₃_pred = [nans_causal_start; coords_pred[4,:]],
                        t = tₘ_DSS)

else

    tstart = tₘ_val[N_val - pred_hor + 1]
    tend = tstart + (μ_m*(pred_hor + abs(num_nans)))

    tₘ_DSS = tstart:μ_m:tend

    DSS = DataFrame(Ψ₁_train = coords_train[2, :],
                        Ψ₂_train = coords_train[3, :],
                        Ψ₃_train = coords_train[4, :],
                        Ψ₁_pred = vec([NaN; coords_pred[2,:]; nans_causal_end]),
                        Ψ₂_pred = vec([NaN; coords_pred[3,:]; nans_causal_end]),
                        Ψ₃_pred = vec([NaN; coords_pred[4,:]; nans_causal_end]),
                        t = tₘ_DSS)

end

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
# Diffustion State Space Time domain signals

traces = [scatter(DSS, x = :t, y = :Ψ₁_train, name = "train-Ψ₁"),
            scatter(DSS, x = :t, y = :Ψ₂_train, name = "train-Ψ₂"),
            scatter(DSS, x = :t, y = :Ψ₁_pred, name = "pred-Ψ₁"),
            scatter(DSS, x = :t, y = :Ψ₂_pred, name = "pred-Ψ₂")]

plot_DSS_t = plot(traces,
                Layout(
                    title = attr(
                        text = "Diffusion State Space: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "Ψ₁, Ψ₂",
                    ),
                )

#display(plot_DSS_t)

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
# Diffusion (Reconstructed) State Space

traces = [scatter(DSS, x = :Ψ₁_train, y = :Ψ₂_train, name = "train", mode = "lines"),
        scatter(DSS, x = :Ψ₁_pred, y = :Ψ₂_pred, name = "pred", mode = "lines")]

plot_DSS = plot(traces,
                Layout(
                    title = attr(
                        text = "Diffusion State Space",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "Ψ₁",
                        yaxis_title = "Ψ₂",
                    ),
                    ),
                )

#display(plot_DSS)

traces = [scatter3d(DSS, x = :Ψ₁_train, y = :Ψ₂_train, z = :Ψ₃_train,
                    name = "train", mode = "lines"),
        scatter3d(DSS, x = :Ψ₁_pred, y = :Ψ₂_pred, z = :Ψ₃_pred,
                    name = "pred", mode = "lines")]

plot_DSS_3d = plot(traces,
                Layout(
                    title = attr(
                        text = "Diffusion State Space",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "Ψ₁",
                        yaxis_title = "Ψ₂",
                        zaxis_title = "Ψ₃",
                    ),
                    scene_aspectratio = attr(x = 2, y = 2, z = 4),
                    scene_camera = attr(
                        up = attr(x = 1, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 2, y = 2, z = 2)
                        ),
                    ),
                )

display(plot_DSS_3d)

# ******************************************************************************************
# Koopman Modes Decomposition

order = sortperm(abs.(DMD[2][:]), rev = true)

Ω = log.(DMD[2])/μ_m # continuous time DMD eigenvalues

Λ1 = (abs.(DMD[2][order[:]])./(sum(abs.(DMD[2]))))

plot_Λ1 = plot(Λ1,
                Layout(
                    title = attr(
                        text = "Discrete DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    yaxis_title = "|Λ| [%]",
                    ),
                )

#display(plot_Λ1)

Λ2 = scatter(x = (real.(DMD[2][order[:]])),
            y = (imag.(DMD[2][order[:]])),
            mode="markers",
            marker=attr(color = order,
                    colorscale = "Bluered",
                    size=6),
            )


plot_Λ2 = plot(Λ2,
                Layout(
                    title = attr(
                        text = "Discrete DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    xaxis_title = "Real",
                    yaxis_title = "Imaginary",
                    ),
                )

display(plot_Λ2)

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

trunk = 10
Phi = DMD[1]

traces = [(real(Phi[:,order[1:trunk]])), (imag(Phi[:,order[1:trunk]]))]

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
        title_text = "Koopman modes",
        title_x = 0.5,)
#display(plot_ϕ)

episode_length = Int(round(size(Gs, 1)/num_eps, digits = 0))

#= real_ψ = transpose((real.((DMD[3][end-4:end-1,:])*coords_train[:,1:episode_length])))
imag_ψ = transpose((imag.((DMD[3][end-4:end-1,:])*coords_train[:,1:episode_length])))

plot_real_K_func = plot(real_ψ,
                Layout(
                    title = attr(
                        text = "real",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "time",
                    ),
                    ),
                )

plot_imag_K_func = plot(imag_ψ,
                Layout(
                    title = attr(
                        text = "imaginary",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "time",
                    ),
                    ),
                )

plot_Kpman_func = [plot_real_K_func; plot_imag_K_func]
relayout!(plot_Kpman_func,
title_text = "Koopman eigenfunctions",
title_x = 0.5,) =#
#display(plot_Kpman_func)

# ******************************************************************************************
# Single Value Decomposition

trunk = 12
Σ = (Svd[2][1:trunk]./(sum(Svd[2])))

plot_Σ = plot(Σ,
                Layout(
                    title = attr(
                        text = "Singular Values",
                    ),
                    title_x = 0.5,
                    yaxis_title = "Σ [%]",
                    ),
                )

#display(plot_Σ)

U = real(Svd[1][:, 2:trunk])

plot_U = plot(U,
                Layout(
                    title = attr(
                        text = "Left Singular Vectors", # SVD modes
                    ),
                    title_x = 0.5,
                    yaxis_title = "U",
                    ),
                )

#display(plot_U)

V = real(Svd[3][:, 1:trunk])

plot_V = plot(V,
                Layout(
                    title = attr(
                        text = "Right Singular Vectors",
                    ),
                    title_x = 0.5,
                    yaxis_title = "V",
                    ),
                )

#display(plot_V)
# ******************************************************************************************

show(to)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
