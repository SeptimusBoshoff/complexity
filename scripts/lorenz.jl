using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------------------
# System Dynamics

# system paramaters
σ = 10 # Prandtl
ρ = 28 # Raleigh
β = 8/3 # geometric aspect ratio

# initial condition ranges
x₀_range = [-17, 17]
y₀_range = [-23, 23]
z₀_range = [7, 43]

# ******************************************************************************************

#Parameter vector
p = [σ, ρ, β]

#-------------------------------------------------------------------------------------------
# Machine parameters

sampling = 20 # machine retain 1 in every so many samples

history = 5 # backward trajectory [seconds]
future = 5 # forward trajectory [seconds]

# ******************************************************************************************
μ_s = 0.005 # time step size (seconds)
μ_m = sampling*μ_s # machine time step
npast = convert(Int, round(history/μ_m)) #Past Series sample size
nfuture = convert(Int, round(future/μ_m)) #Future series sample size

window_size = npast + nfuture

#-------------------------------------------------------------------------------------------
# Training Data:

u0 = [rand(Uniform(x₀_range[1], x₀_range[2])),
        rand(Uniform(y₀_range[1], y₀_range[2])),
        rand(Uniform(z₀_range[1], z₀_range[2]))] # initial conditions
u0 = [9.9116, 13.4339, 23.8203]

μ_s = 0.005 # time step size (seconds)
t_final_train = 300 # max simulation time (seconds)

# ******************************************************************************************

tₛ_train = 0:μ_s:t_final_train # system time
tₘ_train= 0:μ_m:t_final_train # machine time
tspan = (0.0, t_final_train)

prob_train = ODEProblem(Lorenz!, u0, tspan, p)

sol_train = solve(prob_train, saveat = μ_s, wrap = Val(false))
u_train = reduce(hcat, sol_train.u)

#-------------------------------------------------------------------------------------------
# Validation Data:

u0 = [rand(Uniform(x₀_range[1], x₀_range[2])),
        rand(Uniform(y₀_range[1], y₀_range[2])),
        rand(Uniform(z₀_range[1], z₀_range[2]))] # initial conditions
u0 = [-5.479, 2.5106, 32.5884]# .+ 1e-1
u0 = [9.9116, 13.4339, 23.8203]

μ_s = 0.005 # time step size (seconds)
t_final_val = 300 # max simulation time (seconds)

# ******************************************************************************************

tₛ_val = 0:μ_s:t_final_val # system time
tₘ_val= 0:μ_m:t_final_val # machine time
tspan = (0.0, t_final_val)

prob_val = ODEProblem(Lorenz!, u0, tspan, p)

sol_val = solve(prob_val, saveat = μ_s, wrap = Val(false))
u_val = reduce(hcat, sol_val.u)

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

# ******************************************************************************************
# Training

xₘ_train = sol_train(tₘ_train) # machine samples of the continuous system
uₘ_train = reduce(hcat, xₘ_train.u)

# positions
x_train = round.(uₘ_train[1,:], digits = 1)
y_train = round.(uₘ_train[2,:], digits = 1)
z_train = round.(uₘ_train[3,:], digits = 1)

data_train = [vec(x_train),
                vec(y_train),
                vec(z_train)]

#bandwidth, i.e. Kernel Scale
scale = [maximum(data_train[1]) - minimum(data_train[1]);
        maximum(data_train[2]) - minimum(data_train[2]);
        maximum(data_train[3]) - minimum(data_train[3]);]

# ******************************************************************************************
# Validation

xₘ_val = sol_val(tₘ_val) # machine samples of the continuous system
uₘ_val = reduce(hcat, xₘ_val.u)

# positions
x_val = uₘ_val[1,:]
y_val = uₘ_val[2,:]
z_val = uₘ_val[3,:]

data_val = [vec(x_val), vec(y_val), vec(z_val)]

N = length(data_train[1]) # number of samples

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

println("\nA. Training")
@time begin

    @time begin
        println("\n1. Generating gram matrices")
        Gx, Gy, index_map = series_Gxy(data_train, scale, npast, nfuture)
    end

    @time begin
        # Compute the state similarity matrix.
        # Embedding to get the similarity matrix between conditional distributions
        println("\n2. Computing Gs")
        Gs, embedder = embed_states(Gx, Gy, return_embedder = true)
    end

    @time begin
        # Compute a spectral basis for representing the causal states.
        # Find a reduced dimension embedding and extract the significant coordinates"
        println("\n3. Projection")
        eigenvalues, basis, coords_train = spectral_basis(Gs, num_basis = 100)
    end

    @time begin
        # This is the forward operator in diffusion state space. It is built from
        # consecutive indices in the index map. Data series formed by multiple contiguous
        # time blocks are supported, as well as the handling of NaN values
        println("\n4. Forward Shift Operator")
        shift_op, DMD, SVDr = shift_operator(coords_train, alg = :hankel, hankel_rank = 2000, svd_tr = 99)
        #shift_op = shift_operator(coords_train, alg = :nnls)
    end

    @time begin
        # This is the expectation operator, using its default function that predicts
        # the first entry in the future sequence from the current state distribution.
        # You can specify other functions, see the documentation
        println("\n5. Expectation Operator")
        expect_op = expectation_operator(coords_train, index_map, data_train)
    end

end

# ******************************************************************************************
# Validation

println("\nB. Validation")
@time begin

    println("6. Prediction")

    npast_val = npast

    # initial conditions
    ic = [data_val[1][1:npast_val + 1],
            data_val[2][1:npast_val + 1],
            data_val[3][1:npast_val + 1]]

    # step 1. Build a kernel similarity vector with sample data
    Kx = series_newKx(ic, data_train, index_map, scale, npast_val)
    # step 2. Embed to get similarity vector in state space
    Ks = embed_Kx(Kx, Gx, embedder)
    # step 3. Build a probability distribution over states.
    coords_val = new_coords(Ks, Gs, coords_train)

    pred_hor = length(tₘ_val) - npast_val # prediction horizon

    #= pred, coords_pred = predict(pred_hor, coords_val[:, 1], expect_op, shift_op = shift_op,
    knn_convexity = 6, knndim = 100, coords = coords_train, extent = 0.05) =#

    pred, coords_pred = predict(pred_hor, coords_val[:, :], expect_op, DMD = DMD)
    #knn_convexity = 1, knndim = 10, coords = coords_train, extent = 0.05)

end

#-------------------------------------------------------------------------------------------
nans = Array{Float64, 1}(undef, npast_val)
nans = fill!(nans, NaN)

nans2 = Array{Float64, 1}(undef, length(vec([nans; pred[1]])) - length(data_train[1]))
nans2 = fill!(nans2, NaN)

# State Spaces and Data Frames
SS = DataFrame(x_train = vec([data_train[1]; nans2]),
                y_train = vec([data_train[2]; nans2]),
                z_train = vec([data_train[3]; nans2]),
                x_val = data_val[1],
                y_val = data_val[2],
                z_val = data_val[3],
                x_pred = vec([nans; pred[1]]),
                y_pred = vec([nans; pred[2]]),
                z_pred = vec([nans; pred[3]]),
                t = tₘ_val)

nans = Array{Float64, 1}(undef, length(coords_pred[2,:]) - length(coords_train[2,:]))
nans = fill!(nans, NaN)

DSS = DataFrame(Ψ₁_train = vec([coords_train[2,:]; nans]),
                Ψ₂_train = vec([coords_train[3,:]; nans]),
                Ψ₃_train = vec([coords_train[4,:]; nans]),
                Ψ₁_pred = coords_pred[2,:],
                Ψ₂_pred = coords_pred[3,:],
                Ψ₃_pred = coords_pred[4,:],
                t = tₘ_val[npast_val+1:end])

#-------------------------------------------------------------------------------------------
# Plots

# ******************************************************************************************
# State Space Time domain signals

traces = [scatter(SS, x = :t, y = :x_val, name = "val-x"),
            scatter(SS, x = :t, y = :y_val, name = "val-y"),
            scatter(SS, x = :t, y = :z_val, name = "val-z"),
            scatter(SS, x = :t, y = :x_train, name = "train-x"),
            scatter(SS, x = :t, y = :y_train, name = "train-y"),
            scatter(SS, x = :t, y = :z_train, name = "train-z"),
            scatter(SS, x = :t, y = :x_pred, name = "pred-x"),
            scatter(SS, x = :t, y = :y_pred, name = "pred-y"),
            scatter(SS, x = :t, y = :z_pred, name = "pred-z")]

plot_SS_t = plot(traces,
                Layout(
                    title = attr(
                        text = "State Space: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "x, y, z [m]",
                    ),
                )

display(plot_SS_t)

# ******************************************************************************************
# Diffustion State Space Time domain signals

traces = [scatter(DSS, x = :t, y = :Ψ₁_train, name = "train-Ψ₁"),
            scatter(DSS, x = :t, y = :Ψ₂_train, name = "train-Ψ₂"),
            scatter(DSS, x = :t, y = :Ψ₃_train, name = "train-Ψ₃"),
            scatter(DSS, x = :t, y = :Ψ₁_pred, name = "pred-Ψ₁"),
            scatter(DSS, x = :t, y = :Ψ₂_pred, name = "pred-Ψ₂"),
            scatter(DSS, x = :t, y = :Ψ₃_pred, name = "pred-Ψ₃")]

plot_DSS_t = plot(traces,
                Layout(
                    title = attr(
                        text = "Diffusion State Space: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "Ψ₁, Ψ₂, Ψ₃",
                    ),
                )

#display(plot_DSS_t)

# ******************************************************************************************
# State Space

traces = [scatter3d(SS, x = :x_val, y = :y_val, z = :z_val, name = "val", mode = "lines"),
        #scatter3d(SS, x = :x_train, y = :y_train, z = :z_train, name = "train", mode = "lines"),
        scatter3d(SS, x = :x_pred, y = :y_pred, z = :z_pred, name = "pred", mode = "lines")]

plot_SS_3d = plot(traces,
                Layout(
                    title = attr(
                        text = "State Space",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "x [m]",
                        yaxis_title = "y [m]",
                        zaxis_title = "z [m]",
                    ),
                    scene_aspectratio = attr(x = 2, y = 2, z = 2),
                    scene_camera = attr(
                        up = attr(x = 1, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 2, y = 2, z = 2)
                        ),
                    ),
                )

#display(plot_SS_3d)


# ******************************************************************************************
# Diffusion (Reconstructed) State Space

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

#display(plot_DSS_3d)

# ******************************************************************************************
# ******************************************************************************************
# Koopman Modes (Phi)

tr = 10

Phi = DMD[1]

traces = [(real(Phi[:,1:tr])), (imag(Phi[:,1:tr]))]

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
        title_text = "DMD eigenvectors ≈ Koopman Eigenfunctions",
        title_x = 0.5,)
#display(plot_ϕ)

Λ = (abs.(DMD[2][1:tr])./(sum(abs.(DMD[2]))))

plot_Λ = plot(Λ,
                Layout(
                    title = attr(
                        text = "DMD / Koopman Eigenvalues",
                    ),
                    title_x = 0.5,
                    yaxis_title = "Λ [%]",
                    ),
                )

#display(plot_Λ)

# ******************************************************************************************
# Single Value Decomposition

tr = 80
Σ = (SVDr[2][2:tr]./(sum(SVDr[2])))

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

U = real(SVDr[1][:, 1:tr])

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

V = real(SVDr[3][:, 1:tr])

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

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
