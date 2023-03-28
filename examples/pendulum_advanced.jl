using Complexity
using Distributions
using DifferentialEquations
using DataFrames
using PlotlyJS

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------------------
# Damped Pendulum Dynamics

μ_s = 0.005 # time step size (seconds)
t_final = 30 # max simulation time (seconds)

tₛ = 0:μ_s:t_final # system time
tspan = (0.0, t_final)

# Other input constants
m = 1 # mass (kg)
g = 9.81 # gravity (m/s^2)
L = 1 # length (m)
b = 0.2 # damping value (kg/m^2-s)

#Parameter vector
p = [m, g, L, b]

# initial angle (rad) is drawn uniformly in this range
θ₀_range = (π/180)*[30, 90]

# same for initial angular velocity (rad/s)
ω₀_range = [0, 1]

#-------------------------------------------------------------------------------------------
# Machine parameters

sampling = 3 # machine retain 1 in every so many samples

history = 5 # backward trajectory [seconds]
future = 5 # forward trajectory [seconds]

scale = 1 #bandwidth, i.e. Kernel Scale

μ_m = sampling*μ_s # machine time step
npast = convert(Int, round(history/μ_m)) #Past Series sample size
nfuture = convert(Int, round(future/μ_m)) #Future series sample size

window_size = npast + nfuture

tₘ= 0:μ_m:t_final # machine time

#-------------------------------------------------------------------------------------------
# Training Data: Damped Pendulum Dynamics - time simulation

u0 = [rand(Uniform(θ₀_range[1], θ₀_range[2])), rand(Uniform(ω₀_range[1], ω₀_range[2]))] # initial conditions
u0 = [π/4, 1.0]

prob_train = ODEProblem(Pendulum!, u0, tspan, p)

sol_train = solve(prob_train, saveat = μ_s, wrap = Val(false))
u_train = reduce(hcat, sol_train.u)

#-------------------------------------------------------------------------------------------
# Validation Data: Damped Pendulum Dynamics - time simulation

u0 = [rand(Uniform(θ₀_range[1], θ₀_range[2])), rand(Uniform(ω₀_range[1], ω₀_range[2]))] # initial conditions
u0 = [π/8, 1.0]

prob_val = ODEProblem(Pendulum!, u0, tspan, p)

sol_val = solve(prob_val, saveat = μ_s, wrap = Val(false))
u_val = reduce(hcat, sol_val.u)

#-------------------------------------------------------------------------------------------
# Machine Data Perspective

# ******************************************************************************************
# Training

xₘ_train = sol_train(tₘ) # machine samples of the continuous system
uₘ_train = reduce(hcat, xₘ_train.u)

# positions - no velocity, not a 1 to 1 mapping of state space
# => require history for reconstruction
x_train = L * sin.(uₘ_train[1,:])
y_train = L .- L.*cos.(uₘ_train[1,:])

data_train = [vec(x_train), vec(y_train)]

scale = [maximum(data_train[1]) - minimum(data_train[1]);
        maximum(data_train[2]) - minimum(data_train[2])]

scale = 1

# ******************************************************************************************
# Validation

xₘ_val = sol_val(tₘ) # machine samples of the continuous system
uₘ_val = reduce(hcat, xₘ_val.u)

# positions - no velocity, not a 1 to 1 mapping of state space
# => require history for reconstruction
x_val = L * sin.(uₘ_val[1,:])
y_val = L .- L.*cos.(uₘ_val[1,:])

data_val = [vec(x_val), vec(y_val)]

N = length(data_train[1])# number of samples

#-------------------------------------------------------------------------------------------
# Emergence - Pattern Discovery

# ******************************************************************************************
# Training

println("\nA. Training")
@time begin

    @time begin
        println("\n1. Generating gram matrices")
        Gx, Gy, index_map = series_Gxy(data_train[1], scale, npast, nfuture)
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
        # This is the forward operator in state space. It is built from consecutive
        # indices in the index map. Data series formed by multiple contiguous time
        # blocks are supported, as well as the handling of NaN values
        println("\n4. Forward Shift Operator")
        shift_op = shift_operator(coords_train, alg = :pinv)
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
    ic = [data_val[1][1:npast_val], data_val[2][1:npast_val]] # initial condition

    # step 1. Build a kernel similarity vector with sample data
    Kx = series_newKx(ic[1], data_train[1], index_map, scale, npast_val)
    # step 2. Embed to get similarity vector in state space
    Ks = embed_Kx(Kx, Gx, embedder)
    # step 3. Build a probability distribution over states.
    coords_ic = new_coords(Ks, Gs, coords_train)

    pred_hor = N - npast_val # prediction horizon

    pred, coords_val = predict(pred_hor, coords_ic[1, :], shift_op, expect_op, return_dist = 2)

end

#-------------------------------------------------------------------------------------------
# Plots
nans = Array{Float64, 1}(undef, N - pred_hor)
nans = fill!(nans, NaN)

x_pred = vec([nans; pred[1]])
y_pred = vec([nans; pred[2]])

df_x_y_t = DataFrame(x = data_train[1], y = data_train[2],
                    x̂ = data_val[1], ŷ = data_val[2],
                    x_pred = x_pred, y_pred = y_pred,
                    t = tₘ)

trace_x = scatter(df_x_y_t, x = :t, y = :x, name = "train-x")
trace_y = scatter(df_x_y_t, x = :t, y = :y, name = "train-y")
trace_x̂ = scatter(df_x_y_t, x = :t, y = :x̂, name = "val-x")
trace_ŷ = scatter(df_x_y_t, x = :t, y = :ŷ, name = "val-y")
trace_xp = scatter(df_x_y_t, x = :t, y = :x_pred, name = "pred-x")
trace_yp = scatter(df_x_y_t, x = :t, y = :y_pred, name = "pred-y")

plot_x_t = plot([trace_x, trace_y,
                trace_x̂, trace_ŷ,
                trace_xp, trace_yp],
                Layout(
                    title = attr(
                        text = "Damped Pendulum: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "y,x [m]",
                    ),
                )
display(plot_x_t)

Ψ₁ = vec([coords_train[:,2]; nans[1:end]])
Ψ₂ = vec([coords_train[:,3]; nans[1:end]])
#Φ₁ = coords_ic[:,2]
#Φ₂ = coords_ic[:,3]
γ₁ = vec([coords_val[:,2]; nans[1]])
γ₂ = vec([coords_val[:,3]; nans[1]])

RSS = DataFrame(Ψ₁ = Ψ₁,
                Ψ₂ = Ψ₂,
                #Φ₁ = Φ₁,
                #Φ₂ = Φ₂,
                γ₁ = γ₁,
                γ₂ = γ₂)

trace_Ψ = scatter(RSS, x = :Ψ₁, y = :Ψ₂, name = "train")
trace_Φ = scatter(RSS, x = :Φ₁, y = :Φ₂, name = "ic")
trace_γ = scatter(RSS, x = :γ₁, y = :γ₂, name = "val")

plot_RSS = plot([trace_Ψ, trace_γ],
                Layout(
                    title = attr(
                        text = "Reconstructed State Space",
                    ),
                    title_x = 0.5,
                    xaxis_title = "Ψ₁",
                    yaxis_title = "Ψ₂",),
                )

display(plot_RSS)

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
