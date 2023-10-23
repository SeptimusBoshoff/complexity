using Complexity
using Distributions
using DifferentialEquations
using PlotlyJS
using TimerOutputs
using LinearAlgebra
using ArnoldiMethod
using NearestNeighbors

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........\n")

to = TimerOutput()

#-------------------------------------------------------------------------------------------
# User input

Δt_s = 0.005 # numerical time step size for solver (seconds)

t_final_train = 200 # max simulation time (seconds)

subsampling = 5 # retain 1 in every so many samples

history = 5 # backward trajectory [seconds]
future = 5 # forward trajectory [seconds]

x_dims = [40, 50] # PCA past dimensionality reduction
y_dims = [40, 50] # PCA future dimensionality reduction

y_pred = [1, 1] # future prediction horizon

γ = [0.04, 0.025] # PGSO precision parameter
Mmax = [Inf, Inf] # maximum number of centres

k_frac = 9 # the fraction of the sample size used to compute the kernel std - inverse relationship
# option 1 = 5

noise_std = 0.1 # standard deviation of added noise (when gaussian)

#= Options: dynamical systems
    1. Van der Pol
    2. Delayed Van der Pol
    2. Lorenz
=#

option = 2

# ******************************************************************************************
# Data generation

println("Data generation")
@timeit to "Data generation" begin

    Δt_m = subsampling*Δt_s # machine time step
    npast = convert(Int, round(history/Δt_m)) #Past Series sample size
    nfuture = convert(Int, round(future/Δt_m)) #Future series sample size

    window_size = npast + nfuture

    y_pred = clamp.(y_pred, 1, nfuture)

    #-------------------------------------------------------------------------------------------
    # Training Data - time simulation

    t_m= 0:Δt_m:t_final_train # time coordinates
    tspan = (0.0, t_final_train)

    if option == 1

        # system parameters
        μ = 2

        #Parameter vector
        p = [μ]

        # training data
        u0 = [1.012, -0.88]
        prob_train = ODEProblem(Van_der_Pol!, u0, tspan, p)

        # validation data
        u0 = [1.83, -0.36]
        prob_val = ODEProblem(Van_der_Pol!, u0, tspan, p)

        # algorithm
        alg = BS3()

    elseif option == 2

        h(p, t) = [-1.1816783878162982; 1.490038688253868]

        tau = 2
        lags = [tau]

        # system parameters
        μ = 4
        v0 = 1
        v1 = 1
        beta0 = 2
        beta1 = 2

        #Parameter vector
        p = (μ, v0, v1, beta0, beta1, tau)

        # training data
        u0 = [1.9863606839964665, 0.12366121676846706]
        prob_train = DDEProblem(Delayed_Van_der_Pol!, u0, h, tspan, p; constant_lags = lags)

        # validation data
        u0 = [1.9863606839964665, 0.12366121676846706]
        prob_val = DDEProblem(Delayed_Van_der_Pol!, u0, h, tspan, p; constant_lags = lags)

        # algorithm
        alg = MethodOfSteps(Vern6())

    elseif option == 3

        # system paramaters
        σ = 10 # Prandtl
        ρ = 28 # Raleigh
        β = 8/3 # geometric aspect ratio

        #Parameter vector
        p = [σ, ρ, β]

        # training data
        u0 = [9.9116, 13.4339, 23.8203]
        prob_train = ODEProblem(Lorenz!, u0, tspan, p)

        # validation data
        u0 = [-0.4214317138514596, 1.3330311718739656, 22.192285595403675]
        prob_val = ODEProblem(Lorenz!, u0, tspan, p)

        alg = Tsit5()

    end

    # ******************************************************************************************
    # Training and Validation data

    sol_train = solve(prob_train, alg, saveat = Δt_s, wrap = Val(false))
    sol_val = solve(prob_val, alg, saveat = Δt_s, wrap = Val(false))

    data_train = reduce(hcat, sol_train(t_m).u)
    data_val = reduce(hcat, sol_val(t_m).u)

    data_train = data_train[:,1000:end]
    data_val = data_val[:,1000:end]
    t_m = t_m[1000:end]

    dimX = size(data_train, 1) # dimension of state space
    m = size(data_train, 2) # number of data points
    N = m - window_size + 1 # number of data snapshots, i.e., vectors

    data_train = data_train .+ noise_std*randn(dimX, m)

end

# ******************************************************************************************
# Hankel matrices

println("Hankel matrices")
@timeit to "Hankel matrices" begin
    #-------------------------------------------------------------------------------------------
    # Training

    Sx = Vector{Matrix{Float64}}(undef, dimX) # Past Hankel matrix
    Sy = Vector{Matrix{Float64}}(undef, dimX) # Future Hankel matrix

    for d in 1:dimX

        Sx[d] = Matrix{Float64}(undef, npast, N)
        Sy[d] = Matrix{Float64}(undef, nfuture, N)

        for k in 1:N

            Sx[d][:, k] = data_train[d, npast + k - 1:-1:k]
            Sy[d][:, k] = data_train[d, k + npast:k + npast + nfuture - 1]
        end
    end

    #-------------------------------------------------------------------------------------------
    # Validation

    Sx_val = Vector{Matrix{Float64}}(undef, dimX) # Past Hankel matrix
    Sy_val = Vector{Matrix{Float64}}(undef, dimX) # Future Hankel matrix

    for d in 1:dimX

        Sx_val[d] = Matrix{Float64}(undef, npast, N)
        Sy_val[d] = Matrix{Float64}(undef, nfuture, N)

        for k in 1:N

            Sx_val[d][:, k] = data_val[d, npast + k - 1:-1:k]
            Sy_val[d][:, k] = data_val[d, k + npast:k + npast + nfuture - 1]

        end
    end
end

# ******************************************************************************************
# Kernel bandwidths

println("Kernel bandwidths")
@timeit to "Kernel bandwidths" begin

    #---------------------------------------------------------------------------------------
    # Kernel bandwidths - the standard deviation defines the width of the gaussian

    ηx = Array{Float64, 1}(undef, dimX)
    ηy = Array{Float64, 1}(undef, dimX)

    Sx_Kstd = Array{Float64, 1}(undef, dimX)
    Sy_Kstd = Array{Float64, 1}(undef, dimX)

    K_NN = convert(Int, round(N/k_frac)) # the fraction of the sample size

    Threads.@threads for d in 1:dimX

        Sx_Kstd[d], _ = Mean_Distance(Sx[d], KNN = true, K = K_NN)
        Sy_Kstd[d], _ = Mean_Distance(Sy[d], KNN = true, K = K_NN)

        ηx[d] = 1/(2*(Sx_Kstd[d]^2)) # inverse kernel bandwidth, scale: avg_dist = std
        ηy[d] = 1/(2*(Sy_Kstd[d]^2)) # inverse kernel bandwidth, scale: avg_dist = std

    end

    #= if option == 1 # van der pol

        #= Sx_avg_dist = Mean_Distance(Sx[1,:,:]) / 2 # standard deviation
        Sy_avg_dist = Mean_Distance(Sy[1,:,:]) / 2 # standard deviation =#

        ηx = 0.007834085196596792
        ηy = 0.012323447063869173

    elseif option == 2 # Lorenz

        ηx = 8.364928561682756e-5
        ηy = 8.283178640633994e-5
    end =#

    println("\tPast kernel width / standard deviation: ", round.(Sx_Kstd, digits = 3))
    println("\tFuture kernel width / standard deviation: ", round.(Sy_Kstd, digits = 3))

end

# ******************************************************************************************
# Gramian matrices

println("Gramian matrices")
@timeit to "Gramian matrices" begin

    Gxx = Vector{Matrix{Float64}}(undef, dimX)
    Gyy = Vector{Matrix{Float64}}(undef, dimX)

    for d in 1:dimX

        Gxx[d] = Gramian(Sx[d], ηx[d])
        Gyy[d] = Gramian(Sy[d], ηy[d])
    end
end

# ******************************************************************************************
# Partial Gram-Schmidt Orthogonalisation

println("Partial Gram-Schmidt Orthogonalisation")
@timeit to "Partial Gram-Schmidt Orthogonalisation" begin

    Rxx = Vector{Matrix{Float64}}(undef, dimX)
    Ryy = Vector{Matrix{Float64}}(undef, dimX)

    Idx = Vector{Vector{Int64}}(undef, dimX)
    Idy = Vector{Vector{Int64}}(undef, dimX)

    sclx = Vector{Vector{Float64}}(undef, dimX)
    scly = Vector{Vector{Float64}}(undef, dimX)

    Cx = Vector{Int64}(undef, dimX)
    Cy = Vector{Int64}(undef, dimX)

    for d in 1:dimX

        Rxx[d], Idx[d], sclx[d], errorx = ichol(Gxx[d], γ[d], Mmax = Mmax[d])
        Ryy[d], Idy[d], scly[d], errory = ichol(Gyy[d], γ[d], Mmax = Mmax[d])

        Cx[d] = length(Idx[d])
        Cy[d] = length(Idy[d])

        println("\tcoordinate: ", d)
        println("\treconstruction error - Rxx: ", round(errorx*1e3, digits = 3), "e-3")
        println("\treconstruction error - Ryy: ", round(errory*1e3, digits = 3), "e-3")
        println("\tcentres x: ", Cx[d])
        println("\tcentres y: ", Cy[d])

    end
end

# ******************************************************************************************
# Kernel PCA: Dimensionality reduction, Whitening - decorrelation

println("Kernel PCA")
@timeit to "Kernel PCA" begin

    One_N = (1/N)*ones(N,N)
    H = I - One_N

    # centred Gram matrices
    Gxx_c = Vector{Matrix{Float64}}(undef, dimX)
    Gyy_c = Vector{Matrix{Float64}}(undef, dimX)

    # centred PGSO matrices
    Rxx_c = Vector{Matrix{Float64}}(undef, dimX)
    Ryy_c = Vector{Matrix{Float64}}(undef, dimX)

    # principal component variances
    Λx = Vector{Vector{Float64}}(undef, dimX)
    Λy = Vector{Vector{Float64}}(undef, dimX)

    # principal directions coefficients
    Vx = Vector{Matrix{Float64}}(undef, dimX)
    Vy = Vector{Matrix{Float64}}(undef, dimX)

    # principal component coefficients
    Gxx_pc = Vector{Matrix{Float64}}(undef, dimX)
    Gyy_pc = Vector{Matrix{Float64}}(undef, dimX)

    for d in 1:dimX

        #-----------------------------------------------------------------------------------
        # Past Hankels

        # Centring

        Rxx_c[d] = H*Rxx[d]
        Gxx_c[d] = Rxx_c[d]*transpose(Rxx_c[d])

        # Kernel PCA

        Vx[d], Λx[d], _ = svd(Rxx_c[d])
        Λx[d] = (Λx[d].^2)./(N-1) # principal component variances

        # truncate

        Λx[d] = Λx[d][1:x_dims[d]]
        Vx[d] = Vx[d][:, 1:x_dims[d]]

        # Normalising: transpose(Vx[1])*Gxx_c[1]*Vx[1] ≈ (N-1)*Diagonal(Λx[1])*transpose(Vx[1])*Vx[1] ≈ I

        normsx = sqrt.(sum((Vx[d].^2)*(N-1)*Diagonal(Λx[d]), dims = 1))
        Vx[d] = Vx[d] ./ (normsx) # the coefficients for the principal directions

        # principal components

        Gxx_pc[d] = transpose(Vx[d])*Gxx_c[d]

        #-----------------------------------------------------------------------------------
        # Future Hankel

        # Centring

        Ryy_c[d] = H*Ryy[d]
        Gyy_c[d] = Ryy_c[d]*transpose(Ryy_c[d])

        # Kernel PCA

        Vy[d], Λy[d], _ = svd(Ryy_c[d])
        Λy[d] = (Λy[d].^2)./(N-1) # principal component variances

        # truncate

        Λy[d] = Λy[d][1:y_dims[d]]
        Vy[d] = Vy[d][:, 1:y_dims[d]]

        # Normalising: transpose(Vy[e])*Gyy_c[e]*Vy[e] ≈ (N-1)*Diagonal(Λy)*transpose(Vy)*Vy ≈ I

        normsy = sqrt.(sum((Vy[d].^2)*(N-1)*Diagonal(Λy[d]), dims = 1))
        Vy[d] = Vy[d] ./ (normsy) # the coefficients for the principal directions

        # principal components

        Gyy_pc[d] = transpose(Vy[d])*Gyy_c[d] # the principal components coefficients
    end
end

# ******************************************************************************************
# Kernel Canonical Correlation Analysis

println("Kernel CCA")
@timeit to "Kernel CCA" begin

    Cxx_pc = Matrix{Matrix{Float64}}(undef, dimX, dimX)
    Cyy_pc = Matrix{Matrix{Float64}}(undef, dimX, dimX)
    Cxy_pc = Matrix{Matrix{Float64}}(undef, dimX, dimX)
    Cyx_pc = Matrix{Matrix{Float64}}(undef, dimX, dimX)

    for d1 in 1:dimX

        for d2 in 1:dimX

            if d1 == d2

                # Covariance matrices

                Cxx_pc[d1, d2] = Diagonal(Λx[d1])
                Cyy_pc[d1, d2] = Diagonal(Λy[d1])

            else

                # Cross-covariance matrices

                Cxx_pc[d1, d2] = (1/(N-1))*Gxx_pc[d1]*transpose(Gxx_pc[d2])
                Cyy_pc[d1, d2] = (1/(N-1))*Gyy_pc[d1]*transpose(Gyy_pc[d2])
            end

            # Cross-covariance matrices

            Cxy_pc[d1, d2] = (1/(N-1))*Gxx_pc[d1]*transpose(Gyy_pc[d2])
            Cyx_pc[d1, d2] = (1/(N-1))*Gyy_pc[d1]*transpose(Gxx_pc[d2])

        end
    end

    #---------------------------------------------------------------------------------------

    CXX_pc = reduce(vcat, Cxx_pc[:,1])
    CYY_pc = reduce(vcat, Cyy_pc[:,1])
    CXY_pc = reduce(vcat, Cxy_pc[:,1])
    CYX_pc = reduce(vcat, Cyx_pc[:,1])

    if dimX > 1

        for d in 2:dimX

            global CXX_pc, CYY_pc, CXY_pc, CYX_pc

            CXX_pc = hcat(CXX_pc, reduce(vcat, Cxx_pc[:,d]))
            CYY_pc = hcat(CYY_pc, reduce(vcat, Cyy_pc[:,d]))
            CXY_pc = hcat(CXY_pc, reduce(vcat, Cxy_pc[:,d]))
            CYX_pc = hcat(CYX_pc, reduce(vcat, Cyx_pc[:,d]))
        end
    end

    M, Σ, V = svd(CXX_pc)
    CXX_pc_05_inv = M*Diagonal(Σ.^(-0.5))*adjoint(V)

    M, Σ, V = svd(CYY_pc)
    CYY_pc_05_inv = M*Diagonal(Σ.^(-0.5))*adjoint(V)

    M, D, V = svd(CXX_pc_05_inv*CXY_pc*CYY_pc_05_inv)

    WX = CXX_pc_05_inv*M # coefficients of left singular functions - X position vector
    WY = CYY_pc_05_inv*V # coefficients of right singular functions - Y position vector

    ZX = transpose(WX)*reduce(vcat, Gxx_pc) # canonical variables (scores or components)
    ZY = transpose(WY)*reduce(vcat, Gyy_pc) # canonical variables (scores or components)

    #verify:
    CXX_z = (1/(N - 1))*ZX*transpose(ZX) # round.(CXX_z, digits = 5) ≈ I
    CYY_z = (1/(N - 1))*ZY*transpose(ZY) # round.(CYY_z, digits = 5) ≈ I
    CXY_z = (1/(N - 1))*ZX*transpose(ZY) # diag(CXY_z) ≈ D

    #---------------------------------------------------------------------------------------
    # Prediction matrix

    Uᵀ = CYY_pc*WY*Diagonal(D)*transpose(WX)

end

# ******************************************************************************************
# Conformal map - preimage

println("Conformal map")
@timeit to "Conformal map" begin

    M = Vector{Matrix{Float64}}(undef, dimX)

    ε = 1e-9 # ridge regression regularizer

    for d in 1:dimX
        M[d] = pinv(transpose(Sy[d][1:y_pred[d],:]))*(transpose(Sy[d][1:y_pred[d],:])*Sy[d][1:y_pred[d],:] - ε*I)*pinv(Gyy_pc[d])

        # idea the pinv(Gyy) can be done faster given the already performed decomposition
    end
end

# ******************************************************************************************
# Prediction

println("Prediction")
@timeit to "Prediction" begin

    Kx = Vector{Vector{Float64}}(undef, dimX) # centred feature vectors
    rx = Vector{Vector{Float64}}(undef, dimX) # pgso feature vectors
    Gxx_One_Nd = Vector{Vector{Float64}}(undef, dimX) # centring vector

    Dx = Vector{Vector{Float64}}(undef, dimX)  # initial & running condition
    Dy = Vector{Matrix{Float64}}(undef, dimX) # preimage predictions

    VxᵀH = Vector{Matrix{Float64}}(undef, dimX)

    for d in 1:dimX

        Kx[d] = Array{Float64,1}(undef, N)
        rx[d] = Array{Float64,1}(undef, Cx[d])

        Gxx_One_Nd[d] = Gxx[d]*(1/N)*ones(N) # centring operations

        Dx[d] = Sx_val[d][ :, 1]

        Dy[d] = Array{Float64,2}(undef, y_pred[d], N) # preimage predictions # nfuture

        VxᵀH[d] = transpose(Vx[d])*H

    end

    MD = [M[1] zeros(y_pred[1], y_dims[2]);zeros(y_pred[2], y_dims[1]) M[2]]
    VH = [VxᵀH[1] zeros(x_dims[1], N); zeros(x_dims[2], N) VxᵀH[2]]

    Predict = MD*Uᵀ*VH

    for k in 1:N

        for d in 1:dimX

            for j in 1:Cx[d]

                rx[d][j] = Gaussian_kernel(Sx[d][:, Idx[d][j]], Dx[d], ηx[d])
                rx[d][j] = (rx[d][j] - transpose(rx[d][1:j-1])*Rxx[d][Idx[d][j],1:j-1])/sclx[d][j]
            end

            Kx[d] = Rxx[d]*rx[d] .- Gxx_One_Nd[d]
        end

        DY = Predict*[Kx[1]; Kx[2]]

        # preimages
        Dy[1][:, k] = DY[1:y_pred[1]]
        Dy[2][:, k] = DY[y_pred[1]+1:end]

        for d in 1:dimX

            # running condition
            Dx[d][2:end] = @views Dx[d][1:end-1]
            Dx[d][1] = Dy[d][1, k]
        end

    end
end

# ******************************************************************************************
# Plots

#-------------------------------------------------------------------------------------------
# Prediction Matrix

trace_Uᵀ = heatmap(x = 1:size(Uᵀ,2), y = size(Uᵀ,1):-1:1, z = Uᵀ, colorscale = "Electric")
layout = Layout(title = attr(
                text = "Prediction Matrix Uᵀ",
                ),
            title_x = 0.5,)
plot_Uᵀ = Plot(trace_Uᵀ, layout)

#-------------------------------------------------------------------------------------------
# X Past principal component variances

traces = [
            scatter(y = Λx[1], name = "Λx₁"),
            scatter(y = Λx[2], name = "Λx₂"),
            ]

plot_Λx = plot(traces,
                Layout(
                    title = attr(
                        text = "X Past principal component variances",
                        ),
                    title_x = 0.5,
                    xaxis_title = "Order",
                    yaxis_title = "Magnitude",
                    ),
                )

#-------------------------------------------------------------------------------------------
# Y Past principal component variances

traces = [
            scatter(y = Λy[1], name = "Λy₁"),
            scatter(y = Λy[2], name = "Λy₂"),
        ]

plot_Λy = plot(traces,
        Layout(
            title = attr(
                text = "Y Future principal component variances",
                ),
            title_x = 0.5,
            xaxis_title = "Order",
            yaxis_title = "Magnitude",
            ),
        )

#-------------------------------------------------------------------------------------------
# Canonical Scores (variables or components)

traces = [
            scatter(x = t_m, y = ZX[1,:], name = "Zx[1]₁"),
            scatter(x = t_m, y = ZX[2,:], name = "Zx[1]₂"),
            scatter(x = t_m, y = ZX[3,:], name = "Zx[1]₃"),
            scatter(x = t_m, y = ZY[1,:], name = "Zy[1]₁"),
            scatter(x = t_m, y = ZY[2,:], name = "Zy[1]₂"),
            scatter(x = t_m, y = ZY[3,:], name = "Zy[1]₃"),
            ]

plot_CCA_scores = plot(traces,
                Layout(
                    title = attr(
                        text = "Canonical Scores",
                        ),
                    title_x = 0.5,
                    ),
                )

#-------------------------------------------------------------------------------------------
# Canonical correlations

traces = [
            scatter(y = D, name = "ρ"),
            ]

plot_CCA_cor = plot(traces,
                Layout(
                    title = attr(
                        text = "Canonical Correlations",
                        ),
                    title_x = 0.5,
                    ),
                )

#-------------------------------------------------------------------------------------------
# Dynamical system time evolution

traces = [
            scatter(x = t_m, y = data_val[1,:], name = "x₁ - val"),
            scatter(x = t_m, y = data_val[2,:], name = "x₂ - val"),

            scatter(x = t_m, y = vcat(fill(NaN, nfuture), Dy[1][1,:]), name = "y₁(k+1)"),
            scatter(x = t_m, y = vcat(fill(NaN, nfuture), Dy[2][1,:]), name = "y₂(k+1)"),
            ]

plot_dyns = plot(traces,
                Layout(
                    title = attr(
                        text = "Dynamical System",
                        ),
                    title_x = 0.5,
                    xaxis_title = "time [s]",
                    yaxis_title = "Coordinates [m]",
                    ),
                )

#-------------------------------------------------------------------------------------------
# Dynamical system state space

if option == 1 || option == 2

    traces = [
                scatter(x = data_train[1,:], y = data_train[2,:], name = "train",
                        mode="markers",
                        marker=attr(
                        size = 3,),),
                scatter(x = data_val[1,:], y = data_val[2,:], name = "val", mode = "lines"),
                scatter(x = vcat(fill(NaN, nfuture), Dy[1][1,:]),
                        y = vcat(fill(NaN, nfuture), Dy[2][1,:]),
                        name = "pred", mode = "lines"),
                ]

    plot_SS = plot(traces,
                    Layout(
                        title = attr(
                            text = "State Space",
                        ),
                        title_x = 0.5,
                        xaxis_title = "x₁ [m]",
                        yaxis_title = "x₂ [m]"
                        ),
                    )

else

    traces = [scatter3d(x = data_train[1,:], y = data_train[2,:], z = data_train[3,:], name = "train", mode = "lines"),
                #scatter3d(SS, x = :x_train, y = :y_train, z = :z_train, name = "train", mode = "lines"),
                #scatter3d(SS, x = :x_pred, y = :y_pred, z = :z_pred, name = "pred", mode = "lines"),
                ]

    plot_SS_3d = plot(traces,
                    Layout(
                        title = attr(
                            text = "State Space",
                        ),
                        title_x = 0.5,
                        scene = attr(
                            xaxis_title = "x₁ [m]",
                            yaxis_title = "x₂ [m]",
                            zaxis_title = "x₃ [m]",
                        ),
                        scene_aspectratio = attr(x = 2, y = 2, z = 2),
                        scene_camera = attr(
                            up = attr(x = 1, y = 0, z = 0),
                            center = attr(x = 0, y = 0, z = 0),
                            eye = attr(x = 2, y = 2, z = 2)
                            ),
                        ),
                    )
end

# ******************************************************************************************
# Display

#-------------------------------------------------------------------------------------------
# Prediction
display(plot_Uᵀ)

#-------------------------------------------------------------------------------------------
# Kernel PCA

display(plot_Λx)
display(plot_Λy)
#display(plot_pred_pc)

#-------------------------------------------------------------------------------------------
# Kernel CCA

#display(plot_CCA_scores)
display(plot_CCA_cor)

#-------------------------------------------------------------------------------------------
# Dynamical system

display(plot_dyns)
display(plot_SS)
#display(plot_SS_3d)

# ******************************************************************************************

show(to)

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")