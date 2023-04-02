using Complexity
using LinearAlgebra

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

alg = :hankel
svd_r = nothing
hankel_rank = 900

#coords = coords_train[1:35,:]
coords = [x_train, y_train]
coords = transpose(reduce(hcat, coords))

hankel_dims = size(coords, 1)

dims = size(coords, 1)
num_coords = size(coords, 2)

if index_map === nothing

    valid_prev = range(1, num_coords - 1)
    indices_no_next = [num_coords]
else

    # build reverse operators
    valid_prev = deleteat!(collect(range(1, num_coords - 1)), findall(diff(index_map) .!= 1))
    # also known as gaplocs, plus the latest state
    indices_no_next = setdiff(range(1, num_coords), valid_prev)
end

# To replace in formulas:
valid_next = valid_prev .+ 1

shift_op = zeros(dims, dims)
shift_op[1, 1] = 1.0

if alg == :dmd # exact dmd

    # Step 1: Compute the singular value decomposition of coords
    U, S, V = svd(coords[:, valid_prev])
    #coords[:, valid_prev] = U*Diagonal(S)*V'

    # Step 2: Truncate matrices
    if svd_r > dims svd_r = dims end

    if (svd_r > 0 && svd_r <= 1)

        # Find r to capture svd_r% of the energy
        cdS = cumsum(S)./sum(S)
        mean = sum(cdS)/length(S)
        r = findfirst(cdS .>= svd_r) # truncation rank

    else

        r = length(S)
    end

    Ur = U[:, 1:r]
    Sr = Diagonal(S[1:r])
    Vr = V[:, 1:r]

    # Step 3: Project A onto the POD modes of U
    U_map = (coords[:, valid_next])*Vr*inv(Sr)

    # Atilde - tells us how the POD modes evolve in time
    # the least square fit matrix in the low rank subspace
    shift_op = (Ur')*U_map

elseif alg == :hankel

    wind = (num_coords - hankel_rank)

    UH = Array{Float64, 2}(undef, hankel_dims*hankel_rank, wind) # Hankel matrix
    H = Array{Float64, 2}(undef, hankel_dims*hankel_rank, wind) # Hankel matrix

    # Idea: if using diffusion map coordinates, let's throw away the first coordinate
    # cause its always 1

    global j = 1
    for i in 1:hankel_rank

        global j

        UH[j:j + hankel_dims - 1, :] = coords[1:hankel_dims, i+1:wind + i]

        global j = j + hankel_dims

    end

    j = 1
    for i in 1:hankel_rank

        global j

        H[j:j + hankel_dims - 1, :] = coords[1:hankel_dims, i:wind + i - 1]

        j = j + hankel_dims

    end

    # Step 1: Compute the singular value decomposition of H

    U, S, V = svd(H)
    #coords[:, valid_prev] = U*diagm(S)*V'

    # Step 2: Truncate matrices
    if isnothing(svd_r)

        r = length(S)

    elseif svd_r > dims

        svd_r = dims

    elseif (svd_r > 0 && svd_r <= 1)

        # Find r to capture svd_r% of the energy
        cdS = cumsum(S)./sum(S)
        mean = sum(cdS)/length(S)
        r = findfirst(cdS .>= svd_r) # truncation rank

    end

    Ur = U[:, 1:r]
    Sr = Diagonal(S[1:r])
    Vr = V[:, 1:r]

    # Step 3: Project A onto the POD modes of U

    U_map = H*Vr*inv(Sr)

    shift_op = (Ur')*U_map

end

eigval, eigvec_r = eigen(shift_op)

if alg == :dmd || alg == :hankel

    # Step 4: The DMD modes are eigenvectors of the high-dimensional A matrix. The DMD
    # eigenvalues are also the eigenvalues of the full shift operator.

    Phi = U_map*eigvec_r
    lambda = Diagonal(eigval) #discrete time

    #omega = log.(eigval)/timestep # continuous time DMD eigenvalues

    #x1 = coords[:,1]
    #b = Phi\x1 # tells you how much of each mode is going at each time

    alpha1 = Sr*vec(Vr[1,:]')
    b = (eigvec_r*lambda)\alpha1 # mode amplitude

    # A Φ = Λ Φ # the least square fit matrix
    #shift_op = real.(Phi*lambda*pinv(Phi))

end

#DMD = (Phi, eigval, b) # DMD eigenvectors, eigenvalues, modes
#SingleValueDecomposition = (U, S, V)

# ******************************************************************************************
# Koopman Modes (Phi)

tr = 6

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
relayout!(plot_ϕ, title_text = "DMD eigenvectors")

# ******************************************************************************************
# Single Value Decomposition

tr = 10
Σ = (S[1:tr]./(sum(S)))

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

Uo = real(U[:, 1:3])

plot_U = plot(Uo,
                Layout(
                    title = attr(
                        text = "Left Singular Vectors", # SVD modes
                    ),
                    title_x = 0.5,
                    yaxis_title = "U",
                    ),
                )

#display(plot_U)

Vo = real(V[:, 1:3])

plot_V = plot(Vo,
                Layout(
                    title = attr(
                        text = "Right Singular Vectors",
                    ),
                    title_x = 0.5,
                    yaxis_title = "V",
                    ),
                )

#display(plot_V)

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
