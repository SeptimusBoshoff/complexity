#= Authors:
    Source code adapted from the Continuous Causal States method, described in the paper :
    Discovering Causal Structure with Reproducing-Kernel Hilbert Space ε-Machines by Nicolas
    Brodu and James P. Crutchfield
    Original code in python converted to julia code by Septimus Boshoff
=#

"""
    eigenvalues, basis = eigvec_l, coords (or eigvec_r) = spectral_basis(Gs; num_basis = nothing, scaled = true, alpha = 1)


# Description
Performs an eigenvalue spectral decomposition of the similarity matrix, after normalising
the matrix such that it becomes a Markov chain. This uses a variant of the diffusion map
algorithm, applied directly to the provided similarity matrix, which replaces the
"diffusion" step of the diffusion maps. The decay of the spectrum is a measure of the
connectivity of data points in the RKHS. Together, the eigenvalues and eigenvectors provide
a basis for the data set, a reduced embedding in Euclidian space. The distances in this
space, the diffusion space, are so called 'diffusion distances', if Gs were constructed to
be a 'diffusion matrix', which it mostly is - but with extra steps.

# Arguments
- `Gs::Array{Float64, 2}`: A 'N x N' precomputed similarity matrix between every causal
  state. Entries Gs[i, j] can be seen as inner products between states Sᵢ and Sⱼ.

# Keyword Arguments
- `num_basis::Int or Float`: Defines the dimensionality, 'D', of the diffusion coordinates.
    - If 'nothing' then 'D = N/2' eigenvalues are retained. A somewhat arbitrary number,
      which should be verified to be adequate.
    - If '<= 0', then all eigenvalues are retained. This is generally ill advised. Due to
      previous steps (embed_states) a spurious eigenvalues may be artificially created,
      which impacts the results of future steps (shift_operator).
    - If '>= 1', the specifies the number of basis components.
    - If '0 < num_basis < 1', specifies a threshold below which eigenvalues are discarded.
- `scaled::Bool`: Determines whether the right eigenvectors are scaled. If "true" then every
  column is a coordinate in diffusion space. If "false", then every _row_ is a right
  eigenvector (going against convention).
- `α::Float`: An optional normalization exponent between '0' and '1' for the diffusion-map
  like algorithm. The dafault of '1' is the Laplace-Beltrami approximation where the Markov
  chain converges to Brownian motion, allowing for the inference of geometry, with minimal
  interference from the density. In other words, the distribution of the data is separated
  from the geometry of the underlying manifold, i.e. the geometry and the statistics have
  been completely decoupled. In contrast, when alpha is set to '0' we obtain the graph
  Laplacian.

# Return Values
- `eigenvalues::Vector{Float64}`: A vector whose size is dependent on 'num_basis'. Due to
  normalisation, and the fact that we are dealing with stochastic Markov transition
  matrices, the first eigenvalue is always '1'. All other eigenvalues are listed in
  decending order and should be positive.
- `basis::Array{Float64, 2}`: An 'D x N' matrix where every row corresponds to a basis, a
  left eigenvector. 'M' is determined by 'num_basis'. The first row corresponds to the
  stationary distribution and is normalized to sum to '1'. All other rows sum up to '0'.
- `coords (or eigenvectors)::Array{Float64, 2}`: A 'D x N' matrix where every column is a
  difffusion space coordinate, i.e. a point in diffusion space. The first coordinate of
  every point is normalised such that it is always 1, as it ideally should be for a
  stochastic Markov transition matrix, and can be discarded for purposes such as distance
  comparisons, manifold reconstruction, visualisation, etc. If unscaled, then every row is a
  right eigenvector. Together the right eigenvectors and left eigenvectors form a
  bi-orthogonal basis such that "transpose(eigvec_r[i, :]) * eigvec_l[i, :] = 1", or
  equivalently "eigvec_l * transpose(eigvec_r) = Identity"

"""
function spectral_basis(Gs; num_basis = nothing, scaled = true, α = 1.0, return_eigendecomp = false)

    N = size(Gs, 1)

    eigen_cutoff = -1

    if num_basis === nothing

        num_basis = N ÷ 2

    elseif num_basis <= 0

        num_basis = N

    elseif num_basis >= 1

        num_basis = convert(Int64, num_basis)

    else
        eigen_cutoff = num_basis
        num_basis = N
    end

    α = clamp(α, 0., 1.)

    Gs_c = copy(Gs) # we don't want to overwrite the similarity matrix

    if α > 0

        q = vec(sum(Gs_c, dims = 2))

        if α != 1

            q = q.^α
        end

        q = (1) ./ q

        Gs_c = Gs_c .* q
        Gs_c = Gs_c .* transpose(q)

    end

    q = vec(sum(Gs_c, dims = 2))

    P = Diagonal((1) ./ q)*Gs_c # stochastic Markov matrix

    # using ArnoldiMethodTransformations for generalised eigen, mat is symmetric, P is not
    decomp, history  = partialschur(P, nev = num_basis,
    tol = 1e-6, restarts = 200, which = LM())
    eigval, eigvec = partialeigen(decomp)

    #order = sortperm(real.(eigval), rev = true)
    eigval = real.(eigval[end:-1:1])
    eigvec = real.(eigvec[:, end:-1:1])

    if eigen_cutoff >= 0

        num_basis = sum(eigval .>= eigen_cutoff)
    end

    if num_basis < size(eigval, 1)

        eigval = eigval[1:num_basis]
        eigvec = eigvec[:, 1:num_basis]
    end

    # Normalization so that eigvec_r[:,1] entries are all 1
    # and that eigvec_l[:,1] matches a density
    # not really normalisation? maybe removing floating point errors?

    eigvec_r = eigvec ./ eigvec[:, 1]
    eigvec_l = transpose(eigvec .* (eigvec[:,1]))

    if !history.converged

        @warn(history)

        check = maximum(vec(sum(Diagonal((1) ./ q)*Gs_c, dims = 1))) # should be close to 1
    end

    if (norm(Gs_c * eigvec_r - Diagonal(q)*eigvec_r * Diagonal(eigval)) > 1e-6 ||
        norm(eigvec_l*Gs_c - Diagonal(eigval)*eigvec_l*Diagonal(q)) > 1e-6 ||
        maximum(abs.(sum(eigvec_l[2:end, :], dims = 2))) > 4e-4)

        @warn("The eigenvectors weren't correctly computed.")

        @show norm(Gs_c * eigvec_r - Diagonal(q)*eigvec_r * Diagonal(eigval))
        #@show norm(Diagonal((1) ./ q)*Gs_c * eigvec_r - eigvec_r * Diagonal(eigval))

        @show norm(eigvec_l*Gs_c - Diagonal(eigval)*eigvec_l*Diagonal(q))
        #@show norm(eigvec_l*Diagonal((1) ./ q)*Gs_c - Diagonal(eigval)*eigvec_l)

        @show maximum(abs.(sum(eigvec_l[2:end, :], dims = 2)))

    end

    if (sum(eigvec_l[1,:]) - 1 > 1e-6 ||
        norm((q/sum(q)) .- eigvec_l[1,:]) > 1.5e-5)

        @warn("The stationary distribution was not correctly calculated")
        @show sum(eigvec_l[1,:]) - 1
        @show norm((q/sum(q)) .- eigvec_l[1,:])

    end

    if return_eigendecomp

        if scaled
            return eigval, eigvec_l, transpose(transpose(eigval).*eigvec_r)
        else
            return eigval, eigvec_l, transpose(eigvec_r)
        end

    else

        if scaled
            return transpose(transpose(eigval).*eigvec_r)
        else
            return transpose(eigvec_r)
        end
    end
end
