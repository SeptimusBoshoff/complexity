
"""
    eigenvalues, basis, coords = spectral_basis(Gs; num_basis = nothing, scaled = true, alpha = 1)

# Description
Eigenvalue spectral decomposition of the similarity matrix, after normalising the matrix
such that it becomes a Markov chain. This uses a variant of the diffusion map algorithm,
applied directly to the provided similarity matrix, which replaces the "diffusion" step of
the diffusion maps. The decay of the spectrum is a measure of the connectivity of data
points in the RKHS. Together, the eigenvalues and eigenvectors provide a basis for the data
set, a reduced embedding in Euclidian space. The distances in this space, the diffusion
space, are so called 'diffusion distances', if Gs were constructed to be a 'diffusion
matrix', which it mostly is - but with extra steps.

# Arguments
- `Gs::Array{Float64, 2}`: A 'N x N' precomputed similarity matrix between every causal
  state. Entries Gs[i,j] can be seen as inner products between states Sᵢ and Sⱼ.

# Keyword Arguments
- `num_basis::Int or Float`:
    - If 'nothing' then 'N/2' eigenvalues are retained. A somewhat arbitrary and should be
      verified that the value is adequate.
    - If '<= 0', then all eigenvalues are retained. This is generally ill advised. Due to
      previous steps (embed_states) a spurious eigenvalues may be artificially created,
      which impacts the results of future steps (shift_operator).
    - If '<= 1', the specifies the number of basis components.
    - If '0 < num_basis < 1', specifies a threshold below which eigenvalues are discarded.
- `scaled::Bool`: Determines whether the right eigenvectors are scaled or returned as is.
- `alpha::Float`: An optional normalization exponent between '0' and '1' for the
  diffusion-map like algorithm. The dafault of '1' is the Laplace-Beltrami approximation
  where the Markov chain converges to Brownian motion, allowing for the inference of
  geometry, with minimal interference from the density. In other words, the distribution of
  the data is separated from the geometry of the underlying manifold, i.e. the geometry and
  the statistics have been completely decoupled. In contrast, when alpha is set to '0' we
  obtain the graph Laplacian.

# Return Values
- `eigenvalues::Vector{Float64}`: A vector whose size is dependent on 'num_basis'. Due to
  normalisation the first eigenvalue is always '1'. All other eigenvalues are listed in
  decending order and should be positive.
- `basis::Array{Float64, 2}`: A 'M x N' matrix where every row corresponds to a basis, a
  left eigenvector. The first row corresponds to the stationary distribution and is
  normalized to sum to '1'. All other rows should sum up to '0'.
- `coords::Array{Float64, 2}`: A 'N x M' matrix where every column is either a right
  eigenvector or a right eigenvector scaled by its corresponding eigenvlaue. Every vector
  can be considered to be a point in the diffusion space, a diffusion coordinate so to
  speak. The first coordinate of every point is normalised such that it is always 1 and can
  be discarded for purposes such as distance comparisons, manifold reconstruction,
  visualisation, etc. Together the (unscaled) right eigenvectors and left eigenvectors form
 a bi-orthogonal basis such that "transpose(coords[:,x]) * basis[x,:] = 1", or equivalently
 "basis * coords = I"

"""
function spectral_basis(Gs; num_basis = nothing, scaled = true, α = 1.0)

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

    mat = copy(Gs)

    if α > 0

        q = vec(sum(mat, dims = 2))

        if α != 1

            q = q.^α
        end

        q = (1) ./ q

        mat = mat .* q
        mat = mat .* transpose(q)

    end

    q = vec(sum(mat, dims = 2))

    P = Diagonal((1) ./ q)*mat # stochastic Markov matrix

    # using ArnoldiMethodTransformations for generalised eigen, mat is symmetric, P is not
    decomp, history  = partialschur(P, nev = num_basis,
    tol = 1e-6, restarts = 200, which = LM())
    eigval, eigvec = partialeigen(decomp)

    order = sortperm(real.(eigval), rev = true)
    eigval = real.(eigval[order])
    eigvec = real.(eigvec[:, order])

    if eigen_cutoff >= 0

        num_basis = sum(eigval .>= eigen_cutoff)
    end

    if num_basis < size(eigval, 1)

        eigval = eigval[1:num_basis]
        eigvec = eigvec[:, 1:num_basis]
    end

    # Normalization so that eigvec_r[:,1] entries are all 1
    # and that eigvec_l[:,1] matches a density

    eigvec_r = eigvec ./ eigvec[:, 1]
    eigvec_l = transpose(eigvec .* (eigvec[:,1]))

    if !history.converged

        @warn(history)

        check = maximum(vec(sum(Diagonal((1) ./ q)*mat, dims = 1))) # should be close to 1
    end

    if (norm(mat * eigvec_r - Diagonal(q)*eigvec_r * Diagonal(eigval)) > 1e-6 ||
        norm(eigvec_l*mat - Diagonal(eigval)*eigvec_l*Diagonal(q)) > 1e-6 ||
        maximum(abs.(sum(eigvec_l[2:end, :], dims = 2))) > 4e-4)

        @warn("The eigenvectors weren't correctly computed.")

        @show norm(mat * eigvec_r - Diagonal(q)*eigvec_r * Diagonal(eigval))
        #@show norm(Diagonal((1) ./ q)*mat * eigvec_r - eigvec_r * Diagonal(eigval))

        @show norm(eigvec_l*mat - Diagonal(eigval)*eigvec_l*Diagonal(q))
        #@show norm(eigvec_l*Diagonal((1) ./ q)*mat - Diagonal(eigval)*eigvec_l)

        @show maximum(abs.(sum(eigvec_l[2:end, :], dims = 2)))

    end

    if (sum(eigvec_l[1,:]) - 1 > 1e-6 ||
        norm((q/sum(q)) .- eigvec_l[1,:]) > 1e-5)

        @warn("The stationary distribution was not correctly calculated")
        @show sum(eigvec_l[1,:]) - 1
        @show norm((q/sum(q)) .- eigvec_l[1,:])

    end

    if scaled
        return eigval, eigvec_l, transpose(eigval).*eigvec_r
    end

    return eigval, eigvec_l, eigvec_r
end
