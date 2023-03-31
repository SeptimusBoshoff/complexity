#= Authors:
    Source code adapted from the Continuous Causal States method, described in the paper :
    Discovering Causal Structure with Reproducing-Kernel Hilbert Space ε-Machines by Nicolas
    Brodu and James P. Crutchfield
    Original code in python converted to julia code by Septimus Boshoff
=#

"""
    shift_op, (eigval, eigvec_l, eigvec_r) = shift_operator(coords; index_map = nothing, return_eigendecomposition = false, alg = :nnls, coord_eigenvals = nothing)

# Description
Computes a discrete shift operator, similar to the Koopman operator, for the dynamical
system typically defined on a diffusion state space.

# Arguments
- `coords::Array{Float64, 2}`: A 'D x N' matrix of coordinates, where every column defines a
  point in the state space.

# Keyword Arguments
- `index_map::Array{Int, 1}`: The indices of the valid (not NaN) and contiguous
  (past,future) pairs.
- `return_eigendecomp::Bool`: If true the the function returns the eigendecomposition of the
  shift operator.
- `alg::Symbol`: The algorithm. TODO

# Return Values
- `shift_op::Matrix{Float64}`: A 'D x D' matrix which defines the forward operator for coordinates
  distributions, represented in the eigenbasis. if return_eigendecomposition is True, also
  return the eigenvalues and left, right eigenvectors
- `eigval::Vector{Float64}`: TODO
- `eigvec_l::Matrix{Float64}`: TODO
- `eigvec_r::Matrix{Float64}`: TODO
"""
function shift_operator(coords; index_map = nothing, return_eigendecomp = false, alg = :nnls, hankel_rank = 5, hankel_dims = size(coords, 1), svd_r = nothing)

    # TODO: generator, meaning the log in eigen decomposition form
    # This would allow to compute powers more efficiently

    # Shift operator, expressing the new coordinates as a combination of the old ones.
    # This definition is similar to that of the kernel Koopman operator in RKHS,
    # but done here in diffusion map coordinates space

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

    # The idea now is to express the coordinates with no valid next values
    # as a combination of known coordinates

    # we want shift_op * transpose(coords[valid_prev,:]) = transpose(coords[valid_next,:])

    if alg == :nnls

        # Using non-negative least squares works better

        # nnls does not solve A*X = B, it solves A = inv(A)*A, B = inv(A)*B
        X = nonneg_lsq(coords[:, valid_prev], coords[:, indices_no_next], alg = :nnls)
        # :nnls is most accurate

        X = X ./ sum(X, dims = 1) # should already be close to 1

        T = diagm(-1 => ones(num_coords - 1))

        # put that back into T. Process column by column, due to indexing scheme

        for (ic, c) in enumerate(indices_no_next)

            T[valid_next, c] = X[:,ic]
        end

        #shift_op[2:end,:] = transpose(transpose(coords) \ transpose(coords[2:end, :]*T))
        shift_op[2:end,:] = (coords[2:end, :]*T)*pinv(coords)

    elseif alg == :pinv

        # This is slower
        # To test: Using inverses yields the least accurate results.

        shift_op[2:end,:] = (coords[2:end, valid_next]) * pinv(coords[:, valid_prev])

    elseif alg == :dmd

        # Step 1: Compute the singular value decomposition of coords

        U, S, V = svd(coords[:, valid_prev])

        #coords[:, valid_prev] = U*diagm(S)*V'

        # Step 2: Truncate matrices
        # Find r to capture 90% of the energy
        cdS = cumsum(S)./sum(S)
        mean = sum(cdS)/length(S)
        #r = findfirst(cdS .>= 0.99) # truncation rank
        r = svd_r#length(S)

        if r === nothing r = length(S) end

        Ur = U[:, 1:r]
        Sr = Diagonal(S[1:r])
        Vr = V[:, 1:r]

        # Step 3: Project A onto the POD modes of U

        dmd_operator = (coords[:, valid_next])*Vr*inv(Sr)

        # Atilde - tells us how the POD modes evolve in time
        # the least square fit matrix in the low rank subspace
        shift_op = (Ur')*dmd_operator

        #@show (coords[:, valid_prev]) ≈ U*diagm(S)*V'

    elseif alg == :hankel

        hankel_dims = hankel_dims #- 1 # another opportunity for dim reductions

        wind = (num_coords - hankel_rank)

        UH = Array{Float64, 2}(undef, hankel_dims*hankel_rank, wind) # Hankel matrix
        H = Array{Float64, 2}(undef, hankel_dims*hankel_rank, wind) # Hankel matrix

        # Idea: if using diffusion map coordinates, let's throw away the first coordinate
        # cause its always 1

        j = 1
        for i in 1:hankel_rank

            UH[j:j + hankel_dims - 1, :] = coords[1:hankel_dims, i+1:wind + i]

            j = j + hankel_dims

        end

        j = 1
        for i in 1:hankel_rank

            H[j:j + hankel_dims - 1, :] = coords[1:hankel_dims, i:wind + i - 1]

            j = j + hankel_dims

        end

        # Step 1: Compute the singular value decomposition of H

        U, S, V = svd(H)
        #coords[:, valid_prev] = U*diagm(S)*V'

        # Step 2: Truncate matrices
        cdS = cumsum(S)./sum(S)
        mean = sum(cdS)/length(S)
        r = findfirst(cdS .>= mean) # truncation rank
        r = length(S)

        Us = U[:, 1:r]
        Ss = S[1:r]
        Vs = V[:, 1:r]

        # Step 3: Project A onto the POD modes of U

        dmd_operator = H*Vs*inv(Diagonal(Ss))

        shift_op = (Us')*dmd_operator

    end

    eigval, eigvec_r = eigen(shift_op)

    if alg == :dmd || alg == :hankel

        # Step 4: The DMD modes are eigenvectors of the high-dimensional A matrix. The DMD
        # eigenvalues are also the eigenvalues of the full shift operator.

        Phi = dmd_operator*eigvec_r
        lambda = Diagonal(eigval) #discrete time

        #omega = log.(eigval)/timestep # continuous time DMD eigenvalues

        #x1 = coords[:,1]
        #b = Phi\x1 # tells you how much of each mode is going at each time

        alpha1 = Sr*vec(Vr[1,:]')
        b = (eigvec_r*lambda)\alpha1 # mode amplitude

        # A Φ = Λ Φ # the least square fit matrix
        shift_op = real.(Phi*lambda*pinv(Phi))

    end

    # This is to check that the eigendecomposition isn't incorrect
    #= Λ = Diagonal(eigval)
    eigvec_l = inv(eigvec_r)
    println(shift_op * eigvec_r ≈ eigvec_r * Λ) # should return true
    println(eigvec_l * shift_op ≈  Λ * eigvec_l) # should return true =#

    # Now ensure the operator power does not blow up
    # Eigenvalues should have modulus less than 1...

    if maximum(abs.(eigval)) > 1 && (alg != :hankel && alg != :dmd)

        # ... but there may be numerical innacuracies, or irrelevant
        # component in the eigenbasis decomposition (the smaller eigenvalues
        # there have no significance, according to the MMD test, and we use
        # implicitly an inverse here).
        # This should not happen, but if it does, clip eigvals
        n = size(eigval, 1)

        for i in 1:n

            if abs(eigval[i]) > 1

               eigval[i] /= abs(eigval[i])
            end
        end

        # reconstruct best approximation of the shift operator

        Λ = Diagonal(eigval)
        shift_op = real.((eigvec_r * Λ) * pinv(eigvec_r))

        if return_eigendecomp

            eigval, eigvec_r = eigen(shift_op)
        end
    end

    if return_eigendecomp && alg != :dmd

        eigvec_l = inv(eigvec_r) # may cause issues, i.e. slow
        return shift_op, eigval, eigvec_l, eigvec_r

    elseif alg == :dmd

        DMD = (Phi, eigval, b) # DMD eigenvectors, eigenvalues, modes
        SingleValueDecomposition = (U, S, V)
        return shift_op, DMD, SingleValueDecomposition

    else

        return shift_op

    end
end

function immediate_future(data, indices)
    """
    Equivalent to lambda d,i: d[i+1,:] . This function is used as a default argument in expectation_operator. See the documentation there.
    """
    return data[indices .+ 1, :]
end

function expectation_operator(coords, index_map, targets; func = immediate_future)
    """
    Builds the expectation operator, mapping a state distribution expressed in the eigenbasis, into numerical values, expressed in the original series units.

    Parameters
    ----------

    coords : array
        coordinates in the eigenbasis, as returned by the spectral_basis function

    index_map : int array
        This is the index_map returned by the series_Gxy function. It indicates the index in the
        series for the (past, future) pair matching each coordinate entry. That index is that of
        the present, the last value in the past series. If the series consists of several discontiguous
        blocks, the index refers to the valid entries in the series that would be made by concatenating
        these blocks (that series would have invalid entries at each time discontinuity, at nan values, etc).
        The targets parameter may have its own validity pattern, and one that differs for each heterogenous
        data source - it may be that a computed state, has no matching target for one data source, but a valid
        target for another. Expectation operators are computed using all valid data, for each heterogenous data source.

    targets : array or list of array
        Target values that we wish to predict from causal states, to build an operator from. Each array must have
        the same temporal structure (number of samples, discontinuous blocks, etc) as the data series that were
        used to build the causal states - one target is needed for each original series measurement. Some targets
        may be NaN, in which case they will be ignored for building the operator. These NaN patterns may differ
        from these implied by the causal state index_map. If a list is provided as the targets argument,
        these are observations from multiple (possibly heterogenous) data source, and possibly different sources
        from those used for building the causal states. Then, a list of expectation operators is returned, one of
        each data source.

    function : callable, or list of callable, optional
        This is the function of the targets, which expectation is computed by the operator. Different functions
        can be provided for each heterogenous data source by providing a list of callables. Each function takes
        as parameter a data array (which is in fact the stacked targets) and an index array (the index_map), and
        should return the function computed on the data at each specified index. The default is to use the
        immediate_future function, which is equivalent to lambda d,i: d[i+1,:].
        Due to the way the index_map is built, the function is applied only on target data values that match
        valid causal states. Thus, it is guaranteed that data in the range i-npast+1:i+nfuture is valid: i is
        the "present", the last entry of the past sequence in the consecutive (past, future) pair.
        Assuming nfuture>=1, then the next value at i+1 always exists and it is the first entry in the future sequence.


    Returns
    -------
    expect_ops : array or list of arrays
        This is the expectation operator for the function (or functions) that compute a value from the current state.
        The expectation is implicitly performed by the reproducing property and using mean maps for the probability
        distributions. If a list of data sources was provided as the series argument, return one search operator for
        each data source.

    Notes
    ----

    - As suggested by the 'targets' parameter name, these need not be the same as the 'series' argument of the series_Gxy
    that were used to build the causal states. You may very well construct the causal states from some data sources,
    and try to predict only a subset of these sources, or maybe other observables entirely that you assume depend
    on the causal states. The only restriction is that the series and the targets variable share the same temporal
    pattern: the same number of samples in the same number of continuous data blocks. Use NaN if some target values
    are not available for all sample times.

    - Theoretically, any function of the causal state is useable, so long as it can be estimated for each observed
    (past, future) data pair in the target measurements. The function could thus very well take non-numeric data as
    argument, but, currently, it should only return numeric values (scalar or vectorial).

    - TODO: Given the above, it would be possible to extend other machine learning algorithms taking as input the
    (past,future) pairs and producing some value of interest. The machine learning trained instance could be fed
    as the function parameter, then it would benefit from the causal state machinery.

    """

    if size(targets[1], 1) == 1
        targets_list = [targets]
    else
        targets_list = targets
    end

    if func isa Vector
        f_list = func
    else
        f_list = Array{Function, 1}(undef, length(targets_list))
        f_list = fill!(f_list, func)
    end

    eoplist = Vector{Matrix{Float64}}(undef, length(targets_list))

    rtol = sqrt(eps(real(float(one(eltype(coords))))))
    sci = pinv(coords, rtol = rtol)

    e_cnt = 1

    for (tar, fun) in zip(targets_list, f_list)

        fvalues = fun(tar, index_map)

        valid_f = (findall(vec(prod(.!isnan.(fvalues), dims = 2))))

        eoplist[e_cnt] = transpose(fvalues[valid_f,:]) * sci[valid_f,:]
        e_cnt += 1
    end

    return eoplist
end

function predict(npred, coords_ic, shift_op, expect_op; return_dist = 0, bounds = nothing, knn_convexity = nothing, coords = nothing, knndim = nothing, extent = nothing, DMD = nothing)
    """
    Predict values from the current causal states distribution

    Parameters
    ----------
    npred : int
        Number of predictions to generate

    state_dist : array of size (num_basis, 1)
        Current state distribution, expressed in the eigen basis

    shift_op : array of size (num_basis, num_basis)
        Operator to evolve the distributions in time

    expect_op : array of size (data_dim, num_basis) or a list of such arrays
        Operator, or list of operators, that take a distribution of causal states and generate a data value compatible with the original series

    return_dist : {0, 1, 2}, optional
        Whether to return the state distribution, or update it:
        - 0 (default): do not return a state vector
        - 1: return the updated state vector
        - 2: return an array of state_dist vectors, one row for each prediction

    bounds : vector of vectors, optional
        Bounds for each expect_op operator prediction.
        Should have the same length as expect_op.
        Each list entry is itself an array-like of two elements, one for the lower and one for the upper bound.
        Use None or +/- inf values to leave a bound unspecified.

    knn_convexity : int or None, optional
        If specified, restrict each state estimate to the closest point in the convex enveloppe of its nearest neighbors.
        That envelop can be slightly enlarged with the extent parameter, because some valid states may exist out of the currently observed data points.
        knn_convexity specifies the number of neighbors to use. This number should be large enough so as to reach multiple nearby trajectories.
        Otherwise, neighbors consist only of the points just before/after along trajectories and the state synchronizes to the training data.
        Additional arguments are required:

          + coords: The coordinates of each observed state in the eigen basis. This argument is required.

          + extent (optional): Change the bounds of the convexity constraints from [0,1], to [0-extent,1+extent]. The default is 0,
          but you should really try to increase this to something like 0.05, possibly adding data bounds if needed.

          + knndim (optional): Restrict the number of data dimensions to make the nearest neighbors search on.
          This can be useful when a low-dimensional attractor is embedded in a large-dimension space, where only the first few
          dimensions contribute to the distances, for accelerating the computations. knndim can be a number, in which case the first
          dimensions below that number are selected, or an array-like object of indices. The default is 100, possibly lower if less
          components are provided.

    Returns
    -------
    predictions: array, or list of arrays, matching the expect_op type
        As many series of npred values as there are expectation operators.
        Either a list, or a single array, matching the expect_op type

    updated_dist: optional, array
        If return_dist > 0, the updated state distribution or all such distributions
        are returned as a second argument.

    Notes
    -----

    The causal state space (conditional distributions) is not convex: linear combinations of
    causal states do not necessarily correspond to a valid value in the conditionned domain.
    We are dealing here with distributions of causal states, but these distributions are
    ultimately represented as linear combinations of the data span in a Reproducing Kernel
    Hilbert Space, or, in this case, as combinations of eigenbasis vectors (themselves
    represented in RKHS). Therefore, the state distributions are also linear combinations of
    other causal states. Ultimately, the continuous-time model converges to a limit
    distribution, which is an average distribution that need not correspond to any single
    state, so the non-convexity is not an issue for that limit distribution. Making
    predictions with the expectation operator is still feasible, and we get the expected
    value from the limit distribution as a result.

    If, instead, one wants a trajectory, and not the limit average, then a method is
    required to ensure that each predicted state remains valid as a result of applying a
    linear shift operator. The Nearest Neighbours method is an attempt to solve this issue.
    The preimage issue is a recurrent problem in machine learning and no single answer can
    currently solve all cases.

    """

    coord = copy(coords_ic) # the a sequence leading up to the most recent initial condition
    c_dims = size(coord, 1)

    if size(coord, 1) < size(shift_op, 1) && !isnothing(DMD)
        # we are using hankel dmd

        # new predicted diffusion space coordinates
        new_coords = Array{Float64, 2}(undef, size(shift_op, 1), npred)

        hankel_rank = Int(size(shift_op, 1)/size(coord, 1))

        coord = vec(coord[:, end:-1:1])
        display(coord)

    else
        coord = coord[:, 1]
        # new predicted diffusion space coordinates
        new_coords = Array{Float64, 2}(undef, length(coord), npred)
    end

    if !isnothing(DMD)
        Phi = DMD[1]
        Λ = Diagonal(DMD[2]) #discrete time eigenvalues
        Λm = copy(Λ)
        #b = DMD[3]
        U, S, V = svd(coord)
        b = Phi\coord
    end

    eo_len = length(expect_op)

    pred = Vector{Matrix{Float64}}(undef, eo_len)

    problem = nothing

    if isa(knn_convexity, Int) && knn_convexity > 0

        num_basis = size(coords, 1)

        if isnothing(knndim) || knndim > num_basis || !isa(knndim, Int)
            knndim = num_basis
        end
        if isnothing(extent) extent = 0.0 end

        # Euclidean(3.0), Chebyshev, Minkowski(3.5) and Cityblock
        balltree = BallTree(coords[1:knndim, :]; leafsize = 30)

        if knn_convexity > 1

            problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            set_silent(problem)

            # Optimisation problem
            @variable(problem, -extent <= w[i = 1:knn_convexity] <= 1 + extent) # weights between nearest neighbours
            @NLparameter(problem, M[i = 1:num_basis, j = 1:knn_convexity] == 0) # nearest neighbours

            @NLexpression(problem, sum_w, sum(w[i] for i in 1:knn_convexity))
            @NLconstraint(problem, sum_w == 1) # weights have to sum to 1

            # x is defined as a combination of nearest neighours
            x = Array{NonlinearExpression, 1}(undef, num_basis) # the end result - state distribution

            for i in 1:num_basis # maybe parallelize?
                x[i] = @NLexpression(problem, sum(M[i, j]*w[j] for j in 1:knn_convexity)) # matrix multiplication
            end

            # Preserve the state distribution normalization
            # This is always 1, in whatever scaled or coords units, since
            # the first eigenvatlue is 1.
            @NLconstraint(problem, x[1] == 1.0)

            @NLparameter(problem, target_x[i = 1:num_basis] == 0) # the distribution found by the shift operator

            @NLexpression(problem, sum_squares, sum((x[i] - target_x[i])^2 for i in 1:num_basis))
        end
    else

        knn_convexity = nothing
    end

    if !isnothing(bounds)

        if length(expect_op) == 1 && length(bounds) == 2
            bounds = [bounds]
        end

        if isnothing(problem)

            problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
            set_silent(problem)

            @variable(problem, x[i = 1:num_basis] <= 1.0)

            @NLconstraint(problem, x[1] == 1.0)

            @NLparameter(problem, target_x[i = 1:num_basis] == 0)
            @NLexpression(problem, sum_squares, sum((x[i] - target_x[i])^2 for i in 1:num_basis))
        end

        pred_value = Array{NonlinearExpression, 1}(undef, length(expect_op)) # the end result - state distribution

        for (i, b) in enumerate(bounds)

            pred_value[i] = @NLexpression(problem, sum(expect_op[i][j]*x[j] for j in 1:num_basis))

            if b[1] != -Inf && b[1] != Inf && !isnothing(b[1])
                @NLconstraint(problem, pred_value[i] >= b[1])
            end

            if b[2] != Inf && !isnothing(b[2])
                @NLconstraint(problem, pred_value[i] <= b[2])
            end
        end
    end

    for p in 1:npred

        if return_dist == 2
            new_coords[:, p] = coord
        end

        # Apply the expectation operator to the current distribution
        # in state space, to make a prediction in data space

        for (eidx, eop) in enumerate(expect_op)

            new_pred = (eop * coord[1:c_dims])

            if ndims(new_pred) == 1
                new_pred = reshape(new_pred, :, 1)
            end

            if p == 1
                pred[eidx] = cat(new_pred, dims = 1)
            else
                pred[eidx] = cat(pred[eidx], new_pred, dims = 1)
            end

        end

        # Evolve the distribution

        if isnothing(DMD)
            coord = shift_op * coord
        else
            #coord = real(Phi*exp(Diagonal(omega.*(μ_m*p)))*b)
            coord = real(Phi*(Λm)*b)
            Λm = Λm*Λ
        end
        # Normalize - not needed anymore by construction of the shift op??
        coord /= coord[1, 1]

        if !isnothing(knn_convexity)

            idxs, _ = knn(balltree, coord[1:knndim], knn_convexity)

            # idea: it is possible to set the start values of the weights based on the distances

            if knn_convexity > 1

                temp_c = Matrix(coord[:, idxs])

                # Setting the parameters to their newest values
                for i in 1:num_basis

                    for j in 1:knn_convexity

                        set_value(M[i, j], temp_c[i, j])
                    end

                    set_value(target_x[i], coord[i])
                end

            elseif knn_convexity == 1

                coord = coords[:, idxs[1]]

                if !isnothing(problem)

                    for i in 1:num_basis

                        set_value(target_x[i], coord[i])
                        set_start_value(x[i], coord[i])
                    end
                end
            end

        elseif !isnothing(problem)

            for i in 1:num_basis

                set_value(target_x[i], coord[i])
                set_start_value(x[i], coord[i])
            end
        end

        if !isnothing(problem)

            @NLobjective(problem, Min, sum_squares)
            optimize!(problem)

            if value.(x)[1] != NaN && value.(x)[1] != 0.0

                coord = value.(x)
                # should not be needed mathematically if the
                # constraint was perfectly respected.
                # There could be numerical inaccuracies in practice
                coord /= coord[1, 1]
            end
        end
    end

    if return_dist == 1

        return pred, coord

    elseif return_dist == 2

        return pred, new_coords
    end

    return pred
end

"""
    state_dist = new_coords(Ks, Gs, coords; alg = :nnls)

# Description
Compute the best matching state distributions.

# Arguments
- `Ks::Array{Float64, 2}`: A 'N x L' matrix of kernel similarity vectors between a new state
  estimate and reference states. Such vectors can be computed using the series_newKx for
  time series data and embedding that result with embed_Kx. L is the number of such vectors,
  one per column.
- `Gs::Array{Float64, 2}`: A similarity matrix between every causal state. Entries Gs[i,j]
  can be seen as inner products between states Sᵢ and Sⱼ.
- `coords::Array{Float64, 2}`: A 'N x M' matrix where every column is a coordinate in
  diffusion space. These are returned by the spectral_basis function.

# Keyword Arguments
- `alg::Symbol`: What method to use for computing the state distribution.
    - "nnls" -> uses non-negative least squares to ensure that the new state estimate
      remains within the boundaries of existing states.
    - "unbounded" -> unbounded estimate, equivalent to a Nyström extension, which is then
      normalized into a pseudo-distribution.

# Return Values
- `state_dist::Array{Float64, 2}`: array of size 'M x L' The state distribution, represented
  as coefficients in the eigenbasis. The first entry of the state distribution is always be
  1.0 In order to retreive a proper probability distribution, you can do q = basis @
  state_dist. Note that, after evolution by the shift operator, the expression of the causal
  states as linear combinations of RKHS basis elements may induce that some entries of q
  that are negative. q is both the distribution, and its expression as a linear combination
  on the RKHS samples.

"""
function new_coords(Ks, Gs, coords; alg = :nnls)

    if alg == :unbounded

        Ω = copy(Ks)
        ldiv!(bunchkaufman(Gs), Ω) # Gs \ Ks

    else

        # Note that the Non-Negative constraint is enough in this case,
        # Since the solution sums to 1 with the diagonal entries in Gs
        Ω = nonneg_lsq(Gs, Ks, alg = :nnls)
    end

    Ω = Ω ./ sum(Ω, dims = 1)
    # q0 is specified in the original RKHS representation, for each sample.
    # Turn it into an eigen space representation (scaled if coords are scaled)
    nc = transpose(Ω) * transpose(coords)
    # Since q0 is normalized, and since all the first components of the coordinates are 1, the first entry of the state distribution should also be 1
    if !isapprox(nc[:,1],ones(length(nc[:,1])))

        @warn ("The resulting state distribution must be normalized to have 1 as its first entry, but this is not the case. Did you pass the correct coordinates?")
    end
    # ensure there are no numerical roundoff
    nc = nc ./ nc[:,1]

    return transpose(nc)
end
