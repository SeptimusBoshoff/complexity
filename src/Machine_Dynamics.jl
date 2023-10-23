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
function shift_operator(coords; index_map = nothing, return_eigendecomp = false, alg = :dmd, hankel_rank = 1, svd_tr = nothing,  residual = false)

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

        # To replace in formulas:
        valid_next = valid_prev .+ 1

        # The idea now is to express the coordinates with no valid next values
        # as a combination of known coordinates

    elseif alg != :hankel

        # build reverse operators
        valid_prev = deleteat!(collect(range(1, num_coords - 1)), findall(diff(index_map) .!= 1))
        # also known as gaplocs, plus the latest state - these are the non-valid previous coordinates
        indices_no_next = setdiff(range(1, num_coords), valid_prev)

        valid_next = valid_prev .+ 1

    elseif alg == :hankel

        valid_prev = deleteat!(collect(range(1, num_coords - 1)), findall(diff(index_map) .!= 1))
        indices_no_next = setdiff(range(1, num_coords), valid_prev)

        hankel_indices = []

        for i in eachindex(indices_no_next)

            if i == 1

                hankel_indices = [hankel_indices; valid_prev[1:indices_no_next[i]-hankel_rank]]
            else

                hankel_indices = [hankel_indices; valid_prev[indices_no_next[i-1]-i+2:indices_no_next[i]-hankel_rank - i + 1]]
            end
        end
    end

    # we want shift_op * transpose(coords[valid_prev,:]) = transpose(coords[valid_next,:])

    if alg == :nnls

        shift_op = zeros(dims, dims)
        shift_op[1, 1] = 1.0

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

        shift_op = zeros(dims, dims)
        shift_op[1, 1] = 1.0

        # This is slower
        # To test: Using inverses yields the least accurate results.

        shift_op[2:end,:] = (coords[2:end, valid_next]) * pinv(coords[:, valid_prev])

    elseif alg == :dmd # exact dmd

        # Step 1: Compute the singular value decomposition of coords
        U, S, V = svd(coords[:, valid_prev])
        #coords[:, valid_prev] = U*Diagonal(S)*V'

        # Step 2: Truncate matrices
        if isnothing(svd_tr)

            r = length(S)

        elseif svd_tr > dims

            r = dims

        elseif (svd_tr > 0 && svd_tr <= 1)

            # Find r to capture svd_tr% of the energy
            cdS = cumsum(S)./sum(S)
            mean = sum(cdS)/length(S)
            r = findfirst(cdS .>= svd_tr) # truncation rank

        else

            r = svd_tr
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

        #still needs to be rewritten for multiple episodes

        # the first coordinate holds no dynamical information, but it's useful for
        # normalisation later in the prediction
        hankel_dims = dims - 1 # another opportunity for dim reductions

        wind = length(hankel_indices)

        rows = hankel_dims*hankel_rank + 1

        UH = Array{Float64, 2}(undef, rows, wind) # Hankel matrix
        H = Array{Float64, 2}(undef, rows, wind) # Hankel matrix

        j = 1
        for i in 1:hankel_rank

            if j == 1

                @views UH[j:j + hankel_dims, :] = coords[1:hankel_dims+1, hankel_indices .+ i]

                j = j + hankel_dims + 1
            else

                @views UH[j:j + hankel_dims - 1, :] = coords[2:hankel_dims+1, hankel_indices .+ i]

                j = j + hankel_dims
            end
        end

        j = 1
        for i in 1:hankel_rank

            if j == 1

                @views H[j:j + hankel_dims, :] = coords[1:hankel_dims+1, hankel_indices]

                j = j + hankel_dims + 1
            else

                @views H[j:j + hankel_dims - 1, :] = coords[2:hankel_dims+1, hankel_indices .+ (i - 1)]

                j = j + hankel_dims
            end
        end

        # Step 1: Compute the singular value decomposition of H

        U, S, V = svd(H)
        #H = U*diagm(S)*V'

        # Step 2: Truncate matrices
        if isnothing(svd_tr)

            r = length(S)

        elseif svd_tr > rows

            r = rows

        elseif (svd_tr > 0 && svd_tr <= 1)

            # Find r to capture svd_tr% of the energy
            cdS = cumsum(S)./sum(S)
            mean = sum(cdS)/length(S)
            r = findfirst(cdS .>= svd_tr) # truncation rank

        else

            r = svd_tr
        end

        Ur = U[:, 1:r]
        Sr = Diagonal(S[1:r])
        Vr = V[:, 1:r]

        # Step 3: Project A onto the POD modes of U

        U_map = UH*Vr*inv(Sr)

        shift_op = (Ur')*U_map

    elseif alg == :GA

        #m = length(valid_next)

        A = coords[:, valid_next]*transpose(coords[:, valid_prev])#*(1/(m))
        G = coords[:, valid_prev]*transpose(coords[:, valid_prev])#*(1/(m))

        shift_op = A/G

    end

    eigval, eigvec_r = eigen(shift_op)

    if alg == :dmd || alg == :hankel

        # Step 4: The DMD modes are eigenvectors of the high-dimensional A matrix. The DMD
        # eigenvalues are also the eigenvalues of the full shift operator.

        Phi = U_map*eigvec_r

        #= Phi = zeros(Complex, r+1, r+1)
        Phi[1] = 1

        Phi[2:end, 2:end] = U_map*eigvec_r =#

        #lambda = Diagonal(eigval) #discrete time

        #omega = log.(eigval)/timestep # continuous time DMD eigenvalues

        #x1 = coords[:,1]
        #b = Phi\x1 # tells you how much of each mode is going at each time
        #b = Phi\H[:,1] # for hankel dmd

        #alpha1 = Sr*vec(Vr[1,:]')
        #alpha1 = (Ur')*coords[:,1]
        #b = (eigvec_r*lambda)\alpha1 # mode amplitude

        # A Φ = Λ Φ # the least square fit matrix
        #shift_op = real.(Phi*lambda*pinv(Phi)) # bad idea if large

    end

    # This is to check that the eigendecomposition isn't incorrect
    #= Λ = Diagonal(eigval)
    eigvec_l = inv(eigvec_r)
    println(shift_op * eigvec_r ≈ eigvec_r * Λ) # should return true
    println(eigvec_l * shift_op ≈  Λ * eigvec_l) # should return true =#

    # Now ensure the operator power does not blow up
    # Eigenvalues should have modulus less than 1...

    if maximum(abs.(eigval)) > (1 + 1e-5) #&& (alg != :hankel && alg != :dmd)

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

        if !(alg == :dmd || alg == :hankel)

            shift_op = real.((eigvec_r * Diagonal(eigval)) * pinv(eigvec_r))
        end

    end

    if alg == :dmd || alg == :hankel

        W = (eigvec_r*Diagonal(eigval))\(Ur') #Koopman eigenfunctions as rows

        DMD = (Phi, eigval, W) # DMD modes, eigenvalues, eigenfunction weights
        SingleValueDecomposition = (U, S, V)

        if !residual

            return DMD, SingleValueDecomposition
        else

            #shift_op = real(Phi*Diagonal(eigval)*W) #actual shift op

            #=
                If a large number of basis functions are used, then the condition number of
                the problem might deteriorate. Another posibility is to compute the
                residual. A large error indicates that the set of basis functions cannot
                represent the eigenfunctions accurately. In the case of Hankel DMD, this may
                happen when the trajectory of a continuous time system is sampled with a
                short interval.
            =#

            if alg == :dmd

                residual = norm(coords[:, valid_next] - real(Phi*Diagonal(eigval)*W)*coords[:, valid_prev])

            else

                residual = norm(UH - real(Phi*Diagonal(eigval)*W)*H)
            end

            return DMD, SingleValueDecomposition, residual

        end

    else

        if !return_eigendecomp

            return shift_op

        else
            eigvec_l = pinv(eigvec_r)

            return shift_op, eigval, eigvec_l, eigvec_r
        end

    end
end

function immediate_future(data, indices)
    """
    Equivalent to lambda d,i: d[i+1,:] . This function is used as a default argument in expectation_operator. See the documentation there.
    """
    return data[indices .+ 1, :]
end

function present(data, indices)
    """
    Equivalent to lambda d,i: d[i+1,:] . This function is used as a default argument in expectation_operator. See the documentation there.
    """
    return data[indices, :]
end

function expectation_operator(coords, index_map, targets; func = present, delay_coords = 1)
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

        f_list = fill(func, length(targets_list))
    end

    eoplist = Vector{Matrix{Float64}}(undef, length(targets_list))
    residual = Vector{Float64}(undef, length(targets_list))

    if delay_coords != 1

        dims = size(coords, 1)
        num_coords = size(coords, 2)

        indices_no_next = setdiff(range(1, num_coords), deleteat!(collect(range(1, num_coords - 1)), findall(diff(index_map) .!= 1)))

        indices = collect(range(1, num_coords))

        new_indices = []

        for i in eachindex(indices_no_next)

            if i == 1

                new_indices = [new_indices; indices[1:indices_no_next[i]-delay_coords + 1]]
            else

                new_indices = [new_indices; indices[indices_no_next[i-1] + 1:indices_no_next[i]-delay_coords + 1]]

            end
        end

        #-----------------------------------------------------------------------------------

        gap_locs = setdiff(indices, new_indices)

        new_index_map = deleteat!(copy(index_map), gap_locs)

        #-----------------------------------------------------------------------------------

        delay_dims = dims - 1

        wind = length(new_indices)

        rows = delay_dims*delay_coords + 1

        C = Array{Float64, 2}(undef, rows, wind) # coordinate matrix

        j = 1
        for i in 1:delay_coords

            if j == 1

                @views C[j:j + delay_dims, :] = coords[1:delay_dims+1, new_indices]

                j = j + delay_dims + 1
            else

                @views C[j:j + delay_dims - 1, :] = coords[2:delay_dims+1, new_indices .+ (i - 1)]

                j = j + delay_dims
            end
        end

    end

    e_cnt = 1

    for (tar, fun) in zip(targets_list, f_list)

        #=
            We want to solve the least-squares problem:

            fvalues = E*coords

            and identify the best fit linear operator E that maps the causal state back to
            the measurement space
        =#

        if delay_coords == 1

            fvalues = fun(tar, index_map)

            valid_f = (findall(vec(prod(.!isnan.(fvalues), dims = 2))))

            eoplist[e_cnt] = transpose(transpose(coords[:,valid_f]) \ (fvalues[valid_f,:]))

            #eoplist[e_cnt] = transpose(fvalues[valid_f,:]) * pinv(coords[:,valid_f])

            residual[e_cnt] = norm(eoplist[e_cnt]*coords[:,valid_f] .- transpose(fvalues[valid_f,:]))

        else

            fvalues = fun(tar, new_index_map .+ (delay_coords - 1))
            # if this ofset is greater than npast or nfuture there may be issues

            valid_f = (findall(vec(prod(.!isnan.(fvalues), dims = 2))))

            eoplist[e_cnt] = transpose(transpose(C[:,valid_f]) \ (fvalues[valid_f,:]))

            residual[e_cnt] = norm(eoplist[e_cnt]*C[:,valid_f] .- transpose(fvalues[valid_f,:]))

        end

        e_cnt += 1

    end

    return eoplist, residual
end

#= function predict(npred, coords_ic, expect_op; return_dist = 2, bounds = nothing, knn_convexity = nothing, coords = nothing, knndim = nothing, extent = nothing, DMD = nothing, shift_op = nothing)
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

    coord = copy(coords_ic)[:, 1] # the a sequence leading up to the most recent initial condition
    c_dims = size(coord, 1)

    pred_coords = Array{Float64, 2}(undef, c_dims, npred)

    if !isnothing(DMD)

        Phi = DMD[1] # DMD modes / projected Koopman right eigenvectors
        Λ = Diagonal(DMD[2]) #discrete time eigenvalues
        Λm = copy(Λ)
        #b = DMD[3] # DMD mode amplitudes, coefficients of Koopman eigenfunctions

        if size(DMD[2], 1) != size(coord, 1)

            # we're using hankel dmd
            hankel_coord = [1; vec(coords_ic[2:end, 1:end])]

            hankel_rank = (length(DMD[2]) /size(coord, 1))

            #= if length(vec(coords_ic[:, end:-1:1])) != size(Phi, 1)
                #warning
            end =#
            # mode amplitudes, i.e. the coordinates of x(0) in the base defined by the eigenvectors.

            #b = pinv(Phi)*hankel_coord
            #b = Phi\hankel_coord
            b = (DMD[3]*Λ)\((DMD[4]')*hankel_coord) # projected Koopman eigenfunction coefficients

        else

            #b = pinv(Phi)*coord
            #b = Phi\coord
            #b = (Phi*Λ)\((DMD[3]')*coord) # projected Koopman eigenfunction coefficients
            b = (DMD[3]*Λ)\((DMD[4]')*coord) # projected Koopman eigenfunction coefficients
        end
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
            pred_coords[:, p] = coord
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
            coord = real(Phi*(Λm)*b)[1:c_dims]
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

        return pred, pred_coords
    end

    return pred
end =#

function predict(npred, coords_ic, expect_op; return_dist = 2, DMD = nothing, shift_op = nothing)
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

    """

    coord = copy(coords_ic)[:, end] # the sequence leading up to the most recent initial condition

    c_dims = size(coord, 1) # dimension of a causal state

    eo_len = length(expect_op)
    eop_dims = length(expect_op[1]) # dimension of the expectation operator
    eop_rank = Int(round.((eop_dims - 1) / (c_dims - 1), digits = 0))
    hankel_rank = eop_rank

    pred_coords = Array{Float64, 2}(undef, c_dims, npred)

    pred = [Vector{Float64}(undef, npred) for _ = 1:eo_len]

    if !isnothing(DMD)

        hankel_rank = Int(round.((size(DMD[1], 1) - 1) / (c_dims - 1), digits = 0))

        Phi = DMD[1] # projected Koopman left eigenvectors
        Λ = Diagonal(DMD[2]) #discrete time eigenvalues
        #W = DMD[3] # DMD mode amplitudes, coefficients of Koopman eigenfunctions

        Λm = copy(Λ) # the only matrix that changes with the evolution

        if ((size(DMD[1], 1) - 1) % (c_dims - 1)) == 0

            # we're using hankel dmd
            hankel_coord = [1; vec(coords_ic[2:end, end - hankel_rank + 1:end])]

            #b = pinv(Phi)*hankel_coord #b = Phi\hankel_coord

            # projected Koopman eigenfunction coefficients
            # mode amplitudes, i.e. the coordinates of x(0) in the base defined by the eigenvectors.
            b = (DMD[3])*hankel_coord

        else

            b = (DMD[3])*coord
        end

        if hankel_rank < eop_rank

            eop_coord = [1; vec(coords_ic[2:end, 1:eop_rank])]

        end
    end

    for p in 1:npred

        # Evolve the distribution

        if isnothing(DMD)
            coord = shift_op * coord

        else

            coord = real(Phi*Λm*b)
            Λm = Λm*Λ

        end

        # Normalize - to prevent numerical errors from accumulating
        coord /= coord[1, 1]

        # Apply the expectation operator to make a prediction in measurement space

        for (eidx, eop) in enumerate(expect_op)

            if hankel_rank == eop_rank

                pred[eidx][p] = (eop * coord)[1]

            elseif hankel_rank > eop_rank

                # if the present function is used to construct the expectation operator,
                # then the measurement values will lag by 'hankel_rank - eop_rank'

                #pred[eidx][p] = (eop * coord[1:eop_dims])[1]
                pred[eidx][p] = (eop * [1.0; coord[end - eop_dims + 2:end]])[1]

            elseif hankel_rank < eop_rank

                #@views eop_coord[2:end] =

                pred[eidx][p] = (eop * eop_coord)[1]

            end

        end

        # saving
        if return_dist == 2

            #pred_coords[:, p] = coord[1:c_dims]
            pred_coords[:, p] = [1.0; coord[end - c_dims + 2:end]]
        end

    end

    if return_dist == 1

        return pred, coord

    elseif return_dist == 2

        return pred, pred_coords
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
function new_coords(Ks, Gs, coords; alg = :nnls, ϵ = 1e-6)

    if alg == :ldiv

        Ω = copy(Ks) # = inv(Gs)*Ks
        ldiv!(cholesky(Gs + ϵ*I), Ω) # Gs \ Ks

    elseif alg == :nnls

        # Note that the Non-Negative constraint is enough in this case,
        # Since the solution sums to 1 with the diagonal entries in Gs
        #Ω = nonneg_lsq(transpose(Gs)*Gs, transpose(Gs)*Ks, alg = :nnls)
        Ω = nonneg_lsq(Gs, Ks, alg = :nnls)
    end

    Ω = Ω ./ sum(Ω, dims = 1)

    # q0 is specified in the original RKHS representation, for each sample.
    # Turn it into an eigen space representation (scaled if coords are scaled)

    nc = coords*Ω

    # Since q0 is normalized, and since all the first components of the coordinates are 1,
    # the first entry of the state distribution should also be 1

    if !isapprox(nc[1,:], ones(length(nc[1,:])))

        @warn ("The resulting state distribution must be normalized to have 1 as its first entry, but this is not the case. Did you pass the correct coordinates?")
    end

    # ensure there are no numerical roundoff
    nc = nc ./ transpose(nc[1,:])

    return nc
end

function evolve(npred, Ks, DMD)

    pred = Array{Float64, 2}(undef, 2, npred)

    Phi = DMD[1]

    Λ = Diagonal(DMD[2])
    Λm = copy(Λ)

    b = DMD[3]*Ks[1:end-1]

    for p in 1:npred

        pred[:, p] = real.(Phi*Λm*b)
        Λm = Λm*Λ

        # maybe make sure that the first eigenvalue remains equal to 1?
    end

    return pred
end
