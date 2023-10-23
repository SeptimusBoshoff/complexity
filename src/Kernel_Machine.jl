#= Authors:
    Source code adapted from the Continuous Causal States method, described in the paper :
    Discovering Causal Structure with Reproducing-Kernel Hilbert Space ε-Machines by Nicolas
    Brodu and James P. Crutchfield
    Original code in python converted to julia code by Septimus Boshoff
=#

"""
    series_Gxy(series, scale, npast, nfuture; kernel_type = "Gaussian", decay = 1,
    qcflags = nothing, localdiff = 0, take2skipX = 0)

# Description:
Compute the past and the future Gram matrices, using the kernel set by set_kernel (defaults
to Gaussian)

# Arguments:
- `series::Vector{Vector{Float64}} or Vector{Float64}`: Each vector is a time-contiguous
  block of data, measured by a sensor, with as many elements as time samples. NaN values are
  allowed and detected.
- `scale::Vector{Float64}` or Float64: If a single float is provided it applies to all
  series. A vector of floats specifies a different scale for each data source.
- `npast::Vector{Int} or Int`: take that consecutive number of samples for building an
  observed past. If a single integer is provided it applies to all series. Different values
  can be provided for different data sources (=each series). It is assumed that data samples
  are measured at matching times for each data source, hence the first causal state can only
  be computed after the-largest-npast values have been observed.
- `nfuture::Vector{Int} or Int`: take that consecutive number of samples, just after the
  past samples, for building an observed future. If a single integer is provided it applies
  to all series. Similarly as for npast, the largest nfuture value sets the time of the last
  computable causal state.
- `decay::Vector{Int} or Int`: ratio for the furthermost weight compared to the immediate
  past/future. Defaults to 1 = no decay. If a single float is provided it applies to all
  series

# Keyword Arguments:
- `qcflags::Int`: (optional) quality control flags. <= 0 means invalid data. If present
  there must be one qcflag per sample
- `localdiff::Int`: how to compute the differences between the sequences, pasts and futures.
  The default is 0 = just take the difference. This assumes that what matters are the series
  absolute values. localdiff = 1 removes the average (weigthed using decay) of each sequence
  before taking their differences. This may be useful for eliminating a slow drift due to
  the sensor itself, which should be removed from data. localdiff = 2 means that the
  "present" value is used as reference. This assumes that what matters are relative
  variations around the "present" value. Warning: this gives equal importance to all such
  variations. For example, a fluctuation of -3°C in summer (from ~25°C) is not the same as a
  fluctuation of -3°C in winter (which may very well cross the freezing point).
- `take2skipX::Int`: (optional) (default 0) performs a special kind of subsampling: two
  consecutive samples are retained, X samples are discarded, then this (X+2) sequence
  repeats. This scheme is designed to preserve consecutive entries for building a shift
  operator while still allowing subsampling of very large series. The classic subsampling
  scheme (take one out of X) can also be applied a priori to the original series, but it is
  equivalent to a (bad) low-pass filtering. Then, the shift operator would be computed on
  consecutive entries of the subsampled series, hence at a different time scale. The
  take2skipX allows to still work on the original time scale. Both can be combined (after
  appropriate low-pass filtering).

# Return Values
- `Gx::Array{Float64, 2}`: A square Gram matrix of past events
- `Gy::Array{Float64, 2}`: A square Gram matrix of future events
- `idxmap::Array{Int, 1}`: Returns the indices of the valid (not NaN) and contiguous
  (past,future) pairs. For each returned row/col index of the Gx, Gy matrices, the idxmap
  specifies what data in the original series that (past,future) reference pair refers to.

"""
function series_Gxy(series, scale, npast, nfuture; decay=1, qcflags=nothing, localdiff=0, take2skipX=0, kernel_type="Gaussian")

    kernel_params_ = set_kernel(kernel_type = kernel_type)

    series_type = "nothing"

    if typeof(series) == Vector{Float64}

        # the state space is 1-d
        series_list = [series]

        series_type = "R¹"
        num_eps = 1 # the number of runs
        nseries = 1 # the number of sources / sensors

    elseif typeof(series) == Vector{Vector{Float64}}

        series_list = series

        series_type = "Rⁿ"
        num_eps = 1 # the number of runs
        nseries = size(series, 1) # the number of sources / sensors

    elseif typeof(series) == Vector{Vector{Vector{Float64}}}

        series_list = copy(series[1])

        series_type = "E x Rⁿ"
        num_eps = length(series) # the number of runs
        nseries = length(series[1]) # the number of sources / sensors

    end

    if length(scale) != 1
        scales_list = scale
    else
        scales_list = scale * ones(nseries)
    end

    if length(npast) != 1
        npasts_list = npast
    else
        npasts_list = Int.(npast * ones(nseries))
    end

    if length(nfuture) != 1
        nfutures_list = nfuture
    else
        nfutures_list = Int.(nfuture * ones(nseries))
    end

    if length(decay) != 1
        decays_list = decay
    else
        decays_list = decay * ones(nseries)
    end

    if length(localdiff) != 1
        localdiff_list = localdiff
    else
        localdiff_list = localdiff * ones(nseries)
    end

    if qcflags === nothing

        qcflags_list = [nothing for ser in series_list]
    else

        if isa(qcflags, Dict) || isa(qcflags, Tuple)
            qcflags_list = qcflags
        else
            qcflags_list = [qcflags]
        end
    end

    s = 0 # sum of lengths of vectors
    index_map = []

    for i in 1:num_eps

        if series_type == "E x Rⁿ"

            index_map_i = compute_index_map_multiple_sources(series[i], npasts_list, nfutures_list, qcflags_list, take2skipX = take2skipX)

            if i != 1

                for j in 1:nseries
                    series_list[j] = [series_list[j]; series[i][j]]
                end

                index_map = [index_map; index_map_i .+ s]
                s += length(series[i][1])

            else

                index_map = index_map_i
                s += length(series[i][1])
            end

        else

            index_map = compute_index_map_multiple_sources(series_list, npasts_list, nfutures_list, qcflags_list, take2skipX = take2skipX)
        end
    end

    Gx = nothing
    Gy = nothing

    for (ser, sca, npa, nfu, dec, ldiff) in zip(series_list, scales_list, npasts_list, nfutures_list, decays_list, localdiff_list)

        lx, ly = series_xy_logk_indx(ser, sca, npa, nfu, dec, index_map, kernel_params_, ldiff)

        if isnothing(Gx)

            Gx, Gy = lx, ly
        else

            parallel_add_lowtri!(Gx, lx)
            parallel_add_lowtri!(Gy, ly)
        end
    end

    parallel_exp_lowtri!(Gx, nseries)
    parallel_exp_lowtri!(Gy, nseries)

    if series_type == "E x Rⁿ"
        return Gx, Gy, index_map, series_list
    else
        return Gx, Gy, index_map
    end
end

"""
- Index mapping helper - for internal use
- See parameters of compute_index_map_single_source
- This version takes lists of multiple sources as argument.
"""
function compute_index_map_multiple_sources(series_list, npasts_list, nfutures_list, qcflags_list; take2skipX=0)

    #= Each source may have its own NaN patterns and the computed
    index maps do not match. => Compute these index maps for all
    heterogenous sources and then retain only these indices that
    are common to all sources.
    Then, pass that global index map to the _logk function =#

    max_npast = maximum(npasts_list)
    max_nfuture = maximum(nfutures_list)

    valid_map = nothing

    for (ser, npa, nfu, qc) in zip(series_list, npasts_list, nfutures_list, qcflags_list)

        skip_start = max_npast - npa
        skip_end = max_nfuture - nfu

        sidxmap = compute_index_map_single_source(ser, npa, nfu, skip_start=skip_start, skip_end=skip_end, qcflags=qc, take2skipX=take2skipX)

        # combine global indices
        # retain only indices that are valid for all sources
        # A NaN in one source prevents the kernel combination
        # TODO: another possibility is to ignore the source, but how?
        # replace by mean? divide by nsource-1 ?

        if valid_map === nothing
            valid_map = sidxmap
        else
            valid_map = intersect(valid_map, sidxmap)
        end
    end

    return valid_map
end

"""
Index mapping helper - for internal use

Parameters
----------
series : list of arrays list of contiguous data blocks

npast : int take that consecutive number of samples for building an observed past.
    This must be an integer, at least 1 value is needed (the present is part of the
        past)

nfuture : int take that consecutive number of samples, just after the past samples,
    for building an observed future. nfuture can be 0, in which case, the function
    only looks for valid pasts. combined with npast==1, the function only looks for
    valid non-NaN values across all data sources

skip_start : int, optional (default 1) number of data to mark as invalid at the
    beginning of each contiguous data block. Useful to align heterogenous data with
    different npast/nfuture.

skip_end : int, optional (default 1) number of data to mark as invalid at the end of
    each contiguous data block. Useful to align heterogenous data with different
    npast/nfuture.

qcflags : list of arrays, optional quality control flags. <= 0 means invalid data.
    optional, but if present there must be one qcflag per sample

take2skipX : int, optional (default 0) performs a special kind of subsampling: two
    consecutive samples are retained, X samples are discarded, then this (X+2)
    sequence repeats. This scheme is designed to preserve consecutive entries for
    building a shift operator while still allowing subsampling of very large series.
    The classic subsampling scheme (take one out of X) can also be applied a priori
    to the original series, but it is equivalent to a (bad) low-pass filtering.
    Then, the shift operator would be computed on consecutive entries of the
    subsampled series, hence at a different time scale. The take2skipX allows to
    still work on the original time scale. Both can be combined (after appropriate
    low-pass filtering).

Returns
-------
concat_idx_map: (N,) array of int Returns the indices of the valid entries, but
    stacking all series passed as argument in a single series. The indices refer to
    valid entries in that global stacked array. Hence, discontiguous data blocks
    generate invalid entries in the global stacked array.
"""
function compute_index_map_single_source(series, npast, nfuture; skip_start = 0, skip_end = 0, qcflags = nothing, take2skipX = 0)

    if npast < 1

        println("npast must be a strictly positive integer")
    end

    # nfuture can be 0
    if nfuture < 0

        @warn("nfuture must be a strictly positive integer")
    end

    if !isa(series, Dict) && !isa(series, Tuple)

        #println("series must be a list of arrays")
        series = [series]
    end

    if qcflags === nothing

        qcflags_list = [nothing for ser in series]

    else

        if isa(qcflags, Dict) || isa(qcflags, Tuple)

            qcflags_list = qcflags

        else

            qcflags_list = [qcflags]
        end
    end

    concat_idx_map = nothing

    nbefore = 0

    for (sidx, s) in enumerate(series)

        # s is an array
        if ndims(s) == 1

            s = reshape(s, :, 1) # turning the vector into a matrix
        end

        n = size(s, 1) # number of rows

        if n < npast + nfuture

            valid_pf = Array{Float64,2}(undef, 1, n)
            valid_pf = fill!(valid_pf, false)

        else

            # valid instantaneous time points
            valid_t = .!isnan.(s[:, sidx])

            if qcflags !== nothing && qcflags_list[sidx] <= nothing

                valid_t = fill!(valid_t, false) # if quality flag is raised the entire sample is invalidated
            end

            valid_t[1:skip_start] = fill!(valid_t[1:skip_start], false)
            valid_t[n-skip_end+1:end] = fill!(valid_t[n-skip_end+1:end], false)

            # valid past/future combinations
            valid_pf = copy(valid_t)

            for i in 1:nfuture

                valid_pf[1:n-i] = valid_pf[1:n-i] .& valid_t[i+1:n]
            end

            for i in 1:npast-1

                valid_pf[i+1:n] = valid_pf[i+1:n] .& valid_t[1:n-i]
            end

            valid_pf[1:npast-1] .= false
            valid_pf[n-nfuture+1:end] .= false

            if take2skipX > 0

                i = npast

                while i < n

                    i += 2
                    if i >= n

                        break
                    end

                    for k in range(take2skipX)

                        valid_pf[i] = false
                        i += 1

                        if i >= n
                            break
                        end
                    end
                end
            end

        end

        valid_idx = (1:n)[valid_pf]

        if concat_idx_map === nothing

            concat_idx_map = valid_idx .+ nbefore
        else

            concat_idx_map = cat(concat_idx_map, valid_idx .+ nbefore, dims=1)
        end

        nbefore += n
    end

    return concat_idx_map
end

"""
    kernel_params_ = set_kernel(;kernel_type = "Gaussian")

# Description
Sets a kernel type, amongst supported kernels

# Keyword Arguments
- `kernel_type::String`:
    - "Gaussian" -> classical Gausian kernel exp(-0.5 * dist^2 / scale^2)
    - "Laplacian" -> classical Laplacian kernel exp(- dist / scale )

# Return Values
- `kernel_params_::Array{Float64, 1}`: 2-dimensional vector
"""
function set_kernel(; kernel_type = "Gaussian")

    kernel_params_ = Array{Float64,1}(undef, 2)

    if typeof(kernel_type) == String

        if kernel_type == "Gaussian"

            kernel_params_[1] = -0.5
            kernel_params_[2] = 2

        elseif kernel_type == "Laplacian"

            kernel_params_[1] = -1
            kernel_params_[2] = 1

        else

            @warn("Invalid kernel type: Only Gaussian and Laplacian kernels are supported")

            kernel_params_[1] = -0.5
            kernel_params_[2] = 2

        end

    else

        @warn("Invalid kernel type: Enter the strings 'Gaussian' and 'Laplacian' are supported")

        kernel_params_[1] = -0.5
        kernel_params_[2] = 2
    end

    return kernel_params_

end

"""
Wrapper for the function sxy_logk
"""
function series_xy_logk_indx(series, scale, npast, nfuture, decay, concat_valid_map, kernel_params_, localdiff)

    if npast <= 1
        factor_r_past = 1
    else
        factor_r_past = exp(log(decay) / (npast - 1.0))
    end

    sum_r_past = 1
    r = 1

    for t in npast-2:-1:0

        r *= factor_r_past
        sum_r_past += r
    end

    # this factor weights the past sequence
    sum_past_factor = kernel_params_[1] / (sum_r_past * scale^kernel_params_[2])

    #---------------------------------------------------------------------------------------

    if nfuture <= 1
        factor_r_future = 1
    else
        factor_r_future = exp(log(decay) / (nfuture - 1.0))
    end

    sum_r_future = 1
    r = 1

    for t in npast+1:npast+nfuture-1

        r *= factor_r_future
        sum_r_future += r
    end

    # this factor weights the future sequence
    sum_future_factor = kernel_params_[1] / (sum_r_future * scale^kernel_params_[2])

    #---------------------------------------------------------------------------------------

    # The job done for each entry in the matrix
    # computes sum of sq diff for past (sx) and future (sy) sequences
    # n is the number of valid (past, future) pairs
    n = length(concat_valid_map)

    sx = Array{Float64,2}(undef, n, n)
    sy = Array{Float64,2}(undef, n, n)

    # Triangular indexing, folded
    # x
    # y y     =>    z z z y y
    # z z z         w w w w x
    # w w w w
    # outer loops can now be parallelized - all have about the same duration
    if n % 2 == 1
        m = n
    else
        m = n - 1
    end
    #
    Threads.@threads for k in (n+1)÷2:n-1

        for l in 0:k-1

            i = k + 1
            j = l + 1

            # from index of (past,future) pairs to index in data series
            î = concat_valid_map[i]
            ĵ = concat_valid_map[j]

            sx[i, j], sy[i, j] = sxy_logk(î, ĵ, series, npast, nfuture,
                localdiff, kernel_params_[2], factor_r_past, factor_r_future, sum_r_past,
                sum_past_factor, sum_future_factor)

            #sx[j, i] = sumx # we only need the lower triangle
            #sy[j, i] = sumy # we only need the lower triangle
        end

        for l in k:m-1

            i = m - k + 1
            j = m - l

            # from index of (past,future) pairs to index in data series
            î = concat_valid_map[i]
            ĵ = concat_valid_map[j]

            sx[i, j], sy[i, j] = sxy_logk(î, ĵ, series, npast, nfuture,
                localdiff, kernel_params_[2], factor_r_past, factor_r_future, sum_r_past,
                sum_past_factor, sum_future_factor)

            #sx[j, i] = sumx # we only need the lower triangle
            #sy[j, i] = sumy # we only need the lower triangle
        end
    end

    return sx, sy
end

"""
The workhorse which computes the Gram matrices
"""
function sxy_logk(î, ĵ, series, npast, nfuture, localdiff,
    kernel_params_2, factor_r_past, factor_r_future,
    sum_r_past, sum_past_factor, sum_future_factor)

    if localdiff == 1

        # weighted average over each past series
        # diff of these => weighted avg of diffs
        r = 1
        delta = zeros(size(series, 2))

        for t in 0:npast-1

            d = series[î-t] .- series[ĵ-t]
            delta += d * r
            r *= factor_r_past
        end

        delta /= sum_r_past

    elseif localdiff == 2
        # value of the "present"
        delta = series[î] - series[ĵ]
    end

    r = 1
    sumx = 0

    for t in 0:npast-1

        d = series[î-t] - series[ĵ-t]

        if localdiff != 0
            d = d - delta
        end

        ds = abs2(d)

        if kernel_params_2 != 2
            ds = ds^(0.5 * kernel_params_2)
        end

        sumx += ds * r
        r *= factor_r_past

    end

    r = 1
    sumy = 0

    for t in 0:nfuture-1

        d = series[î+1+t] - series[ĵ+1+t]

        if localdiff != 0
            d = d - delta
        end

        ds = abs2(d)

        if kernel_params_2 != 2
            ds = ds^(0.5 * kernel_params_2)
        end

        sumy += ds * r
        r *= factor_r_future
    end

    return sumx * sum_past_factor, sumy * sum_future_factor
end

"""
In the interest of saving time, we only add the lower triangles of two matrices together
"""
function parallel_add_lowtri!(total, mat)

    N = size(mat, 1)
    if N % 2 == 1
        M = N
    else
        M = N - 1
    end

    # outer loops can be parallelized - all have about the same duration
    Threads.@threads for k in (N+1)÷2:N-1

        for l in 0:k-1

            i = k + 1
            j = l + 1
            total[i, j] += mat[i, j]
        end

        for l in k:M-1

            i = M - k + 1
            j = M - l
            total[i, j] += mat[i, j]
        end
    end

    Threads.@threads for d in 1:N

        total[d, d] += mat[d, d]
    end

    return total
end

"""
Iterates through the lower triangular part of the matrix, exponentiating the matrix and
duplicating the results the the upper triangle, i.e. restoring the upper part.
"""
function parallel_exp_lowtri!(mat, dims)

    N = size(mat, 1)

    if N % 2 == 1
        M = N
    else
        M = N - 1
    end

    invdims = 1 / dims

    # outer loops can be parallelized - all have about the same duration
    Threads.@threads for k in (N+1)÷2:N-1

        for l in 0:k-1

            i = k + 1
            j = l + 1

            e = exp(mat[i, j] * invdims)

            mat[i, j] = e
            mat[j, i] = e

        end

        for l in k:M-1

            i = M - k + 1
            j = M - l

            e = exp(mat[i, j] * invdims)

            mat[i, j] = e
            mat[j, i] = e
        end
    end

    Threads.@threads for d in 1:N

        mat[d, d] = exp(mat[d, d] * invdims)
    end

    return nothing
end

"""
    Gs, (embedder) = embed_states(Gx, Gy; ϵ = 1e-8, normalize = true, return_embedder = false)

Compute a similarity matrix for the embedded causal states, seen as distributions P(Y|X),
using the conditional mean embedding.

# Arguments:
- `Gx::Array{Float64, 2}`: A square symmetrical Gram matrix of past events. A similarity
  matrix of pasts X. Gx[i,j] = k(xᵢ , xⱼ) with kˣ a reproducing kernel for X.
- `Gy::Array{Float64, 2}`: A square symmetrical Gram matrix of future events
- `ϵ::Float64`: Amount of regularization. The theory requires a regularizer and shows that
  the regularized estimator is consistent in the limit of N -> infinity. In practice, using
  a too small regularizer may cause divergence, too large causes innaccuracies.
- `normalize::Bool`: normalize: whether to renormalize the returned similarity matrix, so
  that each entry along the diagonal is 1.
- `return_embedder::Bool`: whether to return an embedder object for new, unknown data. With
  normalized kernels, such as used in series_Gxy, the state vectors should also be
  normalized. Also, theoretically, that matrix should be positive definite. In practice,
  numerical innacuracies and estimating from finite samples may destroy both previous
  properties. Renormalization is performed by default, but if positive definiteness issues
  happens, try using a larger regularizer.

# Return values:
- `Gs::Array{Float64, 2}`: A similarity matrix between every causal state. Entries Gs[i,j]
  can be seen as inner products between states Sᵢ and Sⱼ.
- `(embedder)::Array{Float64, 2}`: A matrix used for embedding new kˣ kernel similarity
- vectors.
"""
function embed_states(Gx, Gy; ϵ = 1e-10, normalize = true, return_embedder = true, centre_embedder = false)

    N = size(Gx, 1)

    if !centre_embedder

        #Ω = (Gx + N*ϵ*I) \ Gx

        Ω = copy(Gx) # this is our weight matrix
        ldiv!(cholesky(Gx + N*ϵ*I), Ω)

    else

        #Ω = (H*Gˣ + N*ϵ*I) \ (H*Gx)

        H = I - (1/N)*ones(N, N)# centring matrix

        Ω = H*Gx # this is will become our centred weight matrix
        Ω = 0.5*(Ω .+ transpose(Ω)) # H*Gx should be positive definite, but numerical instabilities
        ldiv!(factorize(Ω + N*ϵ*I), Ω)

        # the results between centring and not, appear to be comparable, but without is faster
    end

    Ω = 0.5*(Ω .+ transpose(Ω)) # should not be needed

    embedder = Ω * Gy

    Gs = embedder * Ω

    if normalize

        #= This normalization is especially useful for avoiding numerical
        errors later in the Gs eigendecomposition. But Gs should already
        have nearly 1s on the diagonal to start with. For embedding new
        vectors, this refinement is probably overkill
        TODO: tests if this is really necessary for the embedder,
        knowing the distributions will be normalized later on.
        Tests: actually, this impacts a bit the numerical resolution
        when expressing the embedded Ks as a combination of Gs lines
        option 1 : all the weights on the left side:
        embedder /= Gs.diagonal()[:,None]
        if applied to Gs, this would yield 1s on the diagonal
        but some asymetry
        Tests: this options yields some inaccuracies
        option 2 : use the same normalization as GS:
        but that normalisation is not applicable on the right
        to new kvecs
        Tests: this options yields some inaccuracies, but less than option 1 =#

        d = (1) ./ (sqrt.(diag(Gs)))
        embedder = embedder .* d

        # Now Gs

        Gs = Gs .* d
        Gs = Gs .* transpose(d)
        Gs = Symmetric(Gs)

        # just to be extra safe, avoid floating-point residual errors
        Gs[diagind(Gs)] .= 1.0

    end

    if return_embedder

        return Gs, embedder
    end

    return Gs
end

function series_newKx(new_series, ref_series, ref_index_map, scale, npast; decay = 1, localdiff = 0, new_index_map = nothing, kernel_type="Gaussian")
    """
    Computes the log of kernel vector evaluations between each past sequence from the new series and the reference sequences.

    This is similar to series_Gxy in that it computes a similarity matrix (or the log of it), but here the similarity is computed with respect to new data values, and only for Gx (past sequences)

    The new_series_map parameter can be computed using compute_index_map, or set to None to be automatically computed. See series_Gxy and compute_index_map for other arguments/doc.

    """

    kernel_params_ = set_kernel(kernel_type = kernel_type)

    if size(new_series[1], 1) == 1
        new_series_list = [new_series]
    else
        new_series_list = new_series
    end

    if size(ref_series[1], 1) == 1
        ref_series_list = [ref_series]
    else
        ref_series_list = ref_series
    end

    nseries = size(ref_series_list, 1) # the number of sources / sensors

    if length(scale) != 1
        scales_list = scale
    else
        scales_list = scale * ones(nseries)
    end

    if length(npast) != 1
        npasts_list = npast
    else
        npasts_list = Int.(npast * ones(nseries))
    end

    if length(decay) != 1
        decays_list = decay
    else
        decays_list = decay * ones(nseries)
    end

    if length(localdiff) != 1
        localdiff_list = localdiff
    else
        localdiff_list = localdiff * ones(nseries)
    end

    qcflags_list = [nothing for ser in new_series_list]

    if new_index_map === nothing
        new_index_map = compute_index_map_multiple_sources(new_series_list, npasts_list, Int.(zeros(nseries)), qcflags_list)
    end

    total_lx = nothing

    for (nser, rser, sca, npa, dec, ldiff) in zip(new_series_list, ref_series_list, scales_list, npasts_list, decays_list, localdiff_list)

        lx = series_newx_logk(nser, new_index_map, rser, ref_index_map,
        sca, npa, kernel_params_, decay = dec, localdiff = ldiff)

        if total_lx === nothing
            total_lx = lx
        else
            total_lx += lx
        end
    end

    invdims = 1/nseries

    total_lx = invdims*total_lx

    return exp.(total_lx)
end

function series_newx_logk(new_series, new_index_map, ref_series, ref_index_map,
    scale, npast, kernel_params_; decay = 1, localdiff = 0)

    if npast <= 1
        factor_r_past = 1
    else
        factor_r_past = exp(log(decay) / (npast - 1.0))
    end

    sum_r_past = 1
    r = 1

    for t in npast-2:-1:0

        r *= factor_r_past
        sum_r_past += r
    end

    # this factor weights the past sequence
    sum_past_factor = kernel_params_[1] / (sum_r_past * scale^kernel_params_[2])

    nr = length(ref_index_map)
    ns = length(new_index_map)

    sx = Array{Float64,2}(undef, nr, ns)

    #for idx, i in enumerate(ref_series_map):
    #  for jdx, j in enumerate(new_series_map):

    Threads.@threads for i_rs in 0:nr*ns-1

        i = (i_rs ÷ ns)
        j = i_rs - (i * ns)

        # from index of (past,future) pairs to index in data series

        î = ref_index_map[i+1]
        ĵ = new_index_map[j+1]

        sx[i+1, j+1] = sx_logk(î, ĵ, new_series, ref_series, npast, factor_r_past, sum_r_past,
         sum_past_factor, kernel_params_[2], localdiff)

    end

    return sx

end

function sx_logk(î, ĵ, new_series, ref_series, npast, factor_r_past, sum_r_past,
    sum_past_factor, kernel_params_2, localdiff)

    if localdiff == 1

        # weighted average over each past series
        # diff of these => weighted avg of diffs
        r = 1
        delta = zeros(size(ref_series, 2))

        for t in 0:npast-1

            d = ref_series[î-t] .- new_series[ĵ-t]
            delta += d * r
            r *= factor_r_past
        end

        delta /= sum_r_past

    elseif localdiff == 2
        # value of the "present"
        delta = ref_series[î] - new_series[ĵ]
    end

    r = 1
    sumx = 0

    #= @show(new_series[ĵ-npast+1])
    @show(ĵ-npast+1)
    @show(ĵ)
    adf =#

    for t in 0:npast-1

        d = ref_series[î-t] - new_series[ĵ-t]

        if localdiff != 0
            d = d - delta
        end

        ds = abs2(d)

        if kernel_params_2 != 2
            ds = ds^(0.5 * kernel_params_2)
        end

        sumx += ds * r
        r *= factor_r_past

    end

    return sumx * sum_past_factor
end

function embed_Kx(Kx, Gx, embedder, ϵ = 1e-8; alg = :ldiv)
    """
    Construct a similarity vector in state space, from a similarity vector in X space and using the embbeder returned by embed_states

    Arguments:
        Kx: a similarity vector in X space, such as returned by series_newKx.
        Gx: a similarity matrix of pasts X.
        embedder: The embedder returned by embed_states
        ϵ: amount of regularization. See embed_states

    Returns:
        Ks: the similarity vector, in state space
    """

    if alg == :nnls # this is more accurate

        omega = nonneg_lsq(Gx + ϵ*I, Kx, alg = :nnls) # is this actually

        Ks = (embedder * omega)

    elseif alg == :ldiv # this is faster

        #= Ω = (Gx + ϵ*I) \ Kx
        Ks = embedder * Ω =#

        Ω = copy(Kx) # this is our weight matrix
        ldiv!(cholesky(Gx + ϵ*I), Ω) # Todo: isn't there an N missing?
        Ks = embedder * Ω

    end

    return Ks
end

function Gaussian_kernel(x₁, x₂, η; dims = -1)

    if dims == -1

        k_x₁_x₂ = exp(-η*sum(Base.abs2, x₁ - x₂))

        return k_x₁_x₂
    else

        k_x₁_x₂ = exp(-η*sum(Base.abs2, x₁ - x₂))

        return k_x₁_x₂*Matrix(I, dims, dims)
    end
end

function Distance_kernel(x₁, x₂, s)

    return s*sum(abs2, x₁ - x₂)

end

function exp_mat(D, η; derivative = false)

    N = size(D, 1)

    if N % 2 == 1
        M = N
    else
        M = N - 1
    end

    G = Array{Float64,2}(undef, N, N)
    if derivative
        ∂G = Array{Float64,2}(undef, N, N)
    end

    # outer loops can be parallelized - all have about the same duration

    if derivative

        Threads.@threads for k in (N+1)÷2:N-1

            for l in 0:k-1

                i = k + 1
                j = l + 1

                e = exp(-η*D[i, j])

                G[i, j] = e
                G[j, i] = e

                ∂G[i, j] = -D[i, j]*e
                ∂G[j, i] = -D[j, i]*e

            end

            for l in k:M-1

                i = M - k + 1
                j = M - l

                e = exp(-η*D[i, j])

                G[i, j] = e
                G[j, i] = e

                ∂G[i, j] = -D[i, j]*e
                ∂G[j, i] = -D[j, i]*e
            end
        end

        Threads.@threads for d in 1:N

            G[d, d] = exp(-η*D[d, d])
            ∂G[d, d] = -D[d, d]*G[d, d]
        end

        return G, ∂G

    else

        Threads.@threads for k in (N+1)÷2:N-1

            for l in 0:k-1

                i = k + 1
                j = l + 1

                e = exp(-η*D[i, j])

                G[i, j] = e
                G[j, i] = e

            end

            for l in k:M-1

                i = M - k + 1
                j = M - l

                e = exp(-η*D[i, j])

                G[i, j] = e
                G[j, i] = e
            end
        end

        Threads.@threads for d in 1:N

            G[d, d] = exp(-η*D[d, d])
        end

        return G
    end

end

function Compatible_kernel(z₁, z₂, η; kernel = Gaussian_kernel, Σinv = nothing)

    dims = length(z₁[2])

    k_z₁_z₂ = transpose(kernel(z₁[1], z₂[1], η, dims = dims)*Σinv*(z₁[2] - z₁[3]))*(Σinv*(z₂[2] - z₂[3]))

    #k_z₁_z₂ = transpose(transpose(kernel(x₁, x₂, ζ, dims = dims))*Σinv*(a₁ - μ₁))*(Σinv*(a₂ - μ₂))

    return k_z₁_z₂
end

function Gramian(data, η = 1; kernel = Gaussian_kernel, Σinv = nothing, derivative = false)

    if !isa(data, Tuple)

        if isa(data, Vector)
            data = transpose(data)
        end

        n = size(data, 2)
    else

        n = size(data[1], 2)
    end

    G = Array{Float64,2}(undef, n, n)
    if derivative
        ∂G = Array{Float64,2}(undef, n, n)
    end

    # Triangular indexing, folded
    # x
    # y y     =>    z z z y y
    # z z z         w w w w x
    # w w w w
    # outer loops can now be parallelized - all have about the same duration
    if n % 2 == 1
        m = n
    else
        m = n - 1
    end
    #Threads.@threads

    if derivative

        Threads.@threads for k in (n+1)÷2:n-1

            for l in 0:k-1

                i = k + 1
                j = l + 1

                G[i, j] = kernel(data[:,i], data[:,j], η)
                ∂G[i, j] = -G[i, j]*sum(abs2, data[:,i] - data[:,j])

                G[j, i] = G[i, j]
                ∂G[j, i] = ∂G[i, j]

            end

            for l in k:m-1

                i = m - k + 1
                j = m - l

                G[i, j] = kernel(data[:,i], data[:,j], η)
                ∂G[i, j] = -G[i, j]*sum(abs2, data[:,i] - data[:,j])

                G[j, i] = G[i, j]
                ∂G[j, i] = ∂G[i, j]

            end

        end

        Threads.@threads for i in 1:n

            G[i, i] = kernel(data[:,i], data[:,i], η)
            ∂G[i, i] = 0

        end

        return G, ∂G

    else

        Threads.@threads for k in (n+1)÷2:n-1

            for l in 0:k-1

                i = k + 1
                j = l + 1

                if kernel == Compatible_kernel

                    z₁ = (data[1][:,i], data[2][:,i], data[3][:,i])
                    z₂ = (data[1][:,j], data[2][:,j], data[3][:,j])

                    G[i, j] = kernel(z₁, z₂, η; kernel = Gaussian_kernel, Σinv = Σinv)

                else

                    G[i, j] = kernel(data[:,i], data[:,j], η)

                end

                #if (i == 1) && (j == 1)

                G[j, i] = G[i, j]

            end

            for l in k:m-1

                i = m - k + 1
                j = m - l

                if kernel == Compatible_kernel

                    z₁ = (data[1][:,i], data[2][:,i], data[3][:,i])
                    z₂ = (data[1][:,j], data[2][:,j], data[3][:,j])

                    G[i, j] = kernel(z₁, z₂, η; kernel = Gaussian_kernel, Σinv = Σinv)

                else
                    G[i, j] = kernel(data[:,i], data[:,j], η)
                end

                G[j, i] = G[i, j]

            end

        end

        Threads.@threads for i in 1:n

            if kernel == Compatible_kernel

                z₁ = (data[1][:,i], data[2][:,i], data[3][:,i])

                G[i, i] = kernel(z₁, z₁, η; kernel = Gaussian_kernel, Σinv = Σinv)

            else
                G[i, i] = kernel(data[:,i], data[:,i], η)
            end

        end

        return G
    end
end

function KLSTD(data, data_ids, C, rewards, η, γ; kernel = Gaussian_kernel, ϵ = 1e-6, Σinv = nothing)

    if kernel == Gaussian_kernel

        D = size(C, 2)
    else

        D = size(C[1], 2)
    end

    N = length(data_ids)

    ϑₖ = Array{Float64,1}(undef, D) # current feature vector
    ϑₙ  = Array{Float64,1}(undef, D) # next feature vector

    A = zeros(D, D)
    b = zeros(D)

    for k in 1:N

        for d in 1:D

            if kernel == Gaussian_kernel

                ϑₖ[d] = kernel(C[:,d], data[:, data_ids[k]], η)
                ϑₙ[d] = kernel(C[:,d], data[:, data_ids[k] + 1], η)

            else

                ϑₖ[d] = kernel((C[1][:,d], C[2][:,d], C[3][:,d]),
                                (data[1][:, data_ids[k]], data[2][:, data_ids[k]], data[3][:, data_ids[k]]),
                                η, Σinv = Σinv)
                ϑₙ[d] = kernel((C[1][:,d], C[2][:,d], C[3][:,d]),
                                (data[1][:, data_ids[k] + 1], data[2][:, data_ids[k] + 1], data[3][:, data_ids[k] + 1]),
                                η, Σinv = Σinv)

            end
        end

        A += ϑₖ*transpose(ϑₖ - γ*ϑₙ)
        b += ϑₖ*rewards[data_ids[k]]

    end

    if N > 1
        Λ = (A + ϵ*I) \ b
    else
        Λ = b
    end

    return transpose(Λ)
end

function function_approximation(x, Λ, C, η; kernel = Gaussian_kernel, Σinv = nothing, β = zeros(size(Λ,1)))

    if kernel == Gaussian_kernel

        #= if isa(Λ, Vector)

            N = length(Λ)
            ϑ = Array{Float64,1}(undef, N) # feature vector
            f_x = 0

            for n in 1:N

                if isa(x, Vector)
                    ϑ[n] = kernel(C[:,n], x, η)
                else
                    ϑ[n] = kernel(C[:,n], [x], η)
                end
            end

            f_x = (transpose(Λ)*ϑ)[1] .+ β

        else

            N = size(Λ, 2)
            ϑ = Array{Float64,1}(undef, N) # feature vector
            f_x = zeros(size(Λ, 1))

            for n in 1:N

                if isa(x, Vector)
                    ϑ[n] = kernel(C[:,n], x, η)
                else
                    ϑ[n] = kernel(C[:,n], [x], η)
                end

            end

            f_x = Λ*ϑ .+ β
        end =#

        N = size(Λ, 2)
        ϑ = Array{Float64,1}(undef, N) # feature vector
        f_x = zeros(size(Λ, 1))

        if isa(x, Vector)

            for n in 1:N

                ϑ[n] = kernel(C[:,n], x, η)

            end

        else

            for n in 1:N

                ϑ[n] = kernel(C[:,n], [x], η)

            end
        end

        f_x = Λ*ϑ .+ β

    else

        N = size(Λ, 2)
        ϑ = Array{Float64,1}(undef, N) # feature vector
        f_x = zeros(size(Λ, 1))

        for n in 1:N

            z₁ = (C[1][:,n], C[2][:,n], C[3][:,n])

            ϑ[n] = kernel(z₁, x, η; kernel = Gaussian_kernel, Σinv = Σinv)

        end

        f_x = Λ*ϑ .+ β
    end

    if size(Λ, 1) == 1
        return f_x[1]
    else
        return f_x
    end

end

function SGA(data, data_ids, T, Z, actor_η, Σinv, critic_Λ, critic_C, critic_η)

    M = size(data[1], 2)

    N = length(data_ids)

    ∇J = zeros(N)
    Q = Array{Float64,1}(undef, T)

    for z in 1:Z

        ids_t = collect(M -z*T + 1: M - (z-1)*T)

        for k in 1:T

            Q[k] = function_approximation((data[1][:, ids_t[k]], data[2][:, ids_t[k]], data[3][:, ids_t[k]]),
                                        critic_Λ, critic_C, critic_η,
                                        kernel = Compatible_kernel, Σinv = Σinv)

            #= Q[k] = function_approximation(vcat(data[1][:, ids_t[k]], data[2][:, ids_t[k]]),
                                        critic_Λ, critic_C, critic_η) =#

        end

        for n in 1:N
            for k in 1:T

                aₖ = data[2][:, ids_t[k]]
                fₖ = data[3][:, ids_t[k]]
                ∇J[n] += (Q[k]*Gaussian_kernel(data[1][:, data_ids[n]], data[1][:, ids_t[k]], actor_η)*Σinv*(aₖ - fₖ))[1]

            end
        end
    end

    return transpose(∇J).*1/Z#, G
end

function OMP(data, η, Y, δ = 0.1; kernel = Gaussian_kernel, N = nothing, ϵ = 1e-6, Σinv = nothing, PRESS = false, sparsity = 1.0)

    if isa(data, Vector)
        data = transpose(data)
    end

    if size(Y,2) == 1
        Y = transpose(Y)
    end

    if kernel == Gaussian_kernel

        G = Gramian(data, η; kernel = kernel)
        G += ϵ*I
        U = copy(G)

        d = size(data,1)

    elseif kernel == Compatible_kernel

        G = Gramian(data, η; kernel = kernel, Σinv = Σinv) + ϵ*I
        U = copy(G)

        d = size(data[1], 1)
    end

    T = size(U,1)
    m = size(Y,1)
    sparsity = clamp(sparsity, 0.0, 1.0)

    for i in 1:T

        U[:,i] = U[:,i]/sum(abs2, U[:,i])

    end

    error_ic = sqrt(sum(sum(abs2, Y, dims = 1))/T) #sqrt(norm(Y)/m)

    if error_ic == 0 error_ic = 1e-6 end

    if isnothing(N)

        Φ = Array{Float64,2}(undef, T, 1)
        ids = Array{Int64,1}(undef, 1)
        ids_vec = ones(T)
        β = zeros(m)

        if isa(data,Tuple)

            C = (Array{Float64,2}(undef, d, 1),
                Array{Float64,2}(undef, m, 1),
                Array{Float64,2}(undef, m, 1))
        else

            C = Array{Float64,2}(undef, d, 1)
        end

        N = 0

        error = 100

        while error > δ

            N += 1

            if N > 1

                vector = diag(transpose(U)*transpose(R)*(R*(U)))

                index = argmax(vector .* ids_vec)

                ids = vcat(ids, index)
                ids_vec[index] = 0
                Φ = hcat(Φ, G[:,ids[N]])

                if isa(data, Tuple)

                    C[1] = hcat(C, copy(data[1][:,ids[N]]))
                    C[2] = hcat(C, copy(data[2][:,ids[N]]))
                    C[3] = hcat(C, copy(data[3][:,ids[N]]))
                else

                    C = hcat(C, copy(data[:,ids[N]]))
                end

            else

                R = Y

                index = argmax(diag(transpose(U)*transpose(R)*(R*(U))))

                ids[1] = index
                ids_vec[index] = 0
                Φ[:,1] = G[:,ids[1]]

                if isa(data,Tuple)

                    C[1][:,1] = copy(data[1][:,ids[1]])
                    C[2][:,1] = copy(data[2][:,ids[1]])
                    C[3][:,1] = copy(data[3][:,ids[1]])
                else

                    C[:,1] = copy(data[:,ids[1]])
                end

            end

            if rand() < sparsity || N == 1

                #Λ[:,1:N] = Y*pinv(transpose(Φ[:,1:N]))
                #Λ = Y*inv(Φ*transpose(Φ) + ϵ*I)*Φ
                #Λ = transpose(Φ \ transpose(Y))
                D = vcat(hcat(Φ, ones(T)), hcat(ones(1,N), 0))
                Λ_β = transpose(D \ transpose(hcat(Y, zeros(m))))
                Λ = Λ_β[:,1:end-1]
                β = Λ_β[:,end]

            else

                Λ = hcat(Λ, R*U[:,ids[N]])

            end

            Yh = Λ*transpose(Φ) + β*ones(1,T)
            R = Y - Yh

            error = sqrt(sum(sum(abs2, R, dims = 1))/T) #100*norm(R)/error_ic

            if N+1 == T
                break;
            end
        end

        if sparsity != 1

            D = vcat(hcat(Φ, ones(T)), hcat(ones(1,N), 0))
            Λ_β = transpose(D \ transpose(hcat(Y, zeros(m))))
            Λ = Λ_β[:,1:end-1]
            β = Λ_β[:,end]

            Yh = Λ*transpose(Φ) + β*ones(1,T)
            R = Y - Yh

            # using root mean square error
            error = sqrt(sum(sum(abs2, R, dims = 1))/T) #100*norm(R)/error_ic
        end

    elseif N < T

        N = clamp(N, 1, T)

        Φ = Array{Float64,2}(undef, T, N)

        if isa(data,Tuple)

            C = (Array{Float64,2}(undef, d, N),
                Array{Float64,2}(undef, m, N),
                Array{Float64,2}(undef, m, N))
        else

            C = Array{Float64,2}(undef, d, N)
        end

        ids = Array{Int64,1}(undef, N)
        ids_vec = ones(T)
        Λ = Array{Float64,2}(undef, m, N)
        β = zeros(m)

        for n in 1:N

            if n > 1

                R = Y - Λ[:,1:n-1]*transpose(Φ[:,1:n-1]) - β*ones(1,T)

                vector = diag(transpose(U)*transpose(R)*(R*(U)))

                index = argmax(vector .* ids_vec)

            else

                R = Y

                index = argmax(diag(transpose(U)*transpose(R)*(R*(U))))
            end

            ids_vec[index] = 0
            ids[n] = index
            Φ[:,n] = G[:,ids[n]]

            if isa(data,Tuple)

                C[1][:,n] = copy(data[1][:,ids[n]])
                C[2][:,n] = copy(data[2][:,ids[n]])
                C[3][:,n] = copy(data[3][:,ids[n]])
            else

                C[:,n] = copy(data[:,ids[n]])
            end

            if rand() < sparsity || n == N

                #Λ[:,1:n] = Y*pinv(transpose(Φ[:,1:n]))
                #Λ[:,1:n] = Y*inv(Φ[:,1:n]*transpose(Φ[:,1:n]) + ϵ*I)*Φ[:,1:n]
                #Λ[:,1:n] = transpose(Φ[:,1:n] \ transpose(Y))

                D = vcat(hcat(Φ[:,1:n], ones(T)), hcat(ones(1,n), 0))
                Λ_β = transpose(D \ transpose(hcat(Y, zeros(m))))
                Λ[:,1:n] = Λ_β[:,1:end-1]
                β = Λ_β[:,end]

            else

                α = R*U[:,ids[n]]
                if typeof(α) != Vector{Float64}
                    Λ[:,n] = [α]
                else
                    Λ[:,n] = α
                end

            end

        end

        Yh = Λ*transpose(Φ) + β*ones(1,T)
        R = Y - Yh
        error = sqrt(sum(sum(abs2, R, dims = 1))/T) #100*norm(R)/error_ic

    elseif N >= T

        C = copy(data)

        N = T

        Λ, β, Ch, ρ = kernel_regression(G, Y; N = N)

        Φ = G
        Yh = Λ*transpose(Φ) + β*ones(1,T)
        R = Y - Yh
        error = sqrt(sum(sum(abs2, R, dims = 1))/T) #100*norm(R)/error_ic

    end

    if PRESS

        if Φ == G

            S = inv(Ch.L)
            Sm = -1*transpose(S*ones(N))*S*ones(N)

            Dinv = Array{Float64,1}(undef, N) #diag(transpose(S)*S) .+ ρ.^2/Sm
            for i in 1:N
                Dinv[i] = transpose(S[:,i])*S[:,i] + ρ[i]^2/Sm
            end

            Press = 0.5*(sum(abs2, transpose(Λ)./(Dinv), dims = 1)) # predictive residual sum of squares

        else

            #H = transpose(Φ*inv(transpose(Φ)*Φ)*transpose(Φ)) # influence matrix
            #h = 1 .- diag(H)
            h = 1 .- diag(transpose(Φ*inv(transpose(Φ)*Φ)*transpose(Φ))) # where is β term? wrong equation
            Press = 0.5*sum(abs2, transpose(R)./h, dims = 1)

        end

        if m > 1
            Press = vec(Press)
        else
            Press = Press[1]
        end

        return Λ, C, β, error, Press

    else

        return Λ, C, β, error

    end

end

function Press(Ch, Λ, ρ, N = size(Ch,2))

    S = inv(Ch.L)
    Sm = -1*transpose(S*ones(N))*S*ones(N)

    Dinv = Array{Float64,1}(undef, N) #diag(transpose(S)*S) .+ ρ.^2/Sm
    for i in 1:N
        Dinv[i] = transpose(S[:,i])*S[:,i] + ρ[i]^2/Sm
    end

    P = 0.5*(sum(abs2, transpose(Λ)./(Dinv), dims = 1)) # predictive residual sum of squares

    return P[1]
end

function ALD(data, data_ids, μ, η; kernel = Gaussian_kernel, ϵ = 1e-6, Σinv = nothing)

    if kernel == Gaussian_kernel

        G = Gramian(data[:, data_ids], η; kernel = kernel)

        C = Array{Float64,2}(undef, size(data, 1), 1)
        C[:,1] = data[:, data_ids[1]]

        ids = Array{Int64,1}(undef, 1)
        ids[1] = 1

    else

        G = Gramian((data[1][:, data_ids], data[2][:, data_ids], data[3][:, data_ids]), η;
        kernel = kernel, Σinv = Σinv)

        C_state = Array{Float64,2}(undef, size(data[1], 1), 1)
        C_action = Array{Float64,2}(undef, size(data[2], 1), 1)
        C_mean = Array{Float64,2}(undef, size(data[3], 1), 1)

        C_state[:, 1] = data[1][:, data_ids[1]]
        C_action[:, 1] = data[2][:, data_ids[1]]
        C_mean[:, 1] = data[3][:, data_ids[1]]

        ids = Array{Int64,1}(undef, 1)
        ids[1] = 1

    end

    N = length(data_ids)

    for k in 2:N

        cₖ = (G[ids, ids] + ϵ*I) \ G[ids, k] #inv(G[ids,ids])*G[ids,k]
        δₖ = G[k, k] - transpose(G[ids, k])*cₖ

        if δₖ > μ

            if kernel == Gaussian_kernel

                C = hcat(C, data[:, data_ids[k]])
                ids = vcat(ids, k)

            else

                C_state = hcat(C_state, data[1][:, data_ids[k]])
                C_action = hcat(C_action, data[2][:, data_ids[k]])
                C_mean = hcat(C_mean, data[3][:, data_ids[k]])

                ids = vcat(ids, k)
            end
        end
    end

    if kernel == Gaussian_kernel

        return C, 100*length(ids)/N, G[ids,:], ids
    else
        return (C_state, C_action, C_mean), 100*length(ids)/N, G[ids, ids], ids
    end
end

function kernel_regression(G, Y, ϵ = 0; N = size(G,2))

    if ϵ != 0
        G += ϵ*I
    end

    if size(Y,2) == 1
        Y = transpose(Y)
    end

    m = size(Y,1)

    ρ = ones(N)
    if m > 1
        v = Matrix(transpose(copy(Y)))
    else
        v = vec(copy(transpose(Y)))
    end

    Ch = cholesky(G)
    ldiv!(Ch, ρ)
    ldiv!(Ch, v)

    if m > 1
        β = (ones(1,N)*v)./(ones(1,N)*ρ)
        Λ = transpose(v - ρ*β)
        β = transpose(β)
    else
        β = (ones(1,N)*v)/(ones(1,N)*ρ)
        Λ = transpose(v - ρ*β)
    end

    return Λ, β, Ch, ρ
end

function Weights_and_Gradients(Y, D, ϵ, η)

    G, ∂G = exp_mat(D, η; derivative = true)
    N = size(G,1)

    #G = 0.5*(G .+ transpose(G))
    #∂G = 0.5*(∂G .+ transpose(∂G))

    Λ, β, Ch = kernel_regression(G, Y, ϵ; N = N)

    #---------------------------------------------------------------------------------------
    # gradient calculation

    ∇P = zeros(2) # error gradient

    S = inv(Ch.L)
    N = size(S,1)
    invG = transpose(S)*S
    invSm = -1/(transpose(ones(N))*invG*ones(N))

    #D = vcat(hcat(G, ones(N)), hcat(ones(1,N), 0))

    Dinv = vcat(hcat(invG + invG*ones(N)*invSm*ones(1,N)*invG, -invG*ones(N)*invSm), hcat(-invSm*ones(1,N)*invG, invSm))

    ∂D_∂η = vcat(hcat(∂G, zeros(N)), zeros(1, N+1))

    ∂Λ_∂ϵ = (-Dinv*transpose(hcat(Λ, β)))[1:end-1]
    ∂Λ_∂η = (-Dinv*∂D_∂η*transpose(hcat(Λ, β)))[1:end-1]

    dii = 1 ./ (diag(Dinv)[1:end-1])

    ri = vec(Λ) .* dii
    t2ϵ = (∂Λ_∂ϵ .* dii)
    t2η = (∂Λ_∂η .* dii)
    t3 = ri .* dii
    t4ϵ = diag(-Dinv[1:end-1,1:end-1]*Dinv[1:end-1,1:end-1])
    t4η = diag((-Dinv*∂D_∂η*Dinv)[1:end-1,1:end-1])

    ∇P[1] = sum(ri.*(t2ϵ .- (t3.*t4ϵ)))*ϵ*log(2)
    ∇P[2] = sum(ri.*(t2η .- (t3.*t4η)))*η*log(2)
    # now for the error

    P = 0.5*(sum(abs2, ri, dims = 1)) # predictive residual sum of squares

    return ∇P, P[1], Λ, β
end

function CGD(data, Y; ϵ = 1e-6, η = 10.0, ps = 0.05, Smax = nothing, ε = 10e-3, λ = 1, τ = 0.01, λmax = 1e12, λmin = 0, nmax = 100, D = nothing, ϵ_max = 1e-3, η_max = 100)

    # ps = 0.1 or ps = 0.05
    # τ around 1.0 is good
    # A Marquardt Algorithm for Choosing the Step-size in Backpropagation Learning with Conjugate gradients

    θϵ_max = log(2, ϵ_max)
    θη_max = log(2, η_max)

    θn = [log(2,ϵ); log(2,η)]

    if isnothing(D)

        D = Gramian(data, kernel = Distance_kernel)

        ∇Pn, Pn, Λ, β = Weights_and_Gradients(Y, D, ϵ, η)

    else

        ∇Pn, Pn, Λ, β = Weights_and_Gradients(Y, D, ϵ, η)
    end

    P = Pn

    if isnothing(Smax)

        Smax = length(θn)

    end

    εn = ε
    λn = λ

    gn = ∇Pn
    sn = -gn # initial search direction. Starts in direction of steepest descent

    success = true
    S = 0

    γn = 0
    κn = 0
    μn = 0
    σn = 0

    converged = false

    for n in 1:nmax

        # Step 1: calculate first and second order directional derivatives rdfd  qrefqfqwef
        #=
            Apart from the initial cycle, this step is only executed if the last cycle
            succeeded in error reduction. Otherwise no change in the weight vector has been
            made and this information is already known.
        =#
        if success

            μn = transpose(sn)*gn # directional gradient

            if μn >= 0

                sn = -gn
                μn = transpose(sn)*gn
                S = 0

            end

            κn = norm(sn)
            σn = εn/sqrt(κn) # renormalisation ensures uniform scaling for varying directions and gradients

            θt = θn + σn*sn

            if (θt[1] > θϵ_max || θt[2] > θη_max  || θt[2] < -15)

                ϵt = ϵ
                ηt = η # should not be 100

            else

                ϵt = 2^θt[1]
                ηt = 2^θt[2]
            end

            ∇Pt, _, _, _ = Weights_and_Gradients(Y, D, ϵt, ηt)
            γn = (transpose(sn)*(∇Pt - ∇Pn))/σn # directional curvature

        end

        # Step 2: increase the working curvature
        δn = γn + λn*κn

        # Step 3: make δn positive and increase λn
        if δn <= 0
            δn = λn*κn
            λn = λn - γn/κn
        end

        # Step 4: calculate step size and adapt ε
        αn = -μn/δn

        εnx = εn*(αn/σn)^ps

        # Step 5: calculate the comparison ratio
        θnx = θn + αn*sn

        if (θnx[1] > θϵ_max || θnx[2] > θη_max || θnx[2] < -15)

            #= display(θnx[2])
            display(θη_max)
            display(θnx[1])
            display(θϵ_max) =#

            success = false

            ρn = -2*Pn/(αn*μn)

            ∇Pnx = -∇Pn
            Pnx = 0.
            Λnx = Λ
            βnx = β

            θn = [log(2,ϵ); log(2,η)]
            θnx = θn

        else

            ϵnx = 2^θnx[1]
            ηnx = 2^θnx[2]

            ∇Pnx, Pnx, Λnx, βnx = Weights_and_Gradients(Y, D, ϵnx, ηnx)

            ρn = 2*(Pnx - Pn)/(αn*μn)

            success = (ρn >= 0)

        end

        # Step 6
        if ρn < 0.25

            λnx = minimum([λn + (δn*(1 - ρn)/κn); λmax])

        elseif ρn > 0.75

            λnx = maximum([λn/2; λmin])

        else
            λnx = λn
        end

        # Step 7: adjust the weights
        if success

            gnx = ∇Pnx

            S += 1

        else

            θnx = θn
            gnx = gn

        end

        # Step 8: choose new search direction
        if S == Smax
            # restart algoritm in direction of steepest descent

            snx = -gnx
            success = true
            S = 0

        else

            if success # create new conjugate direction

                Mn = (transpose(gn - gnx)*gnx)/μn #momentum term: Hestenes-Stiefel
                snx = -gnx + Mn*sn

                #= println("\nchange direction")
                println("norm(gnx) = ", norm(gnx))
                println("∇Pnx = ", ∇Pnx)
                println("Pnx = ", round(Pnx, digits = 3))
                println("ϵ = ", round((2^θnx[1])*1e6, digits = 3), "e-6")
                println("η = ", round((2^θnx[2]), digits = 3)) =#

            else # use current direction again

                snx = sn

            end

        end

        if norm(gnx) < τ

            P = Pnx
            ϵ = 2^θnx[1]
            η = 2^θnx[2]
            Λ = Λnx
            β = βnx

            converged = true

            #= println("\nout")
            println("norm(gnx) = ", norm(gnx))
            println("P = ", round(P, digits = 3))
            println("ϵ = ", round((2^θnx[1])*1e6, digits = 3), "e-6")
            println("η = ", round((2^θnx[2]), digits = 3)) =#

            break
        else

            sn = snx
            ∇Pn = ∇Pnx
            Pn = Pnx

            εn = εnx
            λn = λnx
            θn = θnx
            gn = gnx

        end

        if n == nmax
            P = Pnx
            ϵ = 2^θnx[1]
            η = 2^θnx[2]
            Λ = Λnx
            β = βnx
        end

    end

    # θnx[1] = log(2, ϵ)
    # θnx[2] = log(2, η)= [ϵ, η]

    return Λ, data, β, ϵ, η, P, converged
end

function Q_max(u, p)

    N = size(p[2], 2)

    Q = 0.

    for n in 1:N

        Q += ((p[2][:,n])*exp(-p[4]*sum(abs2, vcat(p[1], u[1]) - p[3][:,n])))[1]

    end

    return -Q
end

function Q_max2(u, p)

    #Q_params = (Λ_Q, C_Q, η_Q, β_Q)

    N = size(p[2], 2)

    Q = 0.

    for n in 1:N

        Q += ((p[1][:,n])*exp(-p[3]*sum(abs2, u[1] .- p[2][:,n])))[1]

    end

    return -Q - p[4]
end

function logistic_sigmoid(y)

    # logistic sigmoid link function

    return 1 ./ (1 .+ exp.(-y))
end

function RVM(data, Y, η; σ2 = nothing, noise_itr = 0, max_itr = 1000, σ2_min = 0.005, σ2_dif = 0.001, re_est = 10e-6, alg = :regress)

    function RVM_helper()

        next_idx = 0
        max_Δℒ = -Inf
        next_update = 0
        next_α = Inf

        #= update key
            0: do nothing
            1: add
            2: delete
            3: re-estimate
        =#

        for m in 1:M

            φm = G[:,m]

            if Mt == 1 || noise

                temp = transpose(φm)*B

                S[m] = temp*φm - temp*G[:, ids]*Σ*transpose(G[:, ids])*transpose(temp)
                Q[m] = temp*Ŷ - temp*G[:, ids]*Σ*transpose(G[:, ids])*B*Ŷ

            else

                if up == 1 # add

                    temp = β*transpose(φm)*ei

                    S[m] = S[m] - Σii*(temp)^2
                    Q[m] = Q[m] - μi*temp

                elseif up == 2 # delete

                    temp = β*transpose(Φ*Σⱼ)*φm

                    S[m] = S[m] + Σjj_inv*(temp)^2
                    Q[m] = Q[m] + Σjj_inv*μⱼ*temp

                elseif up == 3 # re-estimate

                    temp = β*transpose(Φ*Σⱼ)*φm

                    S[m] = S[m] + κⱼ*(temp)^2
                    Q[m] = Q[m] + κⱼ*μⱼ*temp
                end

                if S[m] < 0

                    temp = transpose(φm)*B

                    S[m] = temp*φm - temp*G[:, ids]*Σ*transpose(G[:, ids])*transpose(temp)
                end
            end

            Q2 = Q[m]^2

            if isinf(α[m]) # base is not included

                sm = S[m]
                qm = Q[m]

                θm = qm^2 - sm

                if θm <= 0 # do nothing

                    next_α_m = α[m]

                    Δℒ = -Inf

                    update_m = 0

                else # adding the basis function

                    next_α_m = sm^2/θm

                    Δℒ = (Q2 - S[m])/S[m] + log(S[m]/Q2)

                    update_m = 1
                end

            else # base is already included

                temp = α[m]/(α[m] - S[m])

                sm = temp*S[m]
                qm = temp*Q[m]

                θm = qm^2 - sm

                if θm <= 0 # deleting the basis function

                    next_α_m = Inf

                    Δℒ = Q2/(S[m] - α[m]) - log(1 - S[m]/α[m])

                    update_m = 2

                else # re-estimating the basis function

                    next_α_m = sm^2/θm

                    if log(abs.(next_α_m - α[m])) > re_est

                        temp = next_α_m*α[m]/(α[m] - next_α_m)

                        Δℒ = Q2/(S[m] + temp) - log(1 + S[m]/temp)

                        update_m = 3

                    else

                        next_α_m = α[m]

                        Δℒ = -Inf

                        update_m = 0
                    end
                end
            end

            if Δℒ > max_Δℒ

                next_idx = m
                max_Δℒ = Δℒ
                next_update = update_m
                next_α = next_α_m
            end
        end

        return next_idx, next_update, next_α
    end

    if isa(data, Vector)
        data = transpose(data)
    end

    if size(Y,1) == 1
        Y = transpose(Y)
    end

    if isnothing(σ2) && alg == :regress
        # Step 1: initialise σ² to some sensible value
        σ2 = 0.1*var(Y)
    elseif alg == :regress

        β = 1/σ2
    end

    if alg != :regress && noise_itr != 0
        noise_itr = 0
    end

    # calculate Gramian
    G = Gramian(data, η)

    N = size(G,1)

    G = hcat(ones(N), G)

    M = size(G,2)

    # define weights and hyperparameters

    α = Array{Float64, 1}(undef, M)
    α = fill!(α, Inf)
    μ = zeros(M)
    Σ = Array{Float64, 2}(undef, 1, 1)
    Φ = Array{Float64, 2}(undef, N, 1)

    ids = Array{Int64,1}(undef, 1)
    ids_vec = Array{Int64,1}(undef, M)
    ids_vec = fill!(ids_vec, 0)

    converged = false

    S = Array{Float64, 1}(undef, M) # ∝ sparsity factor
    Q = Array{Float64, 1}(undef, M) # ∝ quality factor

    Σii = 0.
    ei = Array{Float64, 1}(undef, 1)
    μi = 0.
    Σjj_inv = 1.
    μⱼ = 0.
    κⱼ = Array{Float64, 1}(undef, 1)
    Σⱼ = Array{Float64, 1}(undef, 1)

    up = 0
    noise = false
    iop = 1
    noise_cnt = 1
    Mt = 1

    # Step 2: initialise with a single basis vector

    norm2_φ = vec(sum(abs2, G, dims = 1))

    #proj = sum(abs2, (transpose(G)*Y), dims = 2)

    norm2_proj = sum(abs2, (transpose(G)*Y), dims = 2)./norm2_φ

    np_max, idx = findmax(norm2_proj)

    ids[1] = idx
    ids_vec[idx] = 1

    Φ[:,1] = G[:,idx]

    if alg == :regress

        α[idx] = norm2_φ[idx]/(np_max - σ2)

        I_NN = Matrix{Float64}(I, N, N)
        B = β*I_NN

        Ŷ = Y

        # Step 3: calculate variance and mean (weight)

        Σ[1,1] = 1/(α[idx] + β*transpose(vec(Φ))*vec(Φ))
        μ = β*Σ*transpose(Φ)*Y

    else

        # Step 3: calculate variance and mean (weight)

        α[idx] = norm2_φ[idx]/(np_max - σ2)

        Yapprox = logistic_sigmoid(G[:, ids]*μ)

        β_vec = logistic_sigmoid(Yapprox)
        β_vec = β_vec.*(1 .- β_vec)
        B = Diagonal(β_vec)
        Σ[1,1] = 1/(α[idx] + transpose(vec(Φ))*B*vec(Φ))

        Ŷ = vec(Φ)*μ + Diagonal(1 ./ β_vec)*(Y .- Yapprox)

        μ = Σ*transpose(vec(Φ))*B*Ŷ

    end

    next_idx, next_update, next_α = RVM_helper()

    if next_idx != 0

        while !converged

            # Select a candidate basis vector from the set of all M

            up = next_update

            idx = next_idx # indicates the single basis function for which α[i] is to be updated

            φi = G[:,idx]

            if noise_cnt == noise_itr

                γ = 1 .- α[ids].*diag(Σ)

                σ2_new = sum(abs2, Y - G[:, ids]*μ)/(N - sum(γ))

                σ2_new = maximum([σ2_new; σ2_min])

                noise_cnt = 1

                if abs(σ2_new - σ2) > σ2_dif
                    # only update σ² if the error is greater than this value, faster algorithm
                    # at the cost of accuracy

                    σ2 = σ2_new

                    β = 1/σ2

                    B = β*I_NN

                    noise = true

                else

                    noise = false
                end

            elseif noise_itr > 0

                noise_cnt += 1
                noise = false
            end

            if up == 1 # add

                α[idx] = next_α

                ids = vcat(ids, idx)
                Mt += 1
                ids_vec[idx] = Mt

                if !noise

                    temp = β*Σ*transpose(Φ)*φi

                    ei = φi - Φ*temp
                    Σii = 1/(α[idx] + S[idx])
                    μi = Σii*Q[idx]

                    μ = vcat(μ - μi*β*Σ*transpose(Φ)*φi, μi)

                    Σ_newA = Σ + Σii*temp*transpose(temp)
                    Σ_newB = -Σii*temp
                    Σ = vcat(hcat(Σ_newA, Σ_newB), hcat(transpose(Σ_newB), Σii))
                end

                Φ = hcat(Φ, φi)

            elseif up == 2 # delete

                j = ids_vec[idx]

                if j == 1
                    ids_vec[ids] = ids_vec[ids] .- 1
                else
                    ids_vec[ids] = [ids_vec[ids][1:j-1]; ids_vec[ids][j:end] .- 1]
                    ids_vec[idx] = 0
                end

                α[idx] = Inf
                Mt -= 1

                deleteat!(ids, j)

                if !noise

                    Σⱼ = copy(Σ[:, j])
                    Σjj_inv = 1/Σ[j, j]
                    μⱼ = copy(μ[j])

                    temp = Σjj_inv*Σⱼ

                    Σ = Σ - temp*transpose(Σⱼ)
                    μ = μ - μⱼ*temp

                    deleteat!(μ, j)

                    Σ = Σ[1:end .!= j, 1:end .!= j]
                end

            elseif up == 3 # re-estimate

                if !noise

                    j = ids_vec[idx]
                    Σⱼ = copy(Σ[:, j])

                    κⱼ = 1/(Σ[j, j] + 1/(next_α - α[idx]))
                    μⱼ = copy(μ[j])

                    temp = κⱼ*Σⱼ

                    Σ = Σ - temp*transpose(Σⱼ)
                    μ = μ - μⱼ*temp
                end

                α[idx] = next_α
            end

            Σ = 0.5*(Σ + transpose(Σ))

            if noise || !isposdef(Σ)

                temp = β*transpose(G[:, ids])

                # ---
                # option 1

                #Σ = inv(diagm(α[ids]) + β*transpose(G[:, ids])*G[:, ids])

                # ---
                # option 2 - ensures positive definite, results in increased sparsity

                Σ = Matrix{Float64}(I, Mt, Mt)

                Γ = diagm(α[ids]) + temp*G[:, ids]

                Γ = 0.5*(Γ + transpose(Γ))

                ldiv!(cholesky(Γ), Σ)

                # ---

                μ = Σ*temp*Y

            end

            next_idx, next_update, next_α = RVM_helper()

            if up == 2

                Φ = G[:, ids]
            end

            iop += 1

            converged = (next_update == 0) || up == 0 || iop == max_itr
        end

        if noise_itr > 0

            if noise_cnt != 1

                γ = 1 .- α[ids].*diag(Σ)

                σ2_new = sum(abs2, Y - G[:, ids]*μ)/(N - sum(γ))

                σ2_new = maximum([σ2_new; σ2_min])
            end

            σ2 = σ2_new
        end
    end

    #y = G[:, ids]*μ

    if ids_vec[1] != 0

        j = ids_vec[1]

        μ0 = μ[j]

        ids = deleteat!(ids, j) .- 1

        return (transpose(deleteat!(μ, j))), data[:,ids], μ0, σ2

    else

        return (transpose(μ)), data[:,ids .- 1], 0, σ2
    end
end

function KMD(coords, Gs, index_map; num_basis = nothing, tr = 1-1e-8, residual = false, ε = 1e-8, alg = :KEDMD)

    A = transpose(Gs[1:end-1, 2:end]) # Gyx
    G = Gs[1:end-1, 1:end-1] # Gxx

    N = size(G,1)

    if !isnothing(num_basis)

        num_basis = clamp(num_basis, 1, size(G,1))

        if num_basis == size(G,1)
            num_basis = nothing
        end
    end

    if alg == :RKHS

        invG = Matrix{Float64}(I, N, N)
        ldiv!(cholesky(G + N*ε*I), invG)
        #invG = inv(G + N*ε*I)

        U = invG*A

        if !isnothing(num_basis)

            decomp, history  = partialschur(U, nev = num_basis,
            tol = 1e-6, restarts = 200, which = LM())
            eigval, V = partialeigen(decomp)

            num_basis = length(eigval)

            eigval = eigval[end:-1:1]
            V = V[:, end:-1:1]

        else

            tr = clamp(tr, 0.0, 1.0)

            eigval, V = eigen(U)

            eigval = eigval[end:-1:1]
            V = V[:, end:-1:1]

            # Find r to capture % of the energy
            cdλ = abs.(cumsum(eigval.^2)./sum(eigval.^2))

            num_basis = findfirst(cdλ .>= tr) # truncation rank

            eigval = eigval[1:num_basis]
            V = V[:,1:num_basis]

        end

        eigval = eigval./eigval[1]
        V = V * Diagonal(eigval)

        if maximum(abs.(eigval)) > 1

            # ... but there may be numerical innacuracies, or irrelevant
            # component in the eigenbasis decomposition (the smaller eigenvalues
            # there have no significance, according to the MMD test, and we use
            # implicitly an inverse here).
            # This should not happen, but if it does, clip eigvals

            for i in 1:num_basis

                if abs(eigval[i]) > 1

                    eigval[i] /= abs(eigval[i])
                end
            end

            V = V * Diagonal(eigval)
        end

        φ = G*V
        Ξ = φ\transpose(coords[:, index_map[1:end-1]])
        #Ξ = (transpose(φ)*φ + 1e-9*I)\(transpose(φ)*transpose(coords[:, index_map[1:end-1]]))

        W = transpose(V)
        Phi = transpose(Ξ)

        DMD = (Phi, eigval, W)

        if residual

            Ξ = transpose(Phi)

            residual = norm(transpose(coords[:, index_map[2:end]]) - real.(φ*Diagonal(eigval)*Ξ))

            return DMD, residual, φ

        else

            return DMD
        end

    else alg == :KEDMD

        if !isnothing(num_basis)

            decomp, history  = partialschur(G, nev = num_basis, tol = 1e-6, restarts = 200, which = LM())
            Σ², Q = partialeigen(decomp)

            num_basis = length(Σ²)

            #Σ² = abs.(Σ²[end:-1:1])
            Σ² = Σ²[end:-1:1]
            Q = Q[:, end:-1:1]

            Σ = sqrt.(Σ²)

        else

            tr = clamp(tr, 0.0, 1.0)

            Σ², Q = eigen(G) # G = Q*diagm(Σ²)*transpose(Q)

            #Σ² = abs.(Σ²[end:-1:1])
            Σ² = Σ²[end:-1:1]
            Q = Q[:, end:-1:1]

            # Find r to capture % of the energy
            cdΣ = cumsum(Σ².^2)./sum(Σ².^2)
            num_basis = findfirst(cdΣ .>= tr) # truncation rank

            Σ = sqrt.(Σ²[1:num_basis])
            Q = Q[:,1:num_basis]

        end

        Σ = Diagonal(Σ)
        Σ_inv = inv(Σ)
        U = Σ_inv*transpose(Q)*A*Q*Σ_inv

        eigval, V = eigen(U) # U is not the Koopman matrix, it's a projection

        eigval = eigval[end:-1:1]
        V = V[:, end:-1:1]

        eigval = eigval./eigval[1]
        V = V * Diagonal(eigval)

        if maximum(abs.(eigval)) > 1

            # ... but there may be numerical innacuracies, or irrelevant
            # component in the eigenbasis decomposition (the smaller eigenvalues
            # there have no significance, according to the MMD test, and we use
            # implicitly an inverse here).
            # This should not happen, but if it does, clip eigvals

            for i in 1:num_basis

                if abs(eigval[i]) > 1

                    eigval[i] /= abs(eigval[i])
                end
            end

            V = V * Diagonal(eigval)
        end

        Vinv = V \ Matrix{Float64}(I, num_basis, num_basis)
        Phi = coords[:, index_map[1:end-1]]*Q*Σ_inv*transpose(Vinv) # koopman modes

        W = transpose(V)*Σ_inv*transpose(Q)

        DMD = (Phi, eigval, W)

        if residual

            φ = Q*Σ*V
            Ξ = Vinv*Σ_inv*transpose(coords[:, index_map[1:end-1]]*Q)

            residual = norm(transpose(coords[:, index_map[2:end]]) - real.(φ*Diagonal(eigval)*Ξ))

            return DMD, residual, φ

        else

            return DMD
        end
    end
end

function GP(data, Y, η; ϵ = 1e-6, alg = :classify, max_itr = 10, tol = 1e-9)

    #https://nbviewer.org/github/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes_classification.ipynb

    if size(Y,2) != 1
        Y = transpose(Y)
    end

    G = Gramian(data, η)

    G = G + ϵ*I

    N = size(G, 1)

    a = zeros(N)

    σ = Array{Float64, 1}(undef, N)
    W = Array{Float64, 2}(undef, N, N)

    converged = false

    for i in 1:max_itr

        σ = logistic_sigmoid(a)
        W = Diagonal(σ .* (1 .- σ))

        Qinv = (W*G + I) \ I

        a_new = G*Qinv*(Y .- σ .+ W*a)

        if norm(a_new .- a) < tol

            a = a_new
            converged = true
            break
        end

        a = a_new

    end

    μ_weights = Y .- σ

    σ²_weights = Matrix{Float64}(I, N, N)
    ldiv!(cholesky(inv(W) + G), σ²_weights)

    C = copy(data)

    return μ_weights, σ²_weights, C, converged
end

function GP_predict(x, μ_weights, σ²_weights, C, η, kernel = Gaussian_kernel)

    N = size(C, 2)
    ϑ = Array{Float64,1}(undef, N) # feature vector

    for n in 1:N

        ϑ[n] = kernel(C[:,n], x, η)
    end

    μ_a = transpose(ϑ)*μ_weights;
    σ²_a = 1 - transpose(ϑ)*σ²_weights*ϑ
    p = logistic_sigmoid(μ_a*1/(sqrt(1 + π*σ²_a/8)))

    return p
end

function Mean_Distance(X; KNN = false, tree = nothing, K = nothing)

    # if KNN is false then it is the average between all points - very slow

    Mx = size(X, 2) # sample size

    average_distance = 0.0

    if KNN == true

        if isnothing(tree)
            tree = BallTree(X)
        end
        if isnothing(K)
            K = convert(Int, round(N/4)) # K is 1/4 of the sample size
        end

        for i in 1:Mx

            _, dists = knn(tree, X[:,i], K)

            average_distance += sum(dists)/K
        end

        return average_distance/Mx, tree

    else

        for i in 1:Mx

            distance = 0

            for j in 1:Mx

                if i != j
                    distance += norm(X[:, i] .- X[:, j])
                end
            end

            average_distance += distance/Mx
        end

        return average_distance/Mx
    end
end

function ichol(G, γ; alg = :pgso, Mmax = size(G, 1))

    N = size(G, 1)

    Mmax = clamp(Mmax, 1, N)

    nG = norm(G)
    η = γ*nG # precision parameter

    if alg == :pgso

        #=
        hardoon
        R is an unsorted upper triangular matrix
        more precisely this is the partial Gram-Schmidt orthogonalisation
        faster algo
        =#

        # initialisation

        ids = Array{Int64,1}(undef, N)
        scale = Array{Float64,1}(undef, N)

        j = 1
        R = zeros(N, N) # feat
        norm2 = diag(G) #  norm2

        norm2_nxt, ids_nxt = findmax(norm2)

        sum_norm2 = sum(norm2)

        while sum_norm2 > (η) && j != Mmax + 1

            ids[j] = ids_nxt
            scale[j] = sqrt(norm2_nxt)

            norm2_nxt = 0.
            ids_nxt = 0

            for i in 1:N

                R[i, j] = (G[i, ids[j]] - transpose(R[i,1:j-1])*R[ids[j],1:j-1])/scale[j]

                R2 = R[i, j]^2
                norm2[i] = norm2[i] - R2

                sum_norm2 -= R2

                if norm2[i] > norm2_nxt
                    norm2_nxt = norm2[i]
                    ids_nxt = i
                end
            end

            j += 1
        end

        M = j - 1

        R = R[:, 1:M]
        ids = ids[1:M]
        scale = scale[1:M]
        error = norm(G - R*transpose(R))/nG

        # norm(G - R*transpose(R))/nG <= γ; error <= γ

        PGSO = (R, ids, scale, error)

        return PGSO

    elseif alg == :ichol

        # includes symmetric permutations of rows and columns to ensure rank is minimal.

        # Hardoon - CCA
        # if !isposdef(G) then error may occur

        # initialisation
        i = 1

        N = size(G, 1)
        P = Matrix{Float64}(I, N, N)
        R = diagm(diag(G)) # sorted lower triangular matrix
        K = copy(G)

        PerI = Matrix{Float64}(I, N, N)

        retcode = true

        while η < sum(diag(R)) && i != Mmax + 1 && retcode

            # find best new element's index
            j = argmax(diag(R)[i:N])

            # recover index relative to R
            j = (j + i) - 1

            # update permutation matrix P
            Pnext = copy(PerI)
            Pnext[i,i] = 0
            Pnext[j,j] = 0
            Pnext[i,j] = 1
            Pnext[j,i] = 1
            P = P*Pnext

            # permute elements in i and j in k
            K = Pnext*K*Pnext

            # update (due to new permutation) the already calculated elements
            Ri = R[i,1:i-1]

            R[i,1:i-1] = R[j,1:i-1]
            R[j,1:i-1] = Ri

            # Permute elements j,j and i,i of r
            Rjj = R[j,j]

            R[j,j] = R[i,i]
            R[i,i] = Rjj

            # Set
            R[i,i] = sqrt(R[i,i])

            # Calculate i-th column of r
            Rsum = K[i+1:N,i]
            for k in 1:i-1
                Rsum .-= (R[i+1:N,k]*R[i,k])
            end

            R[i+1:N, i] = (1/R[i,i])*Rsum

            # update only diagonal elements
            for k in i+1:N

                R[k,k] = K[k,k] - sum(R[k,1:i].^2)

                if R[k,k] < 0
                    # matrix is not positive definite.
                    retcode = false
                end
            end

            i = i + 1
        end

        M = i-1
        R = R[:,1:M]

        error = norm(transpose(P)*G*P - R*transpose(R))/nG # reconstruction error
        # error <= γ

        Ids = [findfirst(col .== 1) for col in eachcol(P)]

        #G[Ids, Ids] ≈ transpose(P)*G*P ≈ R*transpose(R)

        ichol = (R, Ids, P, error, retcode)

        return ichol
    end
end
