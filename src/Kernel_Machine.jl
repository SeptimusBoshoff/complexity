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

    if size(series[1], 1) == 1
        series_list = [series]
    else
        series_list = series
    end

    nseries = size(series_list, 1) # the number of sources / sensors

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

    index_map = compute_index_map_multiple_sources(series_list, npasts_list, nfutures_list, qcflags_list, take2skipX = take2skipX)

    total_lx = nothing
    total_ly = nothing

    for (ser, sca, npa, nfu, dec, ldiff) in zip(series_list, scales_list, npasts_list, nfutures_list, decays_list, localdiff_list)

        lx, ly = series_xy_logk_indx(ser, sca, npa, nfu, dec, index_map, kernel_params_, ldiff)

        if total_lx === nothing

            total_lx, total_ly = lx, ly
        else

            parallel_add_lowtri!(total_lx, lx)
            parallel_add_lowtri!(total_ly, ly)
        end
    end

    parallel_exp_lowtri!(total_lx, nseries)
    parallel_exp_lowtri!(total_ly, nseries)

    return total_lx, total_ly, index_map
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
function embed_states(Gx, Gy; ϵ = 1e-8, normalize = true, return_embedder = false)

    #Ω = (Gx + ϵ*I) \ Gx
    #Ω = (H*Gˣ + N*ϵ*I) \ (H*Gx)

    centre_embedder = false # under development

    if !centre_embedder

        Ω = copy(Gx) # this is our weight matrix
        ldiv!(cholesky(Gx + ϵ*I), Ω)

    else

        N = size(Gx, 1)
        H = I - (1/N)*ones(N)*transpose(ones(N)) # centring matrix

        Ω = H*copy(Gx) # this is our centred weight matrix? normalised? breaks the symmetry?
        ldiv!(cholesky(H*Gx + N*ϵ*I), Ω)
    end

    Ω = Symmetric(Ω) # should not be needed

    embedder = transpose(Ω) * Gy

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

function embed_Kx(Kx, Gx, embedder, ϵ = 1e-8)
    """
    Construct a similarity vector in state space, from a similarity vector in X space and using the embbeder returned by embed_states

    Arguments:
        Kx: a similarity vector in X space, such as returned by series_newKx.
        Gx: a similarity matrix of pasts X.
        embedder: The embedder returned by embed_states
        eps: amount of regularization. See embed_states

    Returns:
        Ks: the similarity vector, in state space
    """

    omega = nonneg_lsq(Gx + ϵ*I, Kx, alg = :nnls)

    return (embedder * omega)
end
