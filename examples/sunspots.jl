using Complexity
using CSV
using DataFrames
using DSP
using Peaks
using PlotlyJS

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

#-------------------------------------------------------------------------------
# Machine parameters

sampling = 1 # machine retains 1 in every so many samples

history = 22*12 # backward trajectory [months]
future = 22*12 # forward trajectory [months]

#-------------------------------------------------------------------------------
# Machine Perspective

npast = convert(Int, round(history/sampling)) #Past Series sample size
nfuture = convert(Int, round(future/sampling)) #Future series sample size

window_size = npast + nfuture

df = CSV.read(joinpath(pwd(), "data//SN_m_tot_V2.0.csv"), DataFrame, header = false)
year = df.Column1
month = df.Column2
data = df.Column4[1:sampling:end] # sunspots

N = length(data)# number of samples

#-------------------------------------------------------------------------------
# Filtering

# For the data scale: we are interested here in the solar minima and maxima.
# Traditionnal filter used to determine the cycle maxima

traditional_filter = ones(13)
traditional_filter[1] = 0.5
traditional_filter[end] = 0.5

smoothed_sunspots = conv(data, traditional_filter)[6:end-7]
# Note: the traditional filter is not normalized. Normalize now
smoothed_sunspots /= sum(traditional_filter)

pks, pk_vals = findmaxima(smoothed_sunspots)
_, proms = peakproms(pks, smoothed_sunspots)
truepeaks = pks[findall(x->x > 0.4*maximum(proms), proms)]
true_pk_vals = pk_vals[findall(x->x > 0.4*maximum(proms), proms)]

valy, valy_vals = findmaxima(-1*smoothed_sunspots)
_, proms = peakproms(valy, -1*smoothed_sunspots)
truevalleys = valy[findall(x->x > 0.4*maximum(proms), proms)]
true_valy_vals = -1*valy_vals[findall(x->x > 0.4*maximum(proms), proms)]

# and take the average maxima as the relevant scale.
# This means, we focus on variations of that order of magnitude over
# the given time scales, bandwidth, i.e. Kernel Scale
scale = sum(true_pk_vals)/length(true_pk_vals) - sum(true_valy_vals)/length(true_valy_vals)

#-------------------------------------------------------------------------------
# Emergence - Pattern Discovery

@time begin

    @time begin
        #= 1. Generating gram matrices =#
        println("\n1. Generating gram matrices")
        Gx, Gy, index_map = series_Gxy(data, scale, npast, nfuture)
    end

    @time begin
        #=2. Computing Gs
            Compute the state similarity matrix.
            Embedding to get the similarity matrix between conditional distributions
        =#
        println("\n2. Computing Gs")
        Gs = embed_states(Gx, Gy)
    end

    @time begin
        #= 3. Projection
            Compute a spectral basis for representing the causal states.
            Find a reduced dimension embedding and extract the significant coordinates
        =#
        println("\n3. Projection")
        eigenvalues, basis, coords = spectral_basis(Gs, num_basis = 30)
    end

    @time begin
        #= 4. Forward Shift Operator
            This is the forward operator in state space. It is built from consecutive
            indices in the index map. Data series formed by multiple contiguous time
            blocks are supported, as well as the handling of NaN values
        =#
        println("\n4. Forward Shift Operator")
        shift_op = shift_operator(coords, index_map = index_map, alg = :nnls)
    end

    @time begin
        #= 5. Expectation Operator
            This is the expectation operator, using its default function that predicts
            the first entry in the future sequence from the current state distribution.
            You can specify other functions, see the documentation
        =#
        println("\n5. Expectation Operator")
        expect_op = expectation_operator(coords, index_map, data)
    end

    @time begin
        #= 6. Prediction
            Start from the last known point (represented by its coordinates) and
            evolve the state for nfuture+1 points.
        =#
        println("\n6. Prediction")
        pred, dist = predict(2*N - window_size + nfuture, coords[1, :], shift_op, expect_op,
        return_dist = 2, knn_convexity = 5, knndim = 5, coords = coords, extent = 0.05)
    end
end

#-------------------------------------------------------------------------------
# Plots

# (x, y) positions from machine perspective
nans_past = Array{Float64, 1}(undef, npast)
nans_past = fill!(nans_past, NaN)

nans_pred = Array{Float64, 1}(undef, N)
nans_pred = fill!(nans_pred, NaN)

nans_fut = Array{Float64, 1}(undef, nfuture)
nans_fut = fill!(nans_fut, NaN)

nans_c = Array{Float64, 1}(undef, N + nfuture - 1)
nans_c = fill!(nans_c, NaN)

t = 1:1:2*N
predictions = vec([nans_past; pred[1]])
data_extend = vec([data; nans_pred])
smoothed_extend = vec([smoothed_sunspots; nans_pred])

df_sunspots = DataFrame(months = t, sunspots = data_extend,
                        predictions = predictions,
                        smoothed = smoothed_extend)

trace_sunspots = scatter(df_sunspots, x = :months, y = :sunspots, name = "sunspots")
trace_smoothed = scatter(df_sunspots, x = :months, y = :smoothed, name = "smoothed")
trace_predictions = scatter(df_sunspots, x = :months, y = :predictions, name = "predictions")

plot_sun_t = plot([trace_sunspots, trace_predictions],
                Layout(
                    title = attr(
                        text = "Sunspots monthly series from SILSO",
                    ),
                    title_x = 0.5,
                    xaxis_title = "Months",
                    yaxis_title = "Number of Sunspots",),
                    )

display(plot_sun_t)

Φ₁ = dist[:, 2]
Φ₂ = dist[:, 3]
Φ₃ = dist[:, 4]

df_Ψ_Φ = DataFrame(Ψ₁ = [coords[:,2]; nans_c], Ψ₂ = [coords[:,3]; nans_c], Ψ₃ = [coords[:,4]; nans_c], Φ₁ = Φ₁, Φ₂ = Φ₂, Φ₃ = Φ₃)

trace_Ψ = scatter3d(df_Ψ_Φ, x = :Ψ₁, y = :Ψ₂, z = :Ψ₃, name = "Ψ", mode = "lines")
trace_Φ = scatter3d(df_Ψ_Φ, x = :Φ₁, y = :Φ₂, z = :Φ₃, name = "Φ", mode = "lines")

plot_Ψ_Φ_3d = plot([trace_Ψ, trace_Φ],
                Layout(
                    title = attr(
                        text = "Sunspots: Reconstructed State Space",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "Ψ₁",
                        yaxis_title = "Ψ₂",
                        zaxis_title = "Ψ₃",
                    ),
                    scene_aspectratio = attr(x = 1, y = 1, z = 3),
                    scene_camera = attr(
                        up = attr(x = 1, y = 0, z = 0),
                        center = attr(x = 0, y = 0, z = 0),
                        eye = attr(x = 2, y = 2, z = 2)
                        ),
                    ),
                )

display(plot_Ψ_Φ_3d)

trace = surface(z = coords[:,2:5], showscale = false)
layout = Layout(title = "Eigenfunctions", autosize = true,
                scene_aspectratio = attr(x = 2, y = 2, z = 1),)

#display(plot(trace, layout))

println("\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
