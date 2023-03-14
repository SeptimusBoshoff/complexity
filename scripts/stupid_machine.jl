using Complexity
using LinearAlgebra

function Log_Gaussian_kernel(x₁, x₂, ζ)

    sum_past_factor = -1/(length(x₁)*(2*ζ^2))

    k_x₁_x₂ = (sum(abs2, x₁ - x₂)*sum_past_factor)

    return k_x₁_x₂
end

function Log_Laplacian_kernel(x₁, x₂, ζ)

    sum_past_factor = -1/(length(x₁)*ζ)

    k_x₁_x₂ = (norm(x₁ .- x₂, 1)*sum_past_factor)

    return k_x₁_x₂
end

# run pendulum first

sensors = size(data, 1)

N = length(data[1])

npast = 200
nfuture = 200

N_data = N - npast - nfuture + 1

index = 1:N_data

gx = Array{Float64,2}(undef, N_data, N_data)
gy = Array{Float64,2}(undef, N_data, N_data)

for i in index

    x₁_1 = data[1][i:npast+i-1]
    x₁_2 = data[2][i:npast+i-1]

    for j in index

        x₂_1 = data[1][j:npast+j-1]
        x₂_2 = data[2][j:npast+j-1]

        gx[i, j] = Gaussian_kernel(x₁_1, x₂_1, scale[1])
        gx[i, j] = gx[i, j] + Gaussian_kernel(x₁_2, x₂_2, scale[2])

        gx[i, j] = exp(gx[i, j]/length(data))

    end

end

for i in index

    y₁_1 = data[1][npast+i:npast+i+nfuture-1]
    y₁_2 = data[2][npast+i:npast+i+nfuture-1]

    for j in index

        y₂_1 = data[1][npast+j:npast+j+nfuture-1]
        y₂_2 = data[2][npast+j:npast+j+nfuture-1]

        gy[i, j] = Gaussian_kernel(y₁_1, y₂_1, scale[1])
        gy[i, j] = gy[i, j] + Gaussian_kernel(y₁_2, y₂_2, scale[2])

        gy[i, j] = exp(gy[i, j]/length(data))

    end

end
