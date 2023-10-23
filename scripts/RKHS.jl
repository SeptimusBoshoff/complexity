using Complexity
using LinearAlgebra

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

# number of samples
m = 1000

# regularisation parameter
ε = 1e-8

# data
x = Array{Int64, 1}(undef, m)
y = Array{Int64, 1}(undef, m)

# feature functions
ϕ = [1 0; 0 1]

# feature matrices
Υ = Array{Float64, 2}(undef, 2, m)
Φ = Array{Float64, 2}(undef, 2, m)

# centring matrix
H = I - (1/m)*ones(m)*transpose(ones(m))

for i in 1:m

    state_x = rand()
    state_y = rand()
    state_y = state_x

    if state_x < 1/3
        x[i] = 1
    else
        x[i] = 2
    end

    if state_y < 1/2
        y[i] = 1
    else
        y[i] = 2
    end

    Υ[:,i] = ϕ[x[i],:]
    Φ[:,i] = ϕ[y[i],:]
end

# kernel mean
μx = (1/m)*sum(Υ, dims = 2)
μy = (1/m)*sum(Φ, dims = 2)

# kernel (uncentered) cross-covariance
Cxx = (1/m)*Υ*transpose(Υ)
Cyy = (1/m)*Φ*transpose(Φ)
Cyx = (1/m)*Φ*transpose(Υ)
Cxy = (1/m)*Υ*transpose(Φ)

# kernel (centered) cross-covariance
Cxxₕ = (1/m)*Υ*H*transpose(Υ) # (1/m)*(Υ - μx*transpose(ones(m)))*transpose((Υ - μx*transpose(ones(m))))
Cyyₕ = (1/m)*Φ*H*transpose(Φ) # (1/m)*(Φ - μy*transpose(ones(m)))*transpose((Φ - μy*transpose(ones(m))))
Cyxₕ = (1/m)*Φ*H*transpose(Υ) # (1/m)*(Φ - μy*transpose(ones(m)))*transpose((Υ - μx*transpose(ones(m))))
Cxyₕ = (1/m)*Υ*H*transpose(Φ) # (1/m)*(Υ - μx*transpose(ones(m)))*transpose((Φ - μy*transpose(ones(m))))

# Gram matrices
Gˣ = transpose(Υ)*Υ
Gʸ = transpose(Φ)*Φ

# kernel conditional embedding
Cy_x = Φ*((Gˣ + ε*I)\transpose(Υ)) # Cyx*inv(Cxx)
Cx_y = Υ*((Gʸ + ε*I)\transpose(Φ)) # Cxy*inv(Cyy)

# (centered)kernel conditional embedding
Cy_xₕ = Cyxₕ*inv(Cxxₕ + ε*I) # Φ*inv(H*Gˣ + m*ε*I)*H*transpose(Υ)
Cx_yₕ = Cxyₕ*inv(Cyyₕ + ε*I) # Υ*inv(H*Gʸ + m*ε*I)*H*transpose(Φ)

# kernel matrix
kx = transpose(Υ)*ϕ
ky = transpose(Φ)*ϕ

# weight vector
ωx = (Gˣ + ε*I)\kx
ωy = (Gʸ + ε*I)\ky

Ωx = (Gˣ + m*ε*I)\Gˣ
Ωy = (Gʸ + m*ε*I)\Gʸ

# centered weight vector
ωxₕ = (H*Gˣ + m*ε*I)\(H*kx)
ωyₕ = (H*Gʸ + m*ε*I)\(H*ky)

Ωxₕ = (H*Gˣ + m*ε*I)\(H*Gˣ)
Ωyₕ = (H*Gʸ + m*ε*I)\(H*Gʸ)

# kernel conditional mean
μy_x = Φ*ωx # Φ*inv(Gˣ + m*ε*I)*transpose(Υ) # Cyx*inv(Cxx + ε*I)
μx_y = Υ*ωy # Υ*inv(Gʸ + m*ε*I)*transpose(Φ) # Cxy*inv(Cyy + ε*I)

# state similarity matrix
Gsˣ = transpose(ωx)*Gʸ*ωx
Gsˣₕ = transpose(ωxₕ)*Gʸ*ωxₕ

# (centered) kernel conditional mean
μy_xₕ = Φ*ωxₕ # Cyxₕ*inv(Cxxₕ + ε*I)
μx_yₕ = Υ*ωyₕ

check = Array{Float64, 1}(undef, 6)
check[1] = maximum(round.(Cy_x - Cyx*inv(Cxx), digits = 5))
check[2] = maximum(round.(μy_x[:,1] - Cy_x*ϕ[1,:], digits = 5))
check[3] = maximum(round.(μy_x[:,2] - Cy_x*ϕ[2,:], digits = 5))
check[4] = maximum(round.(Cx_y - Cxy*inv(Cyy)))
check[5] = maximum(round.(μx_y[:,1] - Cxy*inv(Cyy)*ϕ[1,:], digits = 5))
check[6] = maximum(round.(μx_y[:,2] - Cxy*inv(Cyy)*ϕ[2,:], digits = 5))

println("\nCxx = ")
display(Cxx)
println("\nCxy = ")
display(Cxy)
println("\nCx_y = ")
display(Cx_y)
println("\nμx_y = ")
display(μx_y)
println("\nCyy = ")
display(Cyy)
println("\nCyx = ")
display(Cyx)
println("\nCy_x = ")
display(Cx_y)
println("\nμy_x = ")
display(μy_x)

#-------------------------------------------------------------------------------
# Hilbert-Schmidt Independence Criterion

hsic = norm(Cyx - μy*transpose(μx))
hsicₕ = norm(Cyxₕ)

println("\nhsic = ",hsic)
println("hsicₕ = ",hsicₕ)

#-------------------------------------------------------------------------------
# Kernel Sum Rule

μxᵖ  = Array{Float64, 1}(undef, 2)
μyᵖ = Array{Float64, 1}(undef, 2)

Cxxᵖ  = Array{Float64, 2}(undef, 2, 2)
Cyyᵖ = Array{Float64, 2}(undef, 2, 2)

αy = Φ\μy
αx = Υ\μx

μxᵖ = Υ*((Gʸ + ε*I)\(Gʸ*αy)) # Cx_y*μy
μyᵖ = Φ*((Gˣ + ε*I)\(Gˣ*αx)) # Cy_x*μx

Cxxᵖ  = Υ*diagm(vec((Gʸ + ε*I)\(Gʸ*αy)))*transpose(Υ)
Cyyᵖ = Φ*diagm(vec((Gˣ + ε*I)\(Gˣ*αx)))*transpose(Φ)

#-------------------------------------------------------------------------------
# Kernel Chain Rule

Λʸ = (Gʸ + ε*I)\(Gʸ*diagm(vec(αy)))
Λˣ = (Gˣ + ε*I)\(Gˣ*diagm(vec(αx)))

Dʸ = diagm(vec((Gʸ + ε*I)\(Gʸ*αy)))
Dˣ = diagm(vec((Gˣ + ε*I)\(Gˣ*αx)))

Cxyᵖ = Υ*Λʸ*transpose(Φ) # Cx_y*Cyy # Υ*Dʸ*transpose(Φ)
Cyxᵖ = Φ*Λˣ*transpose(Υ) # Cy_x*Cxx # Φ*Dˣ*transpose(Υ) # Φ*transpose(Λʸ)*transpose(Υ)

Cxxᵖ = Υ*Λˣ*transpose(Υ) # Υ*Dˣ*transpose(Υ)
Cyyᵖ = Φ*Λʸ*transpose(Φ)

#-------------------------------------------------------------------------------
# Kernel Bayes' Rule

# Cyy = Φ*(αy.*transpose(Φ)) # Φ*diagm(vec(αy))*transpose(Φ)

εᵖ = 10e-6

Cyxᵖ = Φ*transpose(Λʸ)*transpose(Υ) # Φ*Dʸ*transpose(Υ)
Cxxᵖ = Υ*Dʸ*transpose(Υ)

β_x = transpose(Λʸ)*(((Dʸ*Gˣ)^2 + εᵖ*I)\(Gˣ*Dʸ*kx))

μy_xᵖ = Φ*β_x # Φ*Dʸ*Gˣ*(((Dʸ*Gˣ)^2 + εᵖ*I)\(Dʸ*kx)) # Cyxᵖ*((Cxxᵖ + εᵖ*I)\(ϕ))

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
