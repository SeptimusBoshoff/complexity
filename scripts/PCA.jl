# Linear PCA
Sx = copy(data_train)
dimX = size(Sx, 1)
M = size(Sx, 2)
μs = (1/M)*(Sx)*ones(M)
Ms = μs*ones(1,M)
Sx_c = Sx .- Ms
Css = (1/(M-1))*Sx_c*transpose(Sx_c)

Λ, V = eigen(Css)
Λ = real.(Λ[end:-1:1]) # principal component variances, squares of singular values
V = real.(V[:, end:-1:1]) # right eigenvectors

norms1 = sqrt.(sum((V.^2), dims=1))
V = V ./ (norms1)

Ms, Σ, Vs = svd(transpose(Sx_c))

norms2 = sqrt.(sum((Vs.^2), dims=1))
Vs = Vs ./ (norms2)

PCx = transpose(V)*Sx_c # the principal components
PCxx = (1/(M-1))*PCx*transpose(PCx)

println(abs.(V) ≈ abs.(Vs))
println(Λ ≈ (Σ.^2)./(M-1))
println(abs.(PCx) ≈ abs.(transpose(Ms*Diagonal(Σ))))
println(PCxx ≈ diagm(Λ))
