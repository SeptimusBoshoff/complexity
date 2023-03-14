using Complexity
using LinearAlgebra
using ArnoldiMethod

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

num_basis = 20
N = size(Gs, 1)
alpha = 0.1

mat = copy(Gs)

q = vec(sum(mat, dims = 1))

q = q.^alpha

q = (1) ./ q

mat = mat .* q
mat = mat .* transpose(q) # not sure why this is necessary

q = vec(sum(mat, dims = 1))

decomp, history  = partialschur(Diagonal((1) ./ q)*mat, nev = num_basis,
tol = 1e-6, restarts = 200, which = LM())
eigval, eigvec = partialeigen(decomp)

order = sortperm(eigval, rev = true)
eigval = eigval[order]
eigvec = eigvec[:, order]

eigvec_r = eigvec ./ eigvec[:, 1]
eigvec_l = eigvec .* (eigvec[:,1])

eigvec_l = transpose(eigvec_l)

Diagonal((1) ./ q)*mat * eigvec
eigval[2]*eigvec[:,2]

@show norm(mat * eigvec_r - Diagonal(q)*eigvec_r * Diagonal(eigval))
@show norm(eigvec_l*mat - Diagonal(eigval)*eigvec_l*Diagonal(q))
@show sum(eigvec_l[1,:])

P = Diagonal((1) ./ q)*mat

πy1 = q[1]/sum(q)
πy2 = q[2]/sum(q)

πy1*P[1,2] - πy2*P[2,1]

#= Dt = Array{Float64,2}(undef, N, N)

for i in 1:N
    for j in 1:N

        Dt[i, j] = sum(abs2, (P[i,:] .- P[j,:]))

    end
end

Dt2 = Array{Float64,2}(undef, N, N)

for i in 1:N
    for j in 1:N

        Dt2[i, j] = sum(abs2, eigval.*(eigvec[i,:] .- eigvec[j,:]))

    end
end =#

df_X = DataFrame(X_2 = eigvec_r[:,2], X_3 = eigvec_r[:,10], X_4 = eigvec_r[:,20], N = 1:N)

trace_X_2 = scatter(df_X, x = :N, y = :X_2, name = "X_2")
trace_X_3 = scatter(df_X, x = :N, y = :X_3, name = "X_3")
trace_X_4 = scatter(df_X, x = :N, y = :X_4, name = "X_4")

plot_X = plot([trace_X_2, trace_X_3, trace_X_4],
                Layout(
                    title = attr(
                        text = "Damped Pendulum: Evolution in Time",
                        ),
                    title_x = 0.5,
                    xaxis_title = "t [s]",
                    yaxis_title = "y,x [m]",
                    ),
                )
display(plot_X)

trace = surface(z = eigvec_r[:,1:end], showscale = false)
layout = Layout(title = "Eigenfunctions", autosize = true,
                scene_aspectratio = attr(x = 2, y = 2, z = 1),)

#display(plot(trace, layout))

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")


transpose(eigvec_l[2,:])*(eigvec[:,2])
