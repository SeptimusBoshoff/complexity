using Complexity
using LinearAlgebra

K = Array{Float64, 3}(undef, 2, 2, 3)

A = Array{Float64, 2}(undef, 2, 3)
A = fill!(A, randn())

f_xh = zeros(2)

for i in 1:3

    global f_xh, K, A

    K[:,:, i] = randn(2,2)
    A[:,i] = rand(2,1)

    f_xh = f_xh + K[:,:, i]*A[:,i]

end

m = 2
l = 3

a = Array{Float64, 2}(undef, m, l)
a = fill!(a, randn())

f = Array{Float64, 2}(undef, m, l)
f = fill!(f, randn())

r1 = norm(a - f)

r2 = 0
r4 = 0

for i in 1:l

    global a,f, r2

    r2 = r2 + norm(a[:,i] - f[:,i])^2

    for j in 1:m

        global a,f, r4

        r4 = r4 + (a[j,i] - f[j,i])^2 #HS norm

    end

end
r2 = sqrt(r2)
r4 = sqrt(r4)
r3 = sqrt(tr((a-f)*transpose(a-f))) #frobenius
r5 = sqrt(tr(transpose(a - f)*(a - f)))
