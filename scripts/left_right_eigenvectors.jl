using Complexity
using LinearAlgebra

A = [1 2 3; 4 5 6; 7 8 9]

x0 = [1; 2; 3]

u, v = eigen(A)
u, wt = eigen(transpose(A))

w = transpose(wt)

x1a  = A*x0

x1b = v*Diagonal(u)*inv(v)*x0

x1c = inv(w)*Diagonal(u)*w*x0

w*A - Diagonal(u)*w
A*v - v*Diagonal(u)

#w*v - I

#normalising to bi-orthogonal basis, i.e w*v = I
w = w./diag(w*v)

x1d = v*Diagonal(u)*w*x0

x1e = v[:,1]*u[1]*transpose(w[1,:])*x0 + v[:,2]*u[2]*transpose(w[2,:])*x0 + v[:,3]*u[3]*transpose(w[3,:])*x0

r = [1 2 3;1 2 3]
y = [4; 4; 4]

W = [1 3;2 4]
D = [2 0; 0 3]

P = [0.3 0.7; 0.9 0.1]
P = [0.5 0.5 0; 0.25 0.5 0.25; 0 0.5 0.5]

l, b = eigen(P)
l, pst = eigen(transpose(P))

ps = transpose((pst[:,3])./sum(pst[:,3]))

ps = (ps)*P

g = [1 2; 2 3;3 4]
f = [2, 3, 4]

btest = 1*b[:,1] + 1*b[:,2] + b[:,3]
#btest = btest./sum(btest)

#btest = P*btest
#sum(btest)
