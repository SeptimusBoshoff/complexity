using Complexity
using LinearAlgebra
using PlotlyJS

λ = 0:0.1:1000

l = length(λ)

#G = [3.0 2 3; 4 5 6; 7 8 9]

G = Gx[1:20,1:20]

d_eff = zeros(l)

for i in 1:l
    d_eff[i] = tr(inv(G + λ[i]*I)*G)
end

plot(λ,d_eff)
