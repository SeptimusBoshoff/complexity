using Complexity
using LinearAlgebra
import HiGHS
using JuMP

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........")

V = DMD[1] #Koopman modes
L = Diagonal(DMD[2]) #Koopman eigenvalues
W = DMD[3] #Koopman eigenfunctions

println(round.(abs.(V*W),digits = 3) == I) # should be true

Mk  = real(V*L*W)
#Mk = shift_op

K = transpose(Mk)

Ψy = transpose(coords_train[1:end,2:end])
Ψx = transpose(coords_train[1:end,1:end-1])

Ψyt = transpose(Ψy)
Ψxt = transpose(Ψx)

d = size(coords_train[1:end,2:end], 1)
m = size(coords_train[:,2:end], 2)

A = (1/m)*Ψyt*transpose(Ψxt)
G = (1/m)*Ψxt*transpose(Ψxt)

#K1 = transpose(Ψyt*pinv(Ψxt))
K1 = (Ψx\(Ψy)) #pinv(Ψx)*Ψy

Mk2 = A/G
K2 = transpose(Mk2) #pinv(G)*A

Mp = round.(transpose(A)/G, digits = 3)

P1 = transpose(Mp)

P2 = round.(inv(Ψxt*Ψx)*(Ψyt*Ψx), digits = 3)

S = Ψxt*Ψx

P3 = round.(real(inv(S)*transpose(K)*(S)), digits = 3)

Mp2 = round.(real((S)*(K)*inv(S)), digits = 3)

#-------------------------------------------------------------------------------------------

lp, vrp = eigen(P1)
lk, vrk = eigen(K1)
wrp = inv(vrp)

#generalised eigenvalue problem

lp2, vrp2 = eigen((A), transpose(G))
wrp2 = inv(vrp2)

P4 = round.(real.(vrp2*Diagonal(lp2)*wrp2), digits = 3)

test = transpose(vrp2)*transpose(A) - Diagonal(lp2)*transpose(vrp2)*G

ρ = transpose(vrp[:, end])*coords_train[:,end]
ψ1 = transpose(vrk[:, end-1])*coords_train[:,:]
ψ2 = transpose(coords_train[:,:])*vrk[:, end-1]
ψ3 = transpose(W[end,:])*coords_train[:,:]

#-------------------------------------------------------------------------------------------

real_ψ = transpose((real.(transpose(vrp2[:,end-4:end])*coords_train[:,1:900])))
imag_ψ = transpose((imag.(transpose(vrp2[:,end-4:end])*coords_train[:,1:900])))

plot_real_P_func = plot(real_ψ,
                Layout(
                    title = attr(
                        text = "real",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "time",
                    ),
                    ),
                )

plot_imag_P_func = plot(imag_ψ,
                Layout(
                    title = attr(
                        text = "imaginary",
                    ),
                    title_x = 0.5,
                    scene = attr(
                        xaxis_title = "time",
                    ),
                    ),
                )

plot_P_func = [plot_real_P_func; plot_imag_P_func]
relayout!(plot_P_func,
title_text = "Perron-Frobenius eigenfunctions",
title_x = 0.5,)
#display(plot_P_func)

println("...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
