# Import the package and define the problem to optimize
using Complexity
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Zygote

println("...........o0o----ooo0§0ooo~~~  START  ~~~ooo0§0ooo----o0o...........\n")

function rosenbrock(u, p)

    return (p[1][1] - u[1]*p[2])^2 + p[1][2] * (u[2]*p[2] - (u[1]*p[2])^2)^2
end
u0 = zeros(2)
p = ([1.0, 100.0], 2)

a = Gaussian_kernel([1;2;3], [4;4;4], 0.1)
G = Gramian(data1, 0.1)

optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(optf, u0, p)
sol = solve(prob, BFGS())

println("gradient sol = ", sol)

prob = OptimizationProblem(rosenbrock, u0, p)
sol = solve(prob, NelderMead())

println("NelderMead sol = ", sol)

a0 = [1.9/max_torque]
state = [cos(-π/2); sin(-π/2); ω_scale*0.0]
η = 1.0
N = size(critic_Λ, 2)

critic_params = (state, critic_Λ, critic_C, η)

#Q = function_approximation(vcat(p[1], u0[1]), p[2], p[3], p[4])

#= prob = OptimizationProblem(kernel_optim, u0, p, lb = [20.], ub = [100.])
sol = solve(prob, NelderMead()) =#

optf = OptimizationFunction(Q_max, Optimization.AutoZygote())
prob = OptimizationProblem(optf, a0, critic_params, lb = [-1.0], ub = [1.0])
sol = solve(prob, BFGS())

Q = function_approximation(vcat(critic_params[1], sol.u[1]), critic_params[2], critic_params[3], critic_params[4])
Q2 = Q_max(sol.u, critic_params)

println("gradient sol = ", sol)
println("action = ", max_torque*sol.u[1])
println("Q = ", Q)
println("Q2 = ", Q2)


A = collect(-max_torque:0.01:max_torque)
Q_test = Array{Float64, 1}(undef, length(A))

for a_idx in eachindex(A)

    action = A[a_idx]/max_torque

    Q_test[a_idx] = -1*function_approximation(vcat(critic_params[1], action), critic_params[2], critic_params[3], critic_params[4])

end


#-------------------------------------------------------------------------------------------
# Plots

traces = [scatter(x = A, y = Q_test, mode="lines", name = "Q")]

QA_plot = plot(
                traces,
                Layout(
                    title = attr(
                        text = "Actions",
                        ),
                    title_x = 0.5,
                    xaxis_title = "actions",
                    yaxis_title = "Q",
                    ),
                )

display(QA_plot)

println("\n\n...........o0o----ooo0§0ooo~~~   END   ~~~ooo0§0ooo----o0o...........\n")
