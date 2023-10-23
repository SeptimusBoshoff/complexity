#_______________________________________________________________________________
# Dynamic systems

function Logistic_Map(x, λ; r = 3.9277370017867516)

    # Misiurewicz point:
    # r = 3.9277370017867516
    # Accumulation board - onset of chaos
    #r = 3.5699456718695445

    x = r*x*(1 - x)

    if λ == 0
        λ = log.(2, abs.(r .- 2*r*x[1]))
    else
        λ = λ .+ log.(2, abs.(r .- 2*r*x[1]))
    end

    return x, λ
end

function Lorenz!(du, u, p, t)

    #= parameters
        p[1] = σ - Prandtl
        p[2] = ρ - Raleigh
        p[3] = β - geometric aspect ratio

        u0 = [1.0, 0.0, 0.0]
        tspan = (0.0, 100.0)
        p = [10.0, 28.0, 8/3]
    =#

    x, y, z = u
    σ, ρ, β = p

    du[1] = σ*(y - x)
    du[2] = ρ*x- y - x*z
    du[3] = x*y - β*z

    return du
end

function Driven_Duffing!(du, u, p, t)

    #= Theory
        The driven duffing system, a forced harmonic ocillator, is an example
        of a nonautomonous system. That is, it has a time dependence. It is also
        an example of a three-dimensional system. Similarly, an nth-order time-
        dependent equation is a special case of an (n+1)-dimensional system.
        By this trick, we may remove any time dependence by adding an extra
        dimension to the system. A more physical motivation is that we need
        to know three numbers u[1], u[2], and u[3], to predict that future,
        given the present. It is a 2nd order differential equaion

        m = mass
        δ = controls the amount of damping
        α = controls the linear stiffness
        β = controls the amount of non-linearity in the restoring force;
            if zero the Duffing equation is a damped and driven simple harmonic oscillator
        γ = amplitude of the periodic driving force
        ω = angular frequency

        m*ddx + δ*dx + α*x + β*x^3 = γ*cos(ω*t)

        p = [m, δ, α, β, γ, ω]

        p = [1, 0.25, -1, 1, 0.4, 1]
        p = [1, 0.3, -1, 1, 0.5, 1.2] # Chaotic
    =#

    x, y, z = u
    m, δ, α, β, γ, ω = p

    du[1] = y
    du[2] = (1/m)*(-δ*y - α*x - β*x^3 + γ*cos(z))
    du[3] = ω

    return du
end

function Pendulum!(du, u, p, t)

    θ, ω = u
    m, g, L, b = p

    du[1] = ω
    du[2] = -b/m*ω - (g/L)*sin(θ)

    return du
end

function Van_der_Pol!(du, u, p, t)

    x, y = u
    μ = p[1]

    du[1] = y
    du[2] = μ*(1 - x^2)*y - x

    return du
end

function σ_Van_der_Pol!(du, u, p, t)

    du[1] = 0.001
    du[2] = 0.001

    return du
end

function Delayed_Van_der_Pol!(du, u, h, p, t)

    x, y = u
    μ, v0, v1, beta0, beta1, tau = p

    hist = h(p, t - tau)[2]

    du[1] = (v0 / (1 + beta0 * (hist^2))) * y
    du[2] = (v1 / (1 + beta1 * (hist^2))) * μ*(1 - x^2)*y - x
end

function Ornstein_Uhlenbeck!(du, u, p, t)

    α = p[1]
    D = p[2]

    du[1] = -α*D*u[1]

    return du
end

function σ_Ornstein_Uhlenbeck!(du, u, p, t)

    D = p[2]

    du[1] = sqrt(2*D)

    return du
end

function double_gyre(du, u, p, t)

    x, y = u
    A = p[1]

    du[1] = -π*A*sin(π*x)*cos(π*y)
    du[2] = π*A*cos(π*x)*sin(π*y)

    return du

end

function σ_double_gyre(du, u, p, t)

    x, y = u
    ε = p[2]

    du[1] = ε*sqrt(x/4 + 1)
    du[2] = ε

    return du

end

#_______________________________________________________________________________
# Reinforcement Learning

function Mountain_Car(x, a)

    v_next = clamp(x[2] + 0.001*a[1] - 0.0025*cos(3*x[1]), -0.07, 0.07)
    x_next = clamp(x[1] + v_next, -1.2, 0.5)

    if ((x[1] + v_next) < -1.2) || ((x[1] + v_next) > 0.5)

        v_next = 0

    end

    return [x_next, v_next]
end

function pendulum(x, a; max_speed = 8, max_torque = 2.0, g = 9.81, m = 1.0, l = 1.0, dt = 0.05)

    θ = x[1]
    ω = x[2]

    a = clamp(a[1], -max_torque, max_torque)

    ω_next = ω + ((3 * g / (2 * l)) * sin(θ) + (3.0 / (m * l^2) * a[1])) * dt
    ω_next = clamp(ω_next, -max_speed, max_speed)
    θ_next = θ + ω_next * dt

    if θ_next > π
        θ_next = θ_next - 2π
    elseif θ_next < -π
        θ_next = θ_next + 2π
    end

    return [θ_next, ω_next]
end
