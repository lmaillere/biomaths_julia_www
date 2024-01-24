"""
This module provides `structs` and `functions` to simulate and plot the dynamics
and bifurcation diagram of the Rosenzweig MacArthur model.

Defined structs are :

Defined functions are :
"""

module Rma

### imported packages
using StaticArrays
using DifferentialEquations
using CairoMakie

### structs

# model parameters
"`ParRma` objects define model parameters, with default values"
@kwdef struct ParRma
    r::Float64 = 1.0
    K::Float64 = 10.0
    c::Float64 = 1.0
    h::Float64 = 2.0
    b::Float64 = 2.0
    m::Float64 = 1.0
end

# initial value
"""
`IniV` objects define initial values for simulation.

- `x0` and `y0` have default values used to contruct `u0`.
- `u0`, of type `SVector`, is the used field for ODEproblem definition
"""
@kwdef struct IniV
    x0::Float64 = 1.0
    y0::Float64 = 1.95
    u0::SVector{2, Float64} = SVector(x0, y0)
end

# time parameters
"`ParTime` objects define `tspan` and `tstep` for ODEproblem definition"
@kwdef struct ParTime
    tspan::Tuple{Float64, Float64} = (0.0, 60.0)
    tstep::Float64 = 0.1
end


### functions

# additional constructor for IniV objects
"""
    Iniv(u0::SVector{2, Float64})

additional constructor for IniV objects from a length 2 SVector `u0`
"""
IniV(u0::SVector{2, Float64}) = IniV(x0 = u0[1], y0 = u0[2])


# logistic function
"""
    logistic(x, p)

returns the logistic growth of `x` with parameters `p`
- `x` is prey density
- `p` is a ParRma parameter object
"""
function logistic(x, p::ParRma)
    (; r, K) = p    # deconstruct/get r and K from p
    return r*x*(1-x/K)
end


# holling II function
"""
    holling2(x,p)

returns the holling 2 functional response (normalised to 1) on `x` with parameters `p`
- `x` is prey density
- `p` is a ParRma parameter object
"""
function holling2(x, p::ParRma)
    (; h) = p   # deconstruct h from p
    return x/(x+h)
end


# model definition
"""
    mod_rma(u, p, t)

defines the model equations using the SVector method of `DifferentialEquations.jl`
- `u` is a length 2 SVector of floats
- `p` is a ParRma parameter object
- `t` is time, according to the classical `DifferentialEquations.jl` interface
"""
function mod_rma(u::SVector{2, Float64}, p::ParRma, t)
    (; c, b, m) = p     # get c, b, m from p
    x = u[1]            # use x, y notations
    y = u[2]

    dx = logistic(x, p) - c * holling2(x,p) * y
    dy = b * holling2(x, p) * y - m * y

    return SVector(dx, dy) # return derivatives as SVector
end


# model simulation
"""
    sim_rma(iniv, p, pt; final = false)
"""
function sim_rma(iniv::IniV, p::ParRma, pt::ParTime; final::Bool = false)
    # deconstruct time parameter
    (; tspan, tstep) = pt
    (; u0) = iniv

    # define and solve simulation problem
    prob_rma = ODEProblem(mod_rma, u0, tspan, p)
    if !final   # if final == false compute whole solution
        sol_rma = solve(prob_rma; reltol = 1e-6, saveat = tstep)
    else        # if final == true compute only final state
        sol_rma = solve(
            prob_rma;
            reltol = 1e-6,
            save_everystep = false,
            save_start = false,
        )
    end

    return sol_rma
end

### export structs and functions

end
