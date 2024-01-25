"""
The module `RmaMod` provides `structs` and `functions` to simulate and plot the dynamics
and bifurcation diagram of the Rosenzweig MacArthur model.

Exported structs are:
- `Iniv` initial condition objects
- `ParRma` model parameters objects
- `ParTime` time parameters objects

Exported functions are:
- `plot_rma()` to simulate and plot the RMA dynamics against time
- `plot_bif_rma()` to simulate and plot the RMA bifurcation diagram against K
"""
module RmaMod

### imported packages
using StaticArrays
using DifferentialEquations
using CairoMakie

# export structs and functions
export ParRma, IniV, ParTime, plot_rma, plot_bif_rma

### structs

# model parameters
"""
`ParRma` objects define model parameters, with default values:
- r = 1.0
- K = 10.0
- c = 1.0
- h = 2.0
- b = 2.0
- m = 1.0
"""
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

- `x0` and `y0` (default values `x0 = 1, y0 = 1.95`) used to contruct `u0`.
- `u0=[x0, y0]`, of type `SVector`, is the used field for ODEproblem definition
"""
@kwdef struct IniV
    x0::Float64 = 1.0
    y0::Float64 = 1.95
    u0::SVector{2, Float64} = SVector(x0, y0)
end

# time parameters
"""
`ParTime` objects define `tspan` and `tstep` for ODEproblem definition

default values:
tspan = (0.0, 60.0)
tstep = 0.1
"""
@kwdef struct ParTime
    tspan::Tuple{Float64, Float64} = (0.0, 60.0)
    tstep::Float64 = 0.1
end


### functions

# additional constructor for IniV objects
"""
    Iniv(u0::SVector{2, Float64})

additional constructor for `IniV` objects from a length 2 SVector `u0`
"""
IniV(u0::SVector{2, Float64}) = IniV(x0 = u0[1], y0 = u0[2])


# logistic function
"""
    logistic(x, p)

returns the logistic growth of `x` with parameters `p`

arguments:
- `x` is prey density (Real expected)
- `p` is a `ParRma`` parameter object (defined in this module)
"""
function logistic(x::Real, p::ParRma)
    (; r, K) = p    # deconstruct/get r and K from p
    return r*x*(1-x/K)
end


# holling II function
"""
    holling2(x,p)

returns the holling 2 functional response (normalised to 1) on `x` with parameters `p`

arguments:
- `x` is prey density (Real expected)
- `p` is a `ParRma` parameter object (defined in this module)
"""
function holling2(x::Real, p::ParRma)
    (; h) = p   # deconstruct h from p
    return x/(x+h)
end


# model definition
"""
    mod_rma(u, p, t)

defines the model equations using the SVector method of `DifferentialEquations.jl`

arguments:
- `u` is a length 2 SVector
- `p` is a `ParRma`` parameter object (defined in this module)
- `t` is time, according to the classical `DifferentialEquations.jl` interface
"""
function mod_rma(u::SVector{2}, p::ParRma, t)
    (; c, b, m) = p     # get c, b, m from p
    x = u[1]            # use x, y notations
    y = u[2]

    dx = logistic(x, p) - c * holling2(x,p) * y
    dy = b * holling2(x, p) * y - m * y

    return SVector(dx, dy) # return derivatives as SVector
end


# model simulation
"""
    sim_rma(iniv, p, pt; final)

simulates the RMA model, given initial condition `iniv`, model parameters `p`
and time parameters `pt`. returns the whole solution every time step (if
`final = false`, default) or only the final state value (`final = true`).

arguments:
- `iniv` is an `IniV` initial value object (defined in this module)
- `p` is a `ParRma` parameter object (defined in this module)
- `pt` is a `ParTime` time parameter object (defined in this module)
- `final` (kwarg) is a Boolean, defaulted to `false`
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


# equilibria computation
"""
    eqy_rma(p; Kmax, Kstep)

computes the equilibria `y` of the RMA model for `K in 0:Kstep:Kmax` given
model parameters `p`. returns 3 named Tuples `(Krg = , y0 = , yco = )`
corresponding to the different branches of equilibria (`y0`` for null equilibria
and `yco` for co-existence equilibria) corresponding to `Krg``.

drops an error if `Kmax` is smaller than the Hopf bifurcation value.

arguments:
- `p` is a `ParRma` parameter object (defined in this module)
- `Kmax` (kwarg, defaulted to 8.0) is the maximum value of K
- `Kstep` (kwarg, defaulted to 0.1) is the step of K values
"""
function eqy_rma(p::ParRma; Kmax::Real = 8.0, Kstep::Real = 0.1)
    (; r, c, h, b, m) = p # deconstruct p (K is useless since it is varied)

    # define bifurcation K values
    Ktrans = m*h/(b-m)
    Khopf = h+2*m*h/(b-m)

    # drops an error if Kmax is too small
    if Kmax < Khopf
        error("For a full computation of equilibria types, Kmax must be greater than $Khopf")
    end

    # y equilibria
    # below transcritical : only y=0
    Krg1 = 0:Kstep:Ktrans
    y01 = ones(length(Krg1)).*0     # broadcasting
    eqs1 = (Krg = Krg1, y0 = y01, yco = nothing)

    # between transcritical and Hopf : y=0 and y>0
    Krg2 = Ktrans:Kstep:Khopf
    y02 = ones(length(Krg2)).*0
    yco2 = [r/c*(h+m*h/(b-m))*(1-m*h/(b-m)/K) for K in Krg2]
    eqs2 = (Krg = Krg2, y0 = y02, yco = yco2)

    # above Hopf : y=0 and y>0
    Krg3 = Khopf:Kstep:Kmax
    y03 = ones(length(Krg3)).*0
    yco3 = [r/c*(h+m*h/(b-m))*(1-m*h/(b-m)/K) for K in Krg3]
    eqs3 = (Krg = Krg3, y0 = y03, yco = yco3)

    return eqs1, eqs2, eqs3
end


# limit cycle extrema computation
"""
    cy_rma(p; Kmax, Kstep)

computes the extrema `y` of the limit cycle of the RMA model for
`K in Khopf:Kstep:Kmax`, given model parameters `p`. returns a named
Tuple `(Krg = , ycmin = , ycmax = )` with the corresponding min and
max of the limit cycle corresponding to K values.

drops an error if `Kmax` is smaller than the Hopf bifurcation value.

arguments:
- `p` is a `ParRma` parameter object (defined in this module)
- `Kmax` (kwarg, defaulted to 8.0) is the maximum value of K
- `Kstep` (kwarg, defaulted to 0.1) is the step of K values
"""
function cy_rma(p::ParRma; Kmax::Float64 = 8.0, Kstep::Float64 = 0.01)
    # parameters and K range
    (; r, c, h, b, m) = p # deconstruct p (K is useless since it is varied)
    Khopf = h+2*m*h/(b-m)
    Krgh = Khopf-Kstep:Kstep:Kmax

    # drops an error if Kmax is too small
    if Kmax < Khopf
        error("For a computation of the limit cycle, Kmax must be greater than $Khopf")
    end

    # for storage
    ycmin = zero(Krgh)
    ycmax = zero(Krgh)

    # initial value and time parameters
    iniv = IniV()
    ptime = ParTime()

    # transient integration time
    ptrans = ParTime(tspan = (0.0, 8000.0))

    for (i, Kh) in enumerate(Krgh)
        # construct parameter from p, with K = Kh of the loop
        prmabif = ParRma(r, Kh, c, h, b, m)

        # simulate transient, get final state
        utr = sim_rma(iniv, prmabif, ptrans; final = true)[:,1]
        inivtr = IniV(utr) # construct new init value

        # start from end of transient, simulate limit cycle
        sol_cyc = sim_rma(inivtr, prmabif, ptime)

        # get min and max y along the cycle
        ycmin[i] = minimum(sol_cyc[2,:])
        ycmax[i] = maximum(sol_cyc[2,:])
    end

    cycle = (Krg = Krgh, ycmin = ycmin, ycmax = ycmax)
    return cycle
end


# plotting simulation against time
"""
    plot_rma(iniv, p, pt)

simulates and plots predator `x` and prey `y` density dynamics against time, given
model parameters `p` and time parameters `pt`. returns a `CairoMakie` figure object.

arguments:
- `iniv` is an `IniV` initial value object (defined in this module)
- `p` is a `ParRma` parameter object (defined in this module)
- `pt` is a `ParTime` time parameter object (defined in this module)
"""
function plot_rma(iniv::IniV, p::ParRma, pt::ParTime)
    # compute the simulation
    sol_rma = sim_rma(iniv, p, pt)

    # initialize figure
    fig = Figure(; fontsize = 20)
    ax = Axis(fig[1,1];
        title = "Modèle de Rosenzweig MacArthur\n ",
        xlabel = "temps",
        ylabel = "densités",
    )

    # plot solution
    lines!(ax, sol_rma.t, sol_rma[1,:]; lw = 2, label = "proies")
    lines!(ax, sol_rma.t,  sol_rma[2,:]; lw = 2, label = "prédateurs")
    axislegend(; position = :lt)

    return fig
end


# plotting bifurcation diagram
"""
    plot_bif_rma(p; Kmax, Kstep)

simulates and plots the bifurcation diagram: predator asymptotics as a function of `K in 0:Kstep:Kmax`,
given model parameters `p`. returns a `CairoMakie` figure object.

arguments:
- `p` is a `ParRma` parameter object (defined in this module)
- `Kmax` (kwarg, defaulted to 8.0) is the maximum value of K
- `Kstep` (kwarg, defaulted to 0.1) is the step of K values
"""
function plot_bif_rma(p::ParRma; Kmax = 8.0, Kstep = 0.1)
    # initialize figure
    fig = Figure(; fontsize = 20)
    ax = Axis(fig[1,1];
        title = "Bifurcations du modèle de Rosenzweig MacArthur\n ",
        xlabel = "capacité de charge 𝐾",
        ylabel = "densités",
    )

    # plot equilibria
    eqs1, eqs2, eqs3 = eqy_rma(p; Kmax = Kmax, Kstep = Kstep)
    lines!(eqs1.Krg, eqs1.y0; color = Cycled(1), lw = 2, label = "branche stable")
    lines!(eqs2.Krg, eqs2.y0; color = Cycled(2), lw = 2, label = "branche instable")
    lines!(eqs2.Krg, eqs2.yco; color = Cycled(1), lw = 2)
    lines!(eqs3.Krg, eqs3.y0; color = Cycled(2), lw = 2)
    lines!(eqs3.Krg, eqs3.yco; color = Cycled(2), lw = 2)

    # plot limit Cycle
    cycle = cy_rma(p; Kmax = Kmax) # we keep the default Kstep = 0.01 for accuracy
    lines!(cycle.Krg, cycle.ycmin; color = Cycled(3), lw=2, label = "cycle limite")
    lines!(cycle.Krg, cycle.ycmax; color = Cycled(3), lw=2)

    axislegend(ax, position = :lt, labelsize = 14)

    return fig
end

end
