# yada
using Distributions
using DifferentialEquations
using FoodWebs
using Colors
using PyPlot
import PyCall: PyObject

# I need a way to convert the types from Colors.jl (RGB{8}(r, g, b) -> python tuple)
# this code is directly from PyCall.jl -> conversions.jl for tuple conversion
# Ultimately this should get into an upstream package.
function PyObject(t::Color)
    trgb = convert(RGB, t)
    ctup = map(float, (red(trgb), green(trgb), blue(trgb)))
    o = PyObject(ctup)
    return o
end

"""
Let's look at a low diversity example first
"""

function rand_comp(S::Int, C::Float64; trials = 10000)
    adj = may_network(S, C)
    amat = diagm(-rand(Uniform(1, 5), S))
    for i = 1:S, j = i:S
        if adj[i, j] == 1
            amat[i, j] = -rand(Uniform(0, 1))
            amat[j, i] = -rand(Uniform(0, 1))
        end
    end

    bs = []
    for i = 1:trials
        b = rand(Uniform(0, 1), S)
        if all(-inv(amat) * b .> 0.0)
            push!(bs, b)
        end
    end

    return (amat, bs)
end

function make_cmat(amat, b)
    S = size(amat, 1)
    @assert S == size(amat, 2)

    eq = -inv(amat) * b
    cmat = zeros(S, S)
    for i in 1:size(cmat, 1)
        cmat[i, :] = amat[i, :] .* eq[i]
    end
    return cmat
end

find_eq(amat, b) = -inv(amat) * b

amat, bs = rand_comp(15, 0.7, trials = 1000000)

for b in bs
    cmat = make_cmat(amat, b)
    FoodWebs.plot_eigs(cmat)
end

brange = [bs[1] + b for b in linspace(0, 2, 1000)]
#boxplot([find_eq(amat, b) for b in brange])

colors = reverse(colormap("Grays", length(brange)))
for (i, b) in enumerate(brange)
    cmat = make_cmat(amat, b)
    FoodWebs.plot_eigs(cmat, color = colors[i])
end

subplot(121)
FoodWebs.plot_eigs(make_cmat(amat, brange[1]))
subplot(122)
FoodWebs.plot_eigs(make_cmat(amat, brange[10]))

"""
Timeseries
"""
# I need to make a function that takes the `amat`, `b` and give a function
fweb(t, x, amat, b) = amat * x + b
# this should be zero!
fweb(0, find_eq(amat, brange[1]), amat, brange[1]) .== 0.0

x0 = 1.01 * find_eq(amat, brange[1])
tspan = (0.0, 10.0)
odeprob = ODEProblem((t, x) -> fweb(t, x, amat, brange[1]), x0, tspan)
sol = solve(odeprob)
plot(sol.t, sol.u)
