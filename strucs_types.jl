using Statistics

# Layer Dense -------------------------------------------------------------------------------
abstract type Layer end 

struct LayerDense{T<:Real, F} <: Layer
    W::Matrix{T}
    b::Vector{T}
    σ::F
end

function LayerDense(W::AbstractMatrix{T}, b::AbstractVector{S}, σ::F = identity) where {T<:Real, S<:Real, F}
    R = promote_type(T, S)
    LayerDense{R, F}(W, b, σ)
end

function LayerDense(W::AbstractMatrix{<:Real}, σ::Function = identity)
    b = zeros(size(W, 1))
    LayerDense(W, b, σ)
end

function LayerDense(in::Int, out::Int, σ::Function = identity)
    W = randn(out, in)
    b = zeros(out) 
    LayerDense(W, b, σ)
end

function (L::LayerDense)(input::AbstractVector{<:Real})
    W, b, σ = L.W, L.b, L.σ
    σ(W * input + b)
end

function (L::LayerDense)(input::AbstractMatrix{<:Real})
    W, b, σ = L.W, L.b, L.σ
    σ(W * input .+ b)
end


# Chain struct ------------------------------------------------------------------------------
struct Chain{T<:Tuple}
    layers::T
    Chain(layers::Layer...) = new{typeof(layers)}(layers)
end

(L::Chain)(input::AbstractVecOrMat{<:Real}) = reduce(∘, reverse(L.layers))(input)


# Sigmoid Functions -------------------------------------------------------------------------

function σ(x::Real) 
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end	

σ(x::AbstractVecOrMat{T}) where {T<:Real} = σ.(x)

const sigmoid = σ

relu(x::Real) = max(zero(x), x) # zero function keep the type of x
relu(x::AbstractVecOrMat{<:Real}) = relu.(x)

# softplus - where log1p(x) is the seme as log(1 + x)
softplus(x::Real) = ifelse(x > 0, x + log(1 + exp(-x)), log(1 + exp(x)))
softplus(x::AbstractVecOrMat{<:Real}) = softplus.(x)

function softmax(x::AbstractVector{T}) where {T<:Real}
    m = maximum(x)
    e = exp.(x .- m)
    s = sum(e)
    e ./ s
end

function softmax(x::AbstractMatrix{T}) where {T<:Real}
    m = maximum(x, dims=1)
    e = exp.(x .- m)
    s = sum(e, dims=1)
    e ./ s
end


# Cost Functions ----------------------------------------------------------------------------
function xlogy(x::Real, y::Real)
    result = x * log(y)
    ifelse(iszero(x) && !isnan(y), zero(result), result)
end

function crossentropy(ŷ::AbstractVecOrMat{<:Real},
                      y::AbstractVecOrMat{<:Real};
                      dims::Int = 1,
                      agg::Function = Statistics.mean)
    agg(.-sum(xlogy.(y, ŷ), dims = dims))
end
