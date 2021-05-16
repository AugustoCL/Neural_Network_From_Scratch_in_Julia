### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ a5ec2a7e-2dab-46c9-9013-9e7adbd44d78
using Random, Statistics, ForwardDiff

# ╔═╡ 61709a7e-18c5-4017-8e84-6eaf16ba210d
md"# Tipos e Funções"

# ╔═╡ bd149090-b5e2-11eb-34e9-ef542421d474
begin
	Random.seed!(0)
	
	struct LayerDense{T<:Real}
		W::Matrix{T}
		b::Vector{T}
		σ::Function
	
		function LayerDense(W::Matrix{T}, b::Vector{T}, σ::Function = identity) where {T<:Real}
			return new{T}(W, b, σ)
		end
	end
	
	function LayerDense(W::Matrix{T}, b::Vector{S}, σ::Function = identity) where {T<:Real, S<:Real}
		R = promote_type(T, S)
		return LayerDense(Matrix{R}(W), Vector{R}(b), σ)
	end
	
	function LayerDense(W::Matrix{<:Real}, σ::Function = identity)
		b = zeros(size(W)[1])
		return LayerDense(W, b, σ)
	end
	
	function LayerDense(n_in::Int, n_out::Int, σ::Function = identity)
		W = 0.01 .* randn(n_out, n_in)
		b = zeros(n_out) 
		return LayerDense(W, b, σ)
	end
	
	function (L::LayerDense)(input::AbstractVector{<:Real})
		W, b, σ = L.W, L.b, L.σ
		return σ(W * input + b)
	end
	
	function (L::LayerDense)(input::AbstractMatrix{<:Real})
		W, b, σ = L.W, L.b, L.σ
		return σ(W * input .+ b)
	end
end

# ╔═╡ 92a9ff09-50f1-42d1-8b12-c37771610d84
begin
	function σ(x::Real) 
		t = exp(-abs(x))
		return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
	end	
	
	σ(x::AbstractVecOrMat{T}) where {T<:Real} = σ.(x)
	
	const sigmoid = σ
end

# ╔═╡ 468eb0e4-5c6f-49c6-9589-533071bb1271
begin
	relu(x::Real) = max(zero(x), x) # zero function keep the type of x
	relu(x::AbstractVecOrMat{<:Real}) = relu.(x)
	
	#log1p(x) is the seme as log(1 + x)
	softplus(x::Real) = ifelse(x > 0, x + log(1 + exp(-x)), log(1 + exp(x)))
	softplus(x::AbstractVecOrMat{<:Real}) = softplus.(x)
	
	function softmax(x::Vector{T}) where {T<:Real}
		m = maximum(x)
		exp_val = exp.(x .- m)
		s = sum(exp_val)
		return exp_val ./ s
	end
	function softmax(x::Matrix{T}) where {T<:Real}
		m = maximum(x, dims = 1)
		exp_val = exp.(x .- m)
		s = sum(exp_val, dims = 1)
		return exp_val ./ s
	end
end

# ╔═╡ 65358f5c-04a5-4618-a78e-1fd02d72060d
begin
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
end

# ╔═╡ c66c7488-8667-4694-915e-b022488ccd8b
begin
	struct Chain{T<:Tuple}
		layers::T
		Chain(layers...) = new{typeof(layers)}(layers)		
	end
	
	function (L::Chain)(input::AbstractVecOrMat{<:Real})
		return ∘(reverse(L.layers)...)(input)
	end
end

# ╔═╡ bd4c6ca8-93a2-4344-b598-fa533ad7e169
function params(C::Chain)
	p = []
	for layer in C.layers
		#push!(p, layer.W[:], layer.b )
		push!(p, layer.W, layer.b )
	end
	p #return vcat(p...)
end

# ╔═╡ b10cf718-528a-4433-9ae4-25ca78256ec8
md"# Exemplos"

# ╔═╡ 5fefc40a-dbc9-42e5-b9ea-bb2eed97a805
begin
	Random.seed!(1998)
	
	L1 = LayerDense(10, 5, σ)
	L2 = LayerDense(5, 2, softmax)
	
	r = rand(10)
		
	m = Chain(L1, L2)				# C1 = L2(L1(r))
	
	label = [1.0, 0.0]
	
	#function loss(p::Vector) 
		# atualiza o modelo com os parametros p
		#update(m, p)
		# aplica na função de erro
		#crossentropy(m(r), label)
	#end
	
	#∇G = x -> ForwardDiff.gradient(loss, x); # g = ∇f
	
	p = params(m)
	# for iter in iteracoes
		# calcula o loss (crossentropy)
		#loss(r, label)
		# acha o gradiente negativo dos params (p) com o ForwardDiff
		#grad = ∇G(m(r))
		# atualiza o parametro (p) substraindo o gradiente negativo
		#p .-= grad
	# end
end

# ╔═╡ Cell order:
# ╟─61709a7e-18c5-4017-8e84-6eaf16ba210d
# ╠═a5ec2a7e-2dab-46c9-9013-9e7adbd44d78
# ╠═bd149090-b5e2-11eb-34e9-ef542421d474
# ╠═92a9ff09-50f1-42d1-8b12-c37771610d84
# ╠═468eb0e4-5c6f-49c6-9589-533071bb1271
# ╠═65358f5c-04a5-4618-a78e-1fd02d72060d
# ╠═bd4c6ca8-93a2-4344-b598-fa533ad7e169
# ╠═c66c7488-8667-4694-915e-b022488ccd8b
# ╟─b10cf718-528a-4433-9ae4-25ca78256ec8
# ╠═5fefc40a-dbc9-42e5-b9ea-bb2eed97a805
