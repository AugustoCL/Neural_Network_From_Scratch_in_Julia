### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 9af04825-842b-4794-a317-6f1b78a64fb5
begin
	using Statistics, CSV, DataFrames
	using ForwardDiff
end

# ╔═╡ be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# Row Orientation
begin
	using Random
	Random.seed!(0)
	
	struct Layer_dense
		w::Matrix{Float64}
		b::Matrix{Float64}
		σ::Function
	
		function Layer_dense(n_in::Int, n_out::Int; σ::Function = identity)
			w = 0.01 .* randn(n_out, n_in)
			b = reshape(zeros(n_out), (1, n_out)) 
			return new(w, b, σ)
		end
	end

	function (L::Layer_dense)(input) 
		W, b, σ = L.w, L.b, L.σ
		σ( (input * W') .+ b )
	end
end

# ╔═╡ c4ea1432-abb4-11eb-1cf5-edac0735d67d
md"# LayerDense"

# ╔═╡ 45d31083-883b-4e84-913e-c040c6fdfd5f
md"""
>No futuro, podemos pensar em novas formas de fazer esse objeto. Por exemplo, passando a matriz W como input construtor do layer.
"""

# ╔═╡ dc76be1b-292e-4545-887e-1e62faf8e1b6
md"""
### Struct by Column Orientation
"""

# ╔═╡ fe237f5f-8f90-429a-8ddc-e5933b8fd808
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

# ╔═╡ 4e6d165a-9707-4029-af4a-2356b0fe9c41
md"#### Example using the struct LayerDense"

# ╔═╡ 5d3ae0e6-7c5d-48b7-9845-d204a1df9934
W = rand(3, 4)

# ╔═╡ e85bc0f7-65ab-4a41-a4ba-e63c8b7c8184
B = LayerDense(W) # Layer de 4 inputs e 3 outputs

# ╔═╡ 142bb8f9-4188-46d2-94a2-8407d633c496
begin
	obs01 = [2, 3, 4, 5]
	obs02 = [2, 3, 4, 5]
	input = [obs01 obs02] # Each observation must be on a colunm
end 

# ╔═╡ 2377f014-0bd0-457b-b9e9-3ace1ad7c331
B(input) # Each column is an output

# ╔═╡ c99c7f4f-7e62-4036-ba08-4478e5480d19
md"#### Adding sigmoid functions"

# ╔═╡ 96ce3aa7-9261-4e60-93ce-4d9321657d85
md"""
Função **Logística**: \
$$\;\;\;\;\;$$ $$\sigma(x) = {1 \over 1 + e^{-x}}$$ 

Na implementação via código é necessário fazer uma adaptação que evita problemas numéricos de *Overflow* resultando na seguinte função: \
$$\sigma(x) = \left\{
  \begin{array}{lr}
    {1 \over 1 + e^{-|x|}}, & x \ge 0\\
    {e^{-|x|} \over 1 + e^{-|x|}}, & x < 0
  \end{array}
\right.$$ \


Para isso, utilizamos o módulo de $$x$$ na expressão $$e^{-|x|}$$ que possui a seguinte propriedade: \
$$e^{-|x|} = \left\{
  \begin{array}{lr}
    e^{-x}, & x \ge 0\\
    e^x, & x < 0
  \end{array}
\right.$$ 

Com o modulo, quando x é positivo ou zero ($$x ≥ 0$$) a adaptação resulta na própria função logística, mas quando x é negativo são necessários os seguintes passos: \
$$Se \space x \ge 0:$$  $$\;\;\;\;\;$$  $$\sigma(x) = {1 \over 1 + e^{-|x|}}$$ \
$$Se \space x < 0:$$ $$\;\;\;\;\;$$ $$1 = e^{-x} ⋅ e^x → {1 \over 1 + e^{-x}} = {e^{-x} ⋅ e^x \over e^{-x} ⋅ e^x + e^{-x}} = {e^{-x} ⋅ e^x \over e^{-x}(e^x + 1)} = {e^x \over 1 + e^x} = {e^{-|x|} \over 1 + e^{-|x|}}$$
"""

# ╔═╡ 58265283-a36a-4c83-9101-3387165da021
begin
	function σ(x::Real) 
		t = exp(-abs(x))
		return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
	end	
	
	σ(x::AbstractVecOrMat{T}) where {T<:Real} = σ.(x)
	
	const sigmoid = σ
end

# ╔═╡ fc1ddc84-62dc-4af1-9b9e-9d37f581a252
md"""
#### ReLU function
$$ReLU(x) = \left\{
  \begin{array}{lr}
    x, & x \ge 0\\
    0, & x < 0
  \end{array}
\right.$$
$$Or$$
$$ReLU(x) = max(0, x)$$
"""

# ╔═╡ 45bafada-34ec-4c41-b0a2-000f35ebffef
begin
	relu(x::Real) = max(zero(x), x) # zero function keep the type of x
	relu(x::AbstractVecOrMat{<:Real}) = relu.(x)
end

# ╔═╡ c85cd202-7223-4c46-80ef-afd750f28ead
md"""
$$softplus(x) = ln(1 + e^x)$$
$$For \space x > 0:$$
$$1 = e^{-x} ⋅ e^x → ln(1 + e^x) = ln(e^{-x} ⋅ e^x + e^x) = ln(e^x(e^{-x} + 1)) = ln(e^x) + ln(1 + e^{-x})$$
$$softplus(x) = x + ln(1 + e^{-x})$$
"""

# ╔═╡ d6fce490-1498-469d-8ecc-55d0a71d1565
begin
	#log1p(x) is the seme as log(1 + x)
	softplus(x::Real) = ifelse(x > 0, x + log(1 + exp(-x)), log(1 + exp(x)))
	softplus(x::AbstractVecOrMat{<:Real}) = softplus.(x)
end

# ╔═╡ 04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
function softmax(x::Vector{T}) where {T<:Real}
	m = maximum(x)
	exp_val = exp.(x .- m)
	s = sum(exp_val)
	return exp_val ./ s
end

# ╔═╡ a7898ffd-1acb-4220-bbc8-3b428b95db85
function softmax(x::Matrix{T}) where {T<:Real}
	m = maximum(x, dims = 1)
	exp_val = exp.(x .- m)
	s = sum(exp_val, dims = 1)
	return exp_val ./ s
end

# ╔═╡ c116a1e4-fd4d-446f-bdea-02301f43b2c0
md"##### Example with sigmoid functions"

# ╔═╡ d0c91a1c-8798-4985-978b-4c47d1abc212
C = LayerDense(2, 3, σ)

# ╔═╡ 3de0841e-f0f3-436c-a891-dd4af32d2af6
# C(randn(2, 3))
C([[1, 4] [2, 3] [3, 5]])

# ╔═╡ 963f6276-2440-4bfe-ae0f-539d8bfae0a2
D = LayerDense(2, 3, softmax)

# ╔═╡ 56191b0c-fe49-4ed4-9ca7-2e22e7b028ba
md"""
#### Cost Function
"""

# ╔═╡ 7b5a6859-a9e1-48b1-a0fd-1cde06249d03
begin
	function xlogy(x::Real, y::Real) 
		result = x * log(y)
		ifelse(iszero(x) && !isnan(y), zero(result), result)
	end
	
	function crossentropy(ŷ::AbstractVecOrMat{<:Real},
						  y::AbstractVecOrMat{<:Real};
						  dims::Int = 1,
						  agg::Function = mean)
    	agg(.-sum(xlogy.(y, ŷ), dims = dims))
	end
end

# ╔═╡ 4b070348-fdbc-46cd-85f0-29b64432ae21
begin
	outp2 = D([1 4; 2 3; 3 5]')
	target2 = [0, 1, 0]
	crossentropy(outp2, target2)
end

# ╔═╡ e3ec0333-14ad-42a8-9cd3-4e3937dc5ff0
begin
	outp1 = [0.7 0.1 0.2
			 0.1 0.5 0.4
	   		 0.02 0.9 0.08]'
	target1 = [1 0 0
			   0 1 0
		  	   0 1 0]'
	crossentropy(outp1, target1)
end

# ╔═╡ 8f3cbc01-53ba-4ff0-8245-564f8fee53f2
md"##### Importing data from nnfs python package"

# ╔═╡ 099f5fa8-6687-4c77-9f61-119d75377bab
data = CSV.read("spiral_data.csv", DataFrame)

# ╔═╡ b5981a68-16e7-451f-9e13-d74b2b78a958
md"#### Chain type, params(), gradient()"

# ╔═╡ d2fa0422-7f39-4365-8f3b-bfa6997ae0ae
begin
	Random.seed!(1998)
	
	L1 = LayerDense(10, 5, σ)
	L2 = LayerDense(5, 2, softmax)
	
	output = L2(L1(rand(10)))
end

# ╔═╡ 72f68243-8ae1-465c-854d-a76c07c5e346
function params(Layers)
	p = []
	for L in Layers
		push!(p, [ L.W[:]; L.b ] )
	end
	return vcat(p...)
end

# ╔═╡ e84cff98-1bea-42cb-8e8d-6f5ef08226f4
begin 
	Θ = params([L1, L2])
	
	U = ( 
		size(L1.W, 1)*size(L1.W, 2), length(L1.b),
		size(L2.W, 1)*size(L2.W, 2), length(L2.b),
	)
	U = cumsum(U)
	
	L1w = reshape( Θ[ 1:U[1] ], size(L1.W, 1), size(L1.W, 2))
	
	L1b = Θ[ U[1]+1:U[2] ]
	
	L2w = reshape( Θ[ U[2]+1:U[3] ], size(L2.W, 1), size(L2.W, 2))
	
	L2b = Θ[ U[3]+1:U[4] ]
	
	( length(Θ), U, (L1w, L1b, L2w, L2b) )
end

# ╔═╡ a5a511f7-38f0-4074-9192-7f11d611bf42
begin
	
	struct Chain
		Layers::Vector{LayerDense{Float64}}
		Wl::Vector{Int}
		Wc::Vector{Int}
		b::Vector{Int}
		
		function Chain(L...)
			Wl = Int[]
			Wc = Int[]
			b = Int[]
			for (i, l) in enumerate(L)
				Wl[i], Wc[i] = size(l.W)
				b[i] = length(l.b)
			end
			return new{}(L, Wl, Wc, b)
		end

	end

end

# ╔═╡ Cell order:
# ╟─c4ea1432-abb4-11eb-1cf5-edac0735d67d
# ╟─45d31083-883b-4e84-913e-c040c6fdfd5f
# ╟─dc76be1b-292e-4545-887e-1e62faf8e1b6
# ╠═fe237f5f-8f90-429a-8ddc-e5933b8fd808
# ╟─4e6d165a-9707-4029-af4a-2356b0fe9c41
# ╠═5d3ae0e6-7c5d-48b7-9845-d204a1df9934
# ╠═e85bc0f7-65ab-4a41-a4ba-e63c8b7c8184
# ╠═142bb8f9-4188-46d2-94a2-8407d633c496
# ╠═2377f014-0bd0-457b-b9e9-3ace1ad7c331
# ╟─c99c7f4f-7e62-4036-ba08-4478e5480d19
# ╟─96ce3aa7-9261-4e60-93ce-4d9321657d85
# ╠═58265283-a36a-4c83-9101-3387165da021
# ╟─fc1ddc84-62dc-4af1-9b9e-9d37f581a252
# ╠═45bafada-34ec-4c41-b0a2-000f35ebffef
# ╟─c85cd202-7223-4c46-80ef-afd750f28ead
# ╠═d6fce490-1498-469d-8ecc-55d0a71d1565
# ╠═04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
# ╠═a7898ffd-1acb-4220-bbc8-3b428b95db85
# ╟─c116a1e4-fd4d-446f-bdea-02301f43b2c0
# ╠═d0c91a1c-8798-4985-978b-4c47d1abc212
# ╠═3de0841e-f0f3-436c-a891-dd4af32d2af6
# ╠═963f6276-2440-4bfe-ae0f-539d8bfae0a2
# ╟─56191b0c-fe49-4ed4-9ca7-2e22e7b028ba
# ╠═7b5a6859-a9e1-48b1-a0fd-1cde06249d03
# ╠═4b070348-fdbc-46cd-85f0-29b64432ae21
# ╠═e3ec0333-14ad-42a8-9cd3-4e3937dc5ff0
# ╟─8f3cbc01-53ba-4ff0-8245-564f8fee53f2
# ╟─099f5fa8-6687-4c77-9f61-119d75377bab
# ╠═9af04825-842b-4794-a317-6f1b78a64fb5
# ╟─b5981a68-16e7-451f-9e13-d74b2b78a958
# ╠═d2fa0422-7f39-4365-8f3b-bfa6997ae0ae
# ╠═72f68243-8ae1-465c-854d-a76c07c5e346
# ╠═e84cff98-1bea-42cb-8e8d-6f5ef08226f4
# ╠═a5a511f7-38f0-4074-9192-7f11d611bf42
# ╟─be7baa4f-6871-4d74-b4d2-1c2534cdfa17
