### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ be7baa4f-6871-4d74-b4d2-1c2534cdfa17
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
md"# Layer_dense"

# ╔═╡ 45d31083-883b-4e84-913e-c040c6fdfd5f
md"""
>No futuro, podemos pensar em novas formas de fazer esse objeto. Por exemplo, passando a matriz W como input construtor do layer.
"""

# ╔═╡ e22e7c1c-e1c4-4ae3-96c1-9cbd71fcc999
md"""
### Row Orientation
"""

# ╔═╡ dc76be1b-292e-4545-887e-1e62faf8e1b6
md"""
### Column Orientation
"""

# ╔═╡ fe237f5f-8f90-429a-8ddc-e5933b8fd808
begin
	Random.seed!(0)
	
	struct LayerDense
		w::Matrix{Float64}
		b::Vector{Float64}
		σ::Function
	
		function LayerDense(n_in::Int, n_out::Int; σ::Function = identity)
			w = 0.01 .* randn(n_out, n_in)
			b = zeros(n_out) 
			return new(w, b, σ)
		end
	end

	function (L::LayerDense)(input)
		W, b, σ = L.w, L.b, L.σ
		σ( W*input .+ b' )
	end
end

# ╔═╡ c99c7f4f-7e62-4036-ba08-4478e5480d19
md"### Adding new sigmoid functions"

# ╔═╡ 58265283-a36a-4c83-9101-3387165da021
begin
	logistic(x::Float64) = 1 / (1 + ℯ^(-x))
	logistic(x::AbstractVecOrMat{T}) where {T<:Float64} = logistic.(x)
	
	ReLU(x::Float64) = max(0, x)
	ReLU(x::AbstractVecOrMat{T}) where {T<:Float64} = ReLU.(x)
	
	softplus(x::Float64) = log((1+ℯ^x), ℯ)
	softplus(x::AbstractVecOrMat{T}) where {T<:Float64} = softplus.(x)
end

# ╔═╡ d904c2ae-e035-40be-b46b-79ffd500f284
# a few sigmoids options: 
# tanh, atan, ReLU, logistic, softplus, softmax
D = Layer_dense(2,3, σ = logistic) 

# ╔═╡ 00c04f4e-e883-46af-a51c-9fd5f227e685
D([1 4; 2 3; 3 5])

# ╔═╡ d0c91a1c-8798-4985-978b-4c47d1abc212
E = LayerDense(2, 3, σ = logistic)

# ╔═╡ 3de0841e-f0f3-436c-a891-dd4af32d2af6
#E(randn(4,3))
E([1 4; 2 3; 3 5]')

# ╔═╡ 92661b11-5e13-4c8c-918b-1b85f49d583d
(
	by_row = D([1 4; 2 3; 3 5]), 
	# para a orientacao coluna, o input precisa ser transposto
	by_col = E([1 4; 2 3; 3 5]') 
)

# ╔═╡ 04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
function softmax(x::Vector{T}) where {T<:Real}
	m = maximum(x)
	exp_val = exp.(x .- m)
	s = sum(exp_val)
	exp_val ./ s
end

# ╔═╡ a7898ffd-1acb-4220-bbc8-3b428b95db85
function softmax(x::Matrix{T}) where {T<:Real}
	m = maximum.(eachcol(x))
	exp_val = exp.(x .- m')	
	s = sum(eachrow(exp_val))
	return exp_val ./ s' 
end

# ╔═╡ c116a1e4-fd4d-446f-bdea-02301f43b2c0
md"**Aplying some sigmoid functions**"

# ╔═╡ beff8a41-2249-4454-9f2e-b1766e1f3a66
F = Layer_dense(2, 3, σ = softmax)

# ╔═╡ 07c28197-fb7c-4f65-b742-2acd50da1d67
F([1 4; 2 3; 3 5])

# ╔═╡ 963f6276-2440-4bfe-ae0f-539d8bfae0a2
G = LayerDense(2, 3, σ = softmax)

# ╔═╡ f4b146c1-af4a-4c3e-860d-1f92c4c13a6e
( G([1 4; 2 3; 3 5]'), sum(eachrow(G([1 4; 2 3; 3 5]'))) )

# ╔═╡ Cell order:
# ╟─c4ea1432-abb4-11eb-1cf5-edac0735d67d
# ╟─45d31083-883b-4e84-913e-c040c6fdfd5f
# ╟─e22e7c1c-e1c4-4ae3-96c1-9cbd71fcc999
# ╠═be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# ╠═d904c2ae-e035-40be-b46b-79ffd500f284
# ╠═00c04f4e-e883-46af-a51c-9fd5f227e685
# ╟─dc76be1b-292e-4545-887e-1e62faf8e1b6
# ╠═fe237f5f-8f90-429a-8ddc-e5933b8fd808
# ╠═d0c91a1c-8798-4985-978b-4c47d1abc212
# ╠═3de0841e-f0f3-436c-a891-dd4af32d2af6
# ╠═92661b11-5e13-4c8c-918b-1b85f49d583d
# ╟─c99c7f4f-7e62-4036-ba08-4478e5480d19
# ╠═58265283-a36a-4c83-9101-3387165da021
# ╠═04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
# ╠═a7898ffd-1acb-4220-bbc8-3b428b95db85
# ╟─c116a1e4-fd4d-446f-bdea-02301f43b2c0
# ╠═beff8a41-2249-4454-9f2e-b1766e1f3a66
# ╠═07c28197-fb7c-4f65-b742-2acd50da1d67
# ╠═963f6276-2440-4bfe-ae0f-539d8bfae0a2
# ╠═f4b146c1-af4a-4c3e-860d-1f92c4c13a6e
