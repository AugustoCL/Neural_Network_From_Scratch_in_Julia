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
		σ.( (input * W') .+ b )
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

# ╔═╡ 58265283-a36a-4c83-9101-3387165da021
begin
	logistic(x::Float64) = 1 / (1 + ℯ^(-x))
	ReLU(x::Float64) = max(0, x)
	softplus(x::Float64) = log((1+ℯ^x), ℯ)
end

# ╔═╡ d904c2ae-e035-40be-b46b-79ffd500f284
D = Layer_dense(2,3, σ = logistic) # sigmoids: tanh, atan, ReLU, logistic, softplus 

# ╔═╡ 00c04f4e-e883-46af-a51c-9fd5f227e685
D([1 4; 2 3; 3 5])

# ╔═╡ dc76be1b-292e-4545-887e-1e62faf8e1b6
md"""
### Column Orientation
"""

# ╔═╡ fe237f5f-8f90-429a-8ddc-e5933b8fd808
begin
	Random.seed!(0)
	
	struct Layer_dense_v02
		w::Matrix{Float64}
		b::Vector{Float64}
		σ::Function
	
		function Layer_dense_v02(n_in::Int, n_out::Int; σ::Function = identity)
			w = 0.01 .* randn(n_out, n_in)
			b = zeros(n_out) 
			return new(w, b, σ)
		end
	end

	function (L::Layer_dense_v02)(input)
		W, b, σ = L.w, L.b, L.σ
		σ.( (W * input) .+ b' )
	end
end

# ╔═╡ d0c91a1c-8798-4985-978b-4c47d1abc212
E = Layer_dense_v02(2, 3, σ = logistic)

# ╔═╡ 3de0841e-f0f3-436c-a891-dd4af32d2af6
#E(randn(4,3))
E([1 4; 2 3; 3 5]')

# ╔═╡ 92661b11-5e13-4c8c-918b-1b85f49d583d
(
	by_row = D([1 4; 2 3; 3 5]), 
	# para a orientacao coluna, o input precisa ser transposto
	by_col = E([1 4; 2 3; 3 5]') 
)

# ╔═╡ Cell order:
# ╟─c4ea1432-abb4-11eb-1cf5-edac0735d67d
# ╟─45d31083-883b-4e84-913e-c040c6fdfd5f
# ╟─e22e7c1c-e1c4-4ae3-96c1-9cbd71fcc999
# ╠═be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# ╠═58265283-a36a-4c83-9101-3387165da021
# ╠═d904c2ae-e035-40be-b46b-79ffd500f284
# ╠═00c04f4e-e883-46af-a51c-9fd5f227e685
# ╟─dc76be1b-292e-4545-887e-1e62faf8e1b6
# ╠═fe237f5f-8f90-429a-8ddc-e5933b8fd808
# ╠═d0c91a1c-8798-4985-978b-4c47d1abc212
# ╠═3de0841e-f0f3-436c-a891-dd4af32d2af6
# ╠═92661b11-5e13-4c8c-918b-1b85f49d583d
