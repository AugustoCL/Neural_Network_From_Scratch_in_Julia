### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ c4ea1432-abb4-11eb-1cf5-edac0735d67d
md"# Layer_dense"

# ╔═╡ be7baa4f-6871-4d74-b4d2-1c2534cdfa17
begin
	struct Layer_dense
		w::Matrix{Float64}
		b::Matrix{Float64}
		function Layer_dense(n_in::Int, n_out::Int)
			w = 0.01 .* randn(n_in, n_out)
			b = reshape(zeros(n_out), (1, n_out)) 
			return new(w,b)
		end
	end

	(layer::Layer_dense)(input) = (input * layer.w) .+ layer.b
end

# ╔═╡ d904c2ae-e035-40be-b46b-79ffd500f284
D = Layer_dense(2,3)

# ╔═╡ 00c04f4e-e883-46af-a51c-9fd5f227e685
D(randn(2,2))

# ╔═╡ 45d31083-883b-4e84-913e-c040c6fdfd5f
md"""
No futuro, podemos pensar em novas formas de fazer esse objeto. Por exemplo, passando a matriz W como input construtor do layer.
"""

# ╔═╡ Cell order:
# ╟─c4ea1432-abb4-11eb-1cf5-edac0735d67d
# ╠═be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# ╠═d904c2ae-e035-40be-b46b-79ffd500f284
# ╠═00c04f4e-e883-46af-a51c-9fd5f227e685
# ╟─45d31083-883b-4e84-913e-c040c6fdfd5f
