### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ d9cff120-aba1-11eb-0f83-db8e49ac6bab
md"# Chapter 03 from NNFS book"

# ╔═╡ 4a34dd2f-cbf5-4ac0-8f97-fd4050354547
md"### 2 Layers of Neurons (4 inputs, 3 hidden layers)"

# ╔═╡ 773a22f6-99c3-4bf5-936b-0b22759038ed
begin	
	# first layer
	inp01 = [
		   1   2   3  2.5
		   2   5  -1    2 
		-1.5 2.7 3.3 -0.8
	]

	w01 = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]

	b01 = [2 3 0.5]

	# second layer
	w02 = [
		 0.1   -0.14   0.5
		-0.5    0.12  -0.33
		-0.44   0.73  -0.13
	]

	b02 = [-1 2 -0.5]

	# calculating layers
	layer01 = inp01 * w01' .+ b01
	layer02 = layer01 * w02' .+ b02
end

# ╔═╡ 6aabd18d-d0a1-4f7e-9019-92d7671e7265
md"### Creating the layer_dense Struct"

# ╔═╡ 50edc678-40c3-483c-8556-666f01553802
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

# ╔═╡ da9d7fdb-8b5a-466c-9370-654c133e10a3
D = Layer_dense(2,3)

# ╔═╡ ae0e3918-c07e-40a5-880d-15cec3afce71
D(randn(2,2))

# ╔═╡ Cell order:
# ╟─d9cff120-aba1-11eb-0f83-db8e49ac6bab
# ╟─4a34dd2f-cbf5-4ac0-8f97-fd4050354547
# ╠═773a22f6-99c3-4bf5-936b-0b22759038ed
# ╟─6aabd18d-d0a1-4f7e-9019-92d7671e7265
# ╠═50edc678-40c3-483c-8556-666f01553802
# ╠═da9d7fdb-8b5a-466c-9370-654c133e10a3
# ╠═ae0e3918-c07e-40a5-880d-15cec3afce71
