### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ d9cff120-aba1-11eb-0f83-db8e49ac6bab
md"# Chapter 03 from NNFS book"

# ╔═╡ 4a34dd2f-cbf5-4ac0-8f97-fd4050354547
md"### 2 Layers of Neurons (4 inputs, 3 hidden layers)"

# ╔═╡ ab14105e-a405-4d87-80d7-940ce87ef19a
md"""
**One observation:**
"""

# ╔═╡ fed9705c-b748-4cf9-827a-f10fd8073967
begin	
	# first layer
	input1 = [1, 2, 3, 2.5]

	weight1 = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]

	bias1 = [2, 3, 0.5]

	# second layer
	weight2 = [
		 0.1   -0.14   0.5
		-0.5    0.12  -0.33
		-0.44   0.73  -0.13
	]

	bias2 = [-1, 2, -0.5]

	# calculating layers
	layer1 = weight1 * input1 + bias1
	layer2 = weight2 * layer1 + bias2
end

# ╔═╡ 7ec38404-9380-411d-8b0d-f073ba5c6cbc
md"""
**Multiple observations:**
"""

# ╔═╡ 773a22f6-99c3-4bf5-936b-0b22759038ed
begin	
	# first layer
	inp01 = [
		[1, 2, 3, 2.5] [2, 5, -1, 2] [-1.5, 2.7, 3.3, -0.8]
	]

	w01 = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]

	b01 = [2, 3, 0.5]

	# second layer
	w02 = [
		 0.1   -0.14   0.5
		-0.5    0.12  -0.33
		-0.44   0.73  -0.13
	]

	b02 = [-1, 2, -0.5]

	# calculating layers
	layer01 = w01 * inp01 .+ b01
	layer02 = w02 * layer01 .+ b02
end

# ╔═╡ 6aabd18d-d0a1-4f7e-9019-92d7671e7265
md"### Creating the LayerDense Struct"

# ╔═╡ 50edc678-40c3-483c-8556-666f01553802
begin
	struct LayerDense
		w::Matrix{Float64}
		b::Vector{Float64}
		function LayerDense(n_in::Int, n_out::Int)
			w = 0.01 * randn(n_out, n_in)
			b = zeros(n_out)
			return new(w, b)
		end
	end
	
	(layer::LayerDense)(input) = layer.w * input .+ layer.b
end

# ╔═╡ da9d7fdb-8b5a-466c-9370-654c133e10a3
D = LayerDense(2, 3)

# ╔═╡ 173fe819-779e-4d8e-a2b3-e00fb9363cb2
imp01 = [3.5, 4.3]

# ╔═╡ c14fce98-4470-4153-ba62-75521c3e5ed8
imp02 = [[3.5, 4.3] [2.4, -2.8]]

# ╔═╡ f8915121-453b-49bc-83ed-07ea13541160
D(imp01)

# ╔═╡ ffe4c9ac-bbbf-49e0-b882-b6a8ed096622
D(imp02)

# ╔═╡ ae0e3918-c07e-40a5-880d-15cec3afce71
D(randn(2))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─d9cff120-aba1-11eb-0f83-db8e49ac6bab
# ╟─4a34dd2f-cbf5-4ac0-8f97-fd4050354547
# ╟─ab14105e-a405-4d87-80d7-940ce87ef19a
# ╠═fed9705c-b748-4cf9-827a-f10fd8073967
# ╟─7ec38404-9380-411d-8b0d-f073ba5c6cbc
# ╠═773a22f6-99c3-4bf5-936b-0b22759038ed
# ╟─6aabd18d-d0a1-4f7e-9019-92d7671e7265
# ╠═50edc678-40c3-483c-8556-666f01553802
# ╠═da9d7fdb-8b5a-466c-9370-654c133e10a3
# ╠═173fe819-779e-4d8e-a2b3-e00fb9363cb2
# ╠═c14fce98-4470-4153-ba62-75521c3e5ed8
# ╠═f8915121-453b-49bc-83ed-07ea13541160
# ╠═ffe4c9ac-bbbf-49e0-b882-b6a8ed096622
# ╠═ae0e3918-c07e-40a5-880d-15cec3afce71
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
