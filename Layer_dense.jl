### A Pluto.jl notebook ###
# v0.19.18

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
	
	struct LayerDense{T<:Real, F<:Function}
		W::Matrix{T}
		b::Vector{T}
		σ::F

		function LayerDense(W::Matrix{T}, 
							b::Vector{T}, 
							σ::Function = identity) where {T<:Real}                
			@assert size(W, 1) == length(b)

			new{T, typeof(σ)}(W, b, σ)
		end
	end

	function LayerDense(W::Matrix{T}, 
						b::Vector{S}, 
						σ::Function = identity) where {T<:Real, S<:Real}
		R = promote_type(T, S)
		return LayerDense(Matrix{R}(W), Vector{R}(b), σ)
	end

	function LayerDense(W::Matrix{T}, σ::Function = identity) where {T<:Real}
		b = zeros(T, size(W, 1))
		return LayerDense(W, b, σ)
	end

	function LayerDense(n_in::Int, n_out::Int, σ::Function = identity)
		W = 0.01 * randn(n_out, n_in)
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
function params(layers)
	p = []
	for layer in layers
		push!(p, layer.W[:], layer.b )
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
	
	struct Chain2
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

# ╔═╡ 154e4560-0769-429e-89bd-3b0392209d6d
typeof( (1.3,2.12) )

# ╔═╡ 74ccd8c6-f513-4178-a41f-7b70835516e7
begin
	struct Chain{T<:Tuple}
		layers::T
		Chain(layers...) = new{typeof(layers)}(layers)		
	end
	
	function (L::Chain)(input::AbstractVecOrMat{<:Real})
		return ∘(reverse(L.layers)...)(input)
	end
end

# ╔═╡ 4504eaff-0c86-4606-a410-c94a8e8a653e
begin
	#L1 = LayerDense(10, 5, σ)
	#L2 = LayerDense(5, 2, softmax)
	
	C1 = Chain(L1, L2)
	typeof(C1.layers)
end

# ╔═╡ ac678c79-a804-44e0-80d8-15a40dff327d
begin
	p = []
	for layer in C1.layers
		push!(p, layer.W[:], layer.b )
	end
	vcat(p...)
end

# ╔═╡ 4b019add-7eb4-496a-b0ef-3c3eee9d85bd
begin
	C1(rand(10,12))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.2.2"
ForwardDiff = "~0.10.19"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

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
# ╠═ac678c79-a804-44e0-80d8-15a40dff327d
# ╠═72f68243-8ae1-465c-854d-a76c07c5e346
# ╠═e84cff98-1bea-42cb-8e8d-6f5ef08226f4
# ╠═a5a511f7-38f0-4074-9192-7f11d611bf42
# ╠═154e4560-0769-429e-89bd-3b0392209d6d
# ╠═74ccd8c6-f513-4178-a41f-7b70835516e7
# ╠═4504eaff-0c86-4606-a410-c94a8e8a653e
# ╠═4b019add-7eb4-496a-b0ef-3c3eee9d85bd
# ╟─be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
