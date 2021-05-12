using Flux
using Flux.Data.MNIST
using BSON: @load

images = MNIST.images(:test)
labels = MNIST.labels(:test)

@load "MNIST_v2.bson" model

function prediction(img)
    input = Float32.(img)[:]
    pred_vector = model(input) 
    pred_val = findfirst(x -> x == maximum(pred_vector), pred_vector) - 1
    return pred_val
end

images[10]
labels[10]

prediction(images[10])